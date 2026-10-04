#!/usr/bin/env python3
"""云边协同 Pipeline: YOLO分流 → 50%裁剪 → 云端DETR → NMS融合 → 评估"""
import json, os, time, argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
from collections import defaultdict

YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt")
CLOUD_SERVER_URL = os.environ.get("CLOUD_SERVER_URL", "http://localhost:8000")
GT_ANNOTATION_PATH = os.environ.get("GT_ANNOTATION_PATH", "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json")
IMAGE_DIR = os.environ.get("IMAGE_DIR", "/app/data/seadronessee/compressed_version/val2017/")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/Salience-DETR/experiments/cloud_edge_50pct_pipeline_results/")
SPLIT_THRESHOLD = 0.7
YOLO_CONF = 0.01
CLOUD_CONF = 0.05
EVAL_CONF = 0.05
SCALE_FACTOR = 0.50
NMS_IOU = 0.5
CHECKPOINT_INTERVAL = 100
EDGE_BASELINE_AP50 = 0.735
CLOUD_FULL_AP50 = 0.858
ORIGINAL_C_AP50 = 0.544
GT_CROP_50_AP50 = 0.794

def load_gt_annotations(gt_path):
    print(f"[加载] GT 标注: {gt_path}")
    with open(gt_path, "r") as f: coco = json.load(f)
    image_map = {img["id"]: img for img in coco["images"]}
    ann_map = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("bbox") and ann["bbox"][2] > 0 and ann["bbox"][3] > 0:
            ann_map[ann["image_id"]].append(ann)
    print(f"  -> {len(image_map)} 张图片, {sum(len(v) for v in ann_map.values())} 个标注")
    return coco, image_map, ann_map

def load_image(p): return np.array(Image.open(p).convert("RGB"))

def scale_bbox_to_gt(b, aw, ah, gw, gh):
    if aw == gw and ah == gh: return list(b)
    sx, sy = gw/aw, gh/ah
    return [b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy]

def xyxy_to_xywh(x1,y1,x2,y2): return [x1,y1,x2-x1,y2-y1]

def compute_iou(b1, b2):
    xa,ya = max(b1[0],b2[0]), max(b1[1],b2[1])
    xb,yb = min(b1[0]+b1[2],b2[0]+b2[2]), min(b1[1]+b1[3],b2[1]+b2[3])
    if xb<=xa or yb<=ya: return 0.0
    inter = (xb-xa)*(yb-ya)
    return inter / (b1[2]*b1[3] + b2[2]*b2[3] - inter)

def apply_nms(dets, iou_thr=NMS_IOU):
    if not dets: return []
    by_cls = defaultdict(list)
    for d in dets: by_cls[d["category_id"]].append(d)
    result = []
    for cls, ds in by_cls.items():
        ds.sort(key=lambda x: -x["score"])
        kept = []
        for d in ds:
            if not any(compute_iou(d["bbox"], k["bbox"]) > iou_thr for k in kept):
                kept.append(d)
        result.extend(kept)
    return result

def run_yolo(model, img_path, conf=YOLO_CONF):
    results = model(img_path, conf=conf, verbose=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0: return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clses = r.boxes.cls.cpu().numpy().astype(int)
    return [{"bbox":[float(xyxy[i][0]),float(xyxy[i][1]),float(xyxy[i][2]),float(xyxy[i][3])],
             "score":float(confs[i]),"class_id":int(clses[i])} for i in range(len(xyxy))]

def compute_crop_region(aw, ah, low_dets, sf=SCALE_FACTOR):
    ax1=[d["bbox"][0] for d in low_dets]; ay1=[d["bbox"][1] for d in low_dets]
    ax2=[d["bbox"][2] for d in low_dets]; ay2=[d["bbox"][3] for d in low_dets]
    cx=(min(ax1)+max(ax2))/2; cy=(min(ay1)+max(ay2))/2
    ss=np.sqrt(sf); cw,ch=int(aw*ss),int(ah*ss)
    x1,y1=int(cx-cw/2),int(cy-ch/2)
    if x1<0: x1=0
    if y1<0: y1=0
    x2,y2=x1+cw,y1+ch
    if x2>aw: x2=aw; x1=max(0,x2-cw)
    if y2>ah: y2=ah; y1=max(0,y2-ch)
    return x1,y1,x2,y2

def img_to_bytes(a):
    buf=BytesIO(); Image.fromarray(a).save(buf,format="JPEG",quality=95); return buf.getvalue()

def send_to_cloud(a, conf=CLOUD_CONF, retries=3):
    ib=img_to_bytes(a)
    for attempt in range(retries):
        try:
            r=requests.post(f"{CLOUD_SERVER_URL}/detect",
                files={"file":("image.jpg",ib,"image/jpeg")},
                data={"conf_threshold":str(conf)},timeout=60)
            if r.status_code==200:
                d=r.json()
                if "results" in d: return [det for item in d["results"] for det in item.get("detections",[])]
                return d.get("detections", d if isinstance(d,list) else [])
            if attempt<retries-1: time.sleep(1)
        except Exception as e:
            if attempt<retries-1: time.sleep(2)
            else: print(f"  [云端失败] {e}"); return []
    return []

def parse_bbox(b):
    if isinstance(b,list): return b[0],b[1],b[2],b[3]
    if isinstance(b,dict): return b["x1"],b["y1"],b["x2"],b["y2"]
    raise ValueError(f"Unknown bbox: {type(b)}")

def evaluate_coco_subset(gt_coco, results):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if not results:
        return {"AP50":0,"AP":0,"AP75":0,"AR":0,"AR1":0,"AR10":0,
                "AP_small":0,"AP_medium":0,"AP_large":0,"AR_small":0,"AR_medium":0,"AR_large":0,
                "stats":[0]*12,"subset_images":0,"subset_anns":0}
    pids={d["image_id"] for d in results}
    si=[img for img in gt_coco["images"] if img["id"] in pids]
    sa=[ann for ann in gt_coco["annotations"] if ann["image_id"] in pids]
    sgt={"images":si,"annotations":sa,"categories":gt_coco["categories"]}
    print(f"\n  Subset GT: {len(si)} 张图片, {len(sa)} 个标注")
    cg=COCO(); cg.dataset=sgt; cg.createIndex()
    cd=cg.loadRes(results)
    ce=COCOeval(cg,cd,"bbox"); ce.evaluate(); ce.accumulate(); ce.summarize()
    return {"AP50":float(ce.stats[1]),"AP":float(ce.stats[0]),"AP75":float(ce.stats[2]),
            "AR":float(ce.stats[8]),"AR1":float(ce.stats[6]),"AR10":float(ce.stats[7]),
            "AP_small":float(ce.stats[3]),"AP_medium":float(ce.stats[4]),"AP_large":float(ce.stats[5]),
            "AR_small":float(ce.stats[9]),"AR_medium":float(ce.stats[10]),"AR_large":float(ce.stats[11]),
            "stats":[float(s) for s in ce.stats],"subset_images":len(si),"subset_anns":len(sa)}

def run_pipeline(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[加载] YOLO 模型: {YOLO_MODEL_PATH}")
    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL_PATH)
    gt_coco, image_map, ann_map = load_gt_annotations(GT_ANNOTATION_PATH)
    image_ids = sorted(ann_map.keys())
    if MAX_IMAGES > 0: image_ids = image_ids[:MAX_IMAGES]
    print(f"[实验] 处理 {len(image_ids)} 张图片 | 分流: {SPLIT_THRESHOLD} | 裁剪: {SCALE_FACTOR:.0%} | NMS: {NMS_IOU}\n")

    all_yolo=[]; edge_only=[]; fused=[]
    processed=fail=img_low=img_none=0
    hi_cnt=lo_cnt=cloud_cnt=nms_rm=0
    t_yolo=t_cloud=0.0; bw_up=bw_full=0
    pc_hi=defaultdict(int); pc_lo=defaultdict(int); pc_cl=defaultdict(int)

    for img_id in tqdm(image_ids, desc="云边协同"):
        ii=image_map[img_id]; ip=os.path.join(IMAGE_DIR, ii["file_name"])
        if not os.path.exists(ip): fail+=1; continue
        try: img=load_image(ip)
        except Exception as e: print(f"  [跳过] {ii['file_name']}: {e}"); fail+=1; continue
        ah,aw=img.shape[:2]; gw,gh=ii["width"],ii["height"]

        # 1. YOLO推理
        t0=time.time(); yd=run_yolo(model,ip,conf=YOLO_CONF); t_yolo+=time.time()-t0
        if len(yd)==0: img_none+=1; processed+=1; continue

        # 转GT空间COCO格式
        yc=[]
        for d in yd:
            x1,y1,x2,y2=d["bbox"]
            gb=scale_bbox_to_gt(xyxy_to_xywh(x1,y1,x2,y2),aw,ah,gw,gh)
            if gb[2]<=0 or gb[3]<=0: continue
            cd={"image_id":img_id,"category_id":d["class_id"]+1,"bbox":gb,"score":d["score"]}
            yc.append(cd)
            if d["score"]>=EVAL_CONF: all_yolo.append(cd)

        # 2. 分流
        hi=[d for d in yc if d["score"]>=SPLIT_THRESHOLD]
        lo=[d for d in yc if EVAL_CONF<=d["score"]<SPLIT_THRESHOLD]
        for d in hi: edge_only.append(d); pc_hi[d["category_id"]]+=1
        for d in lo: pc_lo[d["category_id"]]+=1
        hi_cnt+=len(hi); lo_cnt+=len(lo)

        # 3. 低置信→裁剪50%→云端
        cd_coco=[]
        if lo:
            img_low+=1
            # low_conf GT空间→实际图片空间
            lo_a=[{"bbox":[d["bbox"][0]*(aw/gw),d["bbox"][1]*(ah/gh),
                    d["bbox"][0]*(aw/gw)+d["bbox"][2]*(aw/gw),
                    d["bbox"][1]*(ah/gh)+d["bbox"][3]*(ah/gh)]} for d in lo]
            cr=compute_crop_region(aw,ah,lo_a,SCALE_FACTOR)
            cx1,cy1,cx2,cy2=cr
            crop=img[cy1:cy2,cx1:cx2].copy()
            bw_up+=len(img_to_bytes(crop)); bw_full+=len(img_to_bytes(img))
            t0=time.time(); cd_raw=send_to_cloud(crop,conf=CLOUD_CONF); t_cloud+=time.time()-t0
            for det in cd_raw:
                bx1,by1,bx2,by2=parse_bbox(det["bbox"])
                sc=det.get("score",det.get("confidence",0.0))
                ci=det.get("class_id",det.get("category_id",0))
                if sc<CLOUD_CONF: continue
                gb=scale_bbox_to_gt(xyxy_to_xywh(cx1+bx1,cy1+by1,cx1+bx2,cy1+by2),aw,ah,gw,gh)
                if gb[2]<=0 or gb[3]<=0: continue
                cd_coco.append({"image_id":img_id,"category_id":ci,"bbox":gb,"score":float(sc)})
                pc_cl[ci]+=1
            cloud_cnt+=len(cd_coco)

        # 4. NMS融合
        combined=list(hi)+list(cd_coco)
        pre=len(combined)
        fd=apply_nms(combined,NMS_IOU)
        fd=[d for d in fd if d["score"]>=EVAL_CONF]
        nms_rm+=pre-len(fd)
        fused.extend(fd)
        processed+=1
        if processed%CHECKPOINT_INTERVAL==0:
            with open(os.path.join(OUTPUT_DIR,"checkpoint.json"),"w") as f:
                json.dump({"processed":processed,"all_yolo":all_yolo,"edge_only":edge_only,"fused":fused},f)

    # 5. 评估
    print("\n"+"="*70)
    print(f"云边协同 Pipeline — 50% ROI | {processed}张 | 分流:{SPLIT_THRESHOLD} | NMS:{NMS_IOU}")
    print("="*70)
    print("\n--- (1) All YOLO: 全部YOLO检测 (无分流) ---")
    m_y=evaluate_coco_subset(gt_coco,all_yolo)
    print("\n--- (2) Edge-only: 仅高置信度YOLO ---")
    m_e=evaluate_coco_subset(gt_coco,edge_only)
    print("\n--- (3) Fused: 高置信+云端NMS融合 ---")
    m_f=evaluate_coco_subset(gt_coco,fused)

    print(f"\n{'='*70}\n三组对比\n{'='*70}")
    print(f"  {'指标':<20} {'AllYOLO':>10} {'EdgeOnly':>10} {'Fused':>10}")
    print(f"  {'-'*50}")
    for k,l in [("AP50","AP50"),("AP","AP"),("AP75","AP75"),("AR","AR"),("AR1","AR1"),("AR10","AR10"),
                ("AP_small","AP small"),("AP_medium","AP medium"),("AP_large","AP large"),
                ("AR_small","AR small"),("AR_medium","AR medium"),("AR_large","AR large")]:
        print(f"  {l:<20} {m_y[k]:>10.4f} {m_e[k]:>10.4f} {m_f[k]:>10.4f}")

    print(f"\n{'='*70}\n基线对比 (AP50)\n{'='*70}")
    print(f"  A组 纯边缘:          {EDGE_BASELINE_AP50:.4f}")
    print(f"  All YOLO(pycoco):    {m_y['AP50']:.4f}  (应接近A组)")
    print(f"  Edge-only(>=0.7):    {m_e['AP50']:.4f}  (云端补充前)")
    print(f"  Fused(边缘+云端):    {m_f['AP50']:.4f}  ★ 最终结果")
    print(f"  B组 纯云端DETR:      {CLOUD_FULL_AP50:.4f}")
    print(f"  C组 原始云边:        {ORIGINAL_C_AP50:.4f}")
    print(f"  50%GT裁剪上限:       {GT_CROP_50_AP50:.4f}")
    d_e=m_f["AP50"]-m_e["AP50"]; d_a=m_f["AP50"]-EDGE_BASELINE_AP50; d_c=m_f["AP50"]-ORIGINAL_C_AP50
    print(f"\n  Fused vs Edge-only:  {'+' if d_e>=0 else ''}{d_e:.4f} (云端增益)")
    print(f"  Fused vs A组:        {'+' if d_a>=0 else ''}{d_a:.4f} ({'超过' if d_a>=0 else '低于'}边缘端)")
    print(f"  Fused vs C组:        {'+' if d_c>=0 else ''}{d_c:.4f} (改善幅度)")

    print(f"\n{'='*70}\nPipeline统计\n{'='*70}")
    print(f"  处理: {processed} | 失败: {fail} | 无检测: {img_none}({img_none/max(processed,1)*100:.1f}%)")
    print(f"  有低置信图片: {img_low}({img_low/max(processed,1)*100:.1f}%)")
    print(f"  高置信: {hi_cnt} | 低置信: {lo_cnt} | 云端返回: {cloud_cnt} | NMS去重: {nms_rm} | 融合后: {len(fused)}")
    cr=lo_cnt/max(hi_cnt+lo_cnt,1)*100
    print(f"  云端分流比: {cr:.1f}%")
    cn={1:"swimmer",2:"boat",3:"jetski",4:"life_saving",5:"buoy"}
    print(f"\n  {'类别':<15} {'高置信':>8} {'低置信':>8} {'云端':>8}")
    for cid in sorted(pc_hi.keys()|pc_lo.keys()|pc_cl.keys()):
        print(f"  {cn.get(cid,f'cls{cid}'):<15} {pc_hi.get(cid,0):>8} {pc_lo.get(cid,0):>8} {pc_cl.get(cid,0):>8}")
    if img_low>0:
        au=bw_up/img_low/1024; af=bw_full/img_low/1024; br=bw_up/max(bw_full,1)
        print(f"\n  带宽: 上传{au:.1f}KB vs 全图{af:.1f}KB | 节省{(1-br)*100:.1f}%")
    print(f"  延迟: YOLO {t_yolo/max(processed,1)*1000:.1f}ms/图 | 云端 {t_cloud/max(img_low,1)*1000:.1f}ms/需云端图")

    with open(os.path.join(OUTPUT_DIR,"pipeline_results.json"),"w") as f:
        json.dump({"all_yolo":{k:v for k,v in m_y.items() if k!="stats"},
                   "edge_only":{k:v for k,v in m_e.items() if k!="stats"},
                   "fused":{k:v for k,v in m_f.items() if k!="stats"},
                   "config":{"split":SPLIT_THRESHOLD,"scale":SCALE_FACTOR,"nms":NMS_IOU,
                             "yolo_conf":YOLO_CONF,"cloud_conf":CLOUD_CONF,"processed":processed},
                   "stats":{"img_low":img_low,"img_none":img_none,"hi":hi_cnt,"lo":lo_cnt,
                            "cloud":cloud_cnt,"nms_rm":nms_rm,"fused":len(fused),"cloud_ratio":cr,
                            "yolo_s":t_yolo,"cloud_s":t_cloud},
                   "baselines":{"edge":EDGE_BASELINE_AP50,"cloud":CLOUD_FULL_AP50,
                                "original_c":ORIGINAL_C_AP50,"gt_crop":GT_CROP_50_AP50}},f,indent=2)
    for name,res in [("all_yolo",all_yolo),("edge_only",edge_only),("fused",fused)]:
        with open(os.path.join(OUTPUT_DIR,f"predictions_{name}.json"),"w") as f:
            json.dump(res,f,indent=2)
    ck=os.path.join(OUTPUT_DIR,"checkpoint.json")
    if os.path.exists(ck): os.remove(ck)
    print(f"\n[保存] -> {OUTPUT_DIR}")

if __name__ == "__main__":
    p=argparse.ArgumentParser(description="云边协同 Pipeline (50% ROI)")
    p.add_argument("--max-images",type=int,default=0,help="最多图片数(0=全部)")
    p.add_argument("--server",type=str,default=CLOUD_SERVER_URL,help="云端地址")
    p.add_argument("--split",type=float,default=SPLIT_THRESHOLD,help="分流阈值")
    a=p.parse_args()
    MAX_IMAGES=a.max_images; CLOUD_SERVER_URL=a.server; SPLIT_THRESHOLD=a.split
    run_pipeline(a)
