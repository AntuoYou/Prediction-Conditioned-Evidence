#!/usr/bin/env python3
"""
ROI 50% 全量测试 — 无Padding vs 有Padding 对比 (1547 张图)
目标：验证 50% 裁剪能否让 AP50 超过边缘端基线 0.735
两组对比：
  A: 50% 裁剪无 Padding — crop 直接上传 (短边764px, resize 1.05×)
  B: 50% 裁剪有 Padding — 灰底padding到800px (resize 1.0×)
"""
import json, os, time, argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
from collections import defaultdict

CLOUD_SERVER_URL = os.environ.get("CLOUD_SERVER_URL", "http://localhost:8000")
GT_ANNOTATION_PATH = os.environ.get("GT_ANNOTATION_PATH", "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json")
IMAGE_DIR = os.environ.get("IMAGE_DIR", "/app/data/seadronessee/compressed_version/val2017/")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/Salience-DETR/experiments/roi_50pct_full_results/")
CONF_THRESHOLD = 0.05
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "0"))
SCALE_FACTOR = 0.50
TARGET_SHORT_SIDE = 800
CHECKPOINT_INTERVAL = 100
EDGE_BASELINE_AP50 = 0.735
CLOUD_BASELINE_AP50 = 0.858
SAMPLE_50_NOPAD_AP50 = 0.875
SAMPLE_50_PAD_AP50 = 0.871

def load_gt_annotations(gt_path):
    print(f"[加载] GT 标注: {gt_path}")
    with open(gt_path, "r") as f:
        coco = json.load(f)
    image_map = {img["id"]: img for img in coco["images"]}
    ann_map = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("bbox") and ann["bbox"][2] > 0 and ann["bbox"][3] > 0:
            ann_map[ann["image_id"]].append(ann)
    print(f"  → {len(image_map)} 张图片, {sum(len(v) for v in ann_map.values())} 个标注")
    return coco, image_map, ann_map

def load_image(image_path):
    return np.array(Image.open(image_path).convert("RGB"))

def gt_bbox_to_actual(bbox_xywh, gt_w, gt_h, aw, ah):
    if gt_w == aw and gt_h == ah: return list(bbox_xywh)
    sx, sy = aw/gt_w, ah/gt_h
    return [bbox_xywh[0]*sx, bbox_xywh[1]*sy, bbox_xywh[2]*sx, bbox_xywh[3]*sy]

def scale_bbox_to_gt(bbox_xywh, aw, ah, gt_w, gt_h):
    if aw == gt_w and ah == gt_h: return list(bbox_xywh)
    sx, sy = gt_w/aw, gt_h/ah
    return [bbox_xywh[0]*sx, bbox_xywh[1]*sy, bbox_xywh[2]*sx, bbox_xywh[3]*sy]

def compute_crop_region(img_array, bboxes_actual, scale_factor=SCALE_FACTOR):
    h, w = img_array.shape[:2]
    ax1 = [b[0] for b in bboxes_actual]
    ay1 = [b[1] for b in bboxes_actual]
    ax2 = [b[0]+b[2] for b in bboxes_actual]
    ay2 = [b[1]+b[3] for b in bboxes_actual]
    cx = (min(ax1)+max(ax2))/2
    cy = (min(ay1)+max(ay2))/2
    ss = np.sqrt(scale_factor)
    cw, ch = int(w*ss), int(h*ss)
    x1, y1 = int(cx-cw/2), int(cy-ch/2)
    if x1<0: x1=0
    if y1<0: y1=0
    x2, y2 = x1+cw, y1+ch
    if x2>w: x2=w; x1=max(0,x2-cw)
    if y2>h: y2=h; y1=max(0,y2-ch)
    return x1, y1, x2, y2

def create_crop_no_padding(img_array, cr):
    x1,y1,x2,y2 = cr
    return img_array[y1:y2, x1:x2].copy()

def create_crop_with_padding(img_array, cr, tss=TARGET_SHORT_SIDE):
    x1,y1,x2,y2 = cr
    crop = img_array[y1:y2, x1:x2].copy()
    ch, cw = crop.shape[:2]
    ss = min(cw, ch)
    if ss < tss:
        ps = tss/ss
        pw, ph = int(cw*ps), int(ch*ps)
    else:
        pw, ph = cw, ch
    mv = img_array.mean(axis=(0,1)).astype(np.uint8)
    canvas = np.full((ph, pw, 3), mv, dtype=np.uint8)
    pl, pt = (pw-cw)//2, (ph-ch)//2
    canvas[pt:pt+ch, pl:pl+cw] = crop
    return canvas, (pl, pt)

def image_to_bytes(img_array):
    buf = BytesIO()
    Image.fromarray(img_array).save(buf, format="JPEG", quality=95)
    return buf.getvalue()

def send_to_cloud(img_array, conf=CONF_THRESHOLD, retries=3):
    ib = image_to_bytes(img_array)
    for a in range(retries):
        try:
            r = requests.post(f"{CLOUD_SERVER_URL}/detect",
                files={"file":("image.jpg",ib,"image/jpeg")},
                data={"conf_threshold":str(conf)}, timeout=60)
            if r.status_code == 200:
                d = r.json()
                if "results" in d:
                    return [det for item in d["results"] for det in item.get("detections",[])]
                return d.get("detections", d if isinstance(d,list) else [])
            if a < retries-1: time.sleep(1)
        except Exception as e:
            if a < retries-1: time.sleep(2)
            else: print(f"  [失败] {e}"); return []
    return []

def parse_bbox(b):
    if isinstance(b, list): return b[0],b[1],b[2],b[3]
    if isinstance(b, dict): return b["x1"],b["y1"],b["x2"],b["y2"]
    raise ValueError(f"Unknown bbox: {type(b)}")

def evaluate_coco_subset(gt_coco, results):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if not results:
        return {"AP50":0,"AP":0,"AR":0,"AP75":0,"AR1":0,"AR10":0,
                "AP_small":0,"AP_medium":0,"AP_large":0,
                "AR_small":0,"AR_medium":0,"AR_large":0,
                "stats":[0]*12,"subset_images":0,"subset_anns":0}
    pids = {d["image_id"] for d in results}
    si = [img for img in gt_coco["images"] if img["id"] in pids]
    sa = [ann for ann in gt_coco["annotations"] if ann["image_id"] in pids]
    sgt = {"images":si,"annotations":sa,"categories":gt_coco["categories"]}
    print(f"\n  Subset GT: {len(si)} 张图片, {len(sa)} 个标注")
    cg = COCO(); cg.dataset = sgt; cg.createIndex()
    cd = cg.loadRes(results)
    ce = COCOeval(cg, cd, "bbox")
    ce.evaluate(); ce.accumulate(); ce.summarize()
    return {"AP50":float(ce.stats[1]),"AP":float(ce.stats[0]),
            "AP75":float(ce.stats[2]),"AR":float(ce.stats[8]),
            "AR1":float(ce.stats[6]),"AR10":float(ce.stats[7]),
            "AP_small":float(ce.stats[3]),"AP_medium":float(ce.stats[4]),
            "AP_large":float(ce.stats[5]),"AR_small":float(ce.stats[9]),
            "AR_medium":float(ce.stats[10]),"AR_large":float(ce.stats[11]),
            "stats":[float(s) for s in ce.stats],
            "subset_images":len(si),"subset_anns":len(sa)}

def print_metrics(m, label, s50):
    print(f"\n  {label}")
    print(f"  {'─'*48}")
    print(f"    AP50 @[0.50]:          {m['AP50']:.4f}")
    print(f"    AP   @[0.50:0.95]:     {m['AP']:.4f}")
    print(f"    AP75 @[0.75]:          {m['AP75']:.4f}")
    print(f"    AR   @[maxDets=1]:     {m['AR1']:.4f}")
    print(f"    AR   @[maxDets=10]:    {m['AR10']:.4f}")
    print(f"    AR   @[maxDets=100]:   {m['AR']:.4f}")
    print(f"    AP  [s/m/l]:           {m['AP_small']:.3f} / {m['AP_medium']:.3f} / {m['AP_large']:.3f}")
    print(f"    AR  [s/m/l]:           {m['AR_small']:.3f} / {m['AR_medium']:.3f} / {m['AR_large']:.3f}")
    print(f"  {'─'*48}")
    d = m['AP50'] - s50
    print(f"    50图小样本 AP50:       {s50:.4f} → 全量 {'↑' if d>=0 else '↓'} {abs(d):.4f}")
    ed = m['AP50'] - EDGE_BASELINE_AP50
    if ed >= 0:
        print(f"    ★ vs 边缘端 0.735:     ↑ +{ed:.4f} (超过边缘端!)")
    else:
        print(f"    ✗ vs 边缘端 0.735:     ↓ {abs(ed):.4f}")
    print(f"    云端全图恢复率:        {m['AP50']/CLOUD_BASELINE_AP50:.1%}")

def run_experiment(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gt_coco, image_map, ann_map = load_gt_annotations(GT_ANNOTATION_PATH)
    image_ids = sorted(ann_map.keys())
    if MAX_IMAGES > 0: image_ids = image_ids[:MAX_IMAGES]
    print(f"[实验] 处理 {len(image_ids)} 张图 | 裁剪: {SCALE_FACTOR:.0%} | 阈值: {CONF_THRESHOLD}")
    print(f"  A组: 50% 无Padding | B组: 50% 有Padding({TARGET_SHORT_SIDE}px)\n")

    ra, rb = [], []
    processed = fail = 0
    ta = tb = 0.0
    ba = bb = bf = 0

    for img_id in tqdm(image_ids, desc="处理图片"):
        ii = image_map[img_id]
        ip = os.path.join(IMAGE_DIR, ii["file_name"])
        if not os.path.exists(ip): fail += 1; continue
        try:
            img = load_image(ip)
        except Exception as e:
            print(f"  [跳过] {ii['file_name']}: {e}"); fail += 1; continue
        ah, aw = img.shape[:2]
        gw, gh = ii["width"], ii["height"]
        bgt = [ann["bbox"] for ann in ann_map[img_id]]
        bac = [gt_bbox_to_actual(b, gw, gh, aw, ah) for b in bgt]
        cr = compute_crop_region(img, bac, SCALE_FACTOR)
        cx1, cy1 = cr[0], cr[1]
        bf += len(image_to_bytes(img))

        # A组: 无padding
        ca = create_crop_no_padding(img, cr)
        ba += len(image_to_bytes(ca))
        t0 = time.time()
        da = send_to_cloud(ca)
        ta += time.time() - t0
        for det in da:
            bx1,by1,bx2,by2 = parse_bbox(det["bbox"])
            sc = det.get("score", det.get("confidence", 0.0))
            ci = det.get("class_id", det.get("category_id", 0))
            if sc < CONF_THRESHOLD: continue
            gb = scale_bbox_to_gt([cx1+bx1, cy1+by1, bx2-bx1, by2-by1], aw, ah, gw, gh)
            if gb[2] <= 0 or gb[3] <= 0: continue
            ra.append({"image_id":img_id,"category_id":ci,"bbox":gb,"score":float(sc)})

        # B组: 有padding
        cb, (pl, pt) = create_crop_with_padding(img, cr, TARGET_SHORT_SIDE)
        bb += len(image_to_bytes(cb))
        t0 = time.time()
        db = send_to_cloud(cb)
        tb += time.time() - t0
        for det in db:
            bx1,by1,bx2,by2 = parse_bbox(det["bbox"])
            sc = det.get("score", det.get("confidence", 0.0))
            ci = det.get("class_id", det.get("category_id", 0))
            if sc < CONF_THRESHOLD: continue
            lx1,ly1 = bx1-pl, by1-pt
            lx2,ly2 = bx2-pl, by2-pt
            gb = scale_bbox_to_gt([cx1+lx1, cy1+ly1, lx2-lx1, ly2-ly1], aw, ah, gw, gh)
            if gb[2] <= 0 or gb[3] <= 0: continue
            rb.append({"image_id":img_id,"category_id":ci,"bbox":gb,"score":float(sc)})

        processed += 1
        if processed % CHECKPOINT_INTERVAL == 0:
            with open(os.path.join(OUTPUT_DIR, "checkpoint.json"), "w") as f:
                json.dump({"processed":processed,"A":ra,"B":rb}, f)

    print("\n" + "="*70)
    print(f"ROI 50% 全量测试 — 无Padding vs 有Padding | {processed}张")
    print("="*70)
    ma = evaluate_coco_subset(gt_coco, ra)
    mb = evaluate_coco_subset(gt_coco, rb)
    print_metrics(ma, "A组: 50% 无Padding", SAMPLE_50_NOPAD_AP50)
    print_metrics(mb, "B组: 50% 有Padding", SAMPLE_50_PAD_AP50)

    print(f"\n{'='*70}\nA vs B 对比\n{'='*70}")
    print(f"  {'指标':<20} {'无Pad':>10} {'有Pad':>10} {'差值':>10}")
    print(f"  {'─'*50}")
    for k in ["AP50","AP","AP75","AR","AR1","AR10","AP_small","AP_medium","AP_large"]:
        va, vb = ma[k], mb[k]
        d = va - vb
        print(f"  {k:<20} {va:>10.4f} {vb:>10.4f} {'↑' if d>0 else '↓' if d<0 else '='} {abs(d):.4f}")

    a_kb = ba/max(processed,1)/1024
    b_kb = bb/max(processed,1)/1024
    f_kb = bf/max(processed,1)/1024
    print(f"\n  带宽: 无Pad {a_kb:.1f}KB(省{(1-a_kb/f_kb)*100:.1f}%) | 有Pad {b_kb:.1f}KB(省{(1-b_kb/f_kb)*100:.1f}%) | 全图 {f_kb:.1f}KB")
    print(f"  时间: 无Pad {ta/max(processed,1):.3f}s/图 | 有Pad {tb/max(processed,1):.3f}s/图")

    best_ap50 = max(ma["AP50"], mb["AP50"])
    best_name = "无Padding" if ma["AP50"] >= mb["AP50"] else "有Padding"
    print(f"\n  ★ 最优: {best_name} AP50={best_ap50:.4f}")
    if best_ap50 >= EDGE_BASELINE_AP50:
        print(f"  ★★ 超过边缘端 0.735! (↑+{best_ap50-EDGE_BASELINE_AP50:.4f})")
    else:
        print(f"  ✗✗ 低于边缘端 0.735 (↓{EDGE_BASELINE_AP50-best_ap50:.4f})")

    with open(os.path.join(OUTPUT_DIR, "roi_50pct_full_results.json"), "w") as f:
        json.dump({"A_nopad":{k:v for k,v in ma.items() if k!="stats"},
                   "B_pad":{k:v for k,v in mb.items() if k!="stats"},
                   "config":{"scale":SCALE_FACTOR,"tss":TARGET_SHORT_SIDE,
                             "conf":CONF_THRESHOLD,"processed":processed},
                   "bandwidth":{"nopad_kb":a_kb,"pad_kb":b_kb,"full_kb":f_kb},
                   "time":{"nopad_s":ta,"pad_s":tb},
                   "baselines":{"edge":EDGE_BASELINE_AP50,"cloud":CLOUD_BASELINE_AP50}}, f, indent=2)
    for name, res in [("A_nopad",ra),("B_pad",rb)]:
        with open(os.path.join(OUTPUT_DIR, f"predictions_{name}.json"), "w") as f:
            json.dump(res, f, indent=2)
    ckpt = os.path.join(OUTPUT_DIR, "checkpoint.json")
    if os.path.exists(ckpt): os.remove(ckpt)
    print(f"\n[保存] → {OUTPUT_DIR}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-images", type=int, default=MAX_IMAGES)
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    p.add_argument("--server", type=str, default=CLOUD_SERVER_URL)
    a = p.parse_args()
    MAX_IMAGES = a.max_images; CONF_THRESHOLD = a.conf; CLOUD_SERVER_URL = a.server
    run_experiment(a)
