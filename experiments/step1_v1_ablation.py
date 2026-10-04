#!/usr/bin/env python3
"""Step 1 V1 权重消融: U(uncovered) + P(proximity) 不同权重对比

V1-A: 0.7U + 0.3P (当前版本)
V1-B: 0.5U + 0.5P
V1-C: 0.3U + 0.7P
"""
import json, os
import numpy as np
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
from collections import defaultdict

YOLO_MODEL_PATH = "/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt"
GT_ANNOTATION_PATH = "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"
IMAGE_DIR = "/app/data/seadronessee/compressed_version/val2017/"
YOLO_CONF = 0.01
SUPPORT_EXPAND = 0.20
GRID_ROWS = 8; GRID_COLS = 8
TOPK = 8

VARIANTS = [
    ("V1-A", 0.7, 0.3),
    ("V1-B", 0.5, 0.5),
    ("V1-C", 0.3, 0.7),
]

def run_yolo(model, img_path, conf=YOLO_CONF):
    results = model(img_path, conf=conf, verbose=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0: return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clses = r.boxes.cls.cpu().numpy().astype(int)
    return [{"bbox":[float(xyxy[i][0]),float(xyxy[i][1]),float(xyxy[i][2]),float(xyxy[i][3])],
             "score":float(confs[i]),"class_id":int(clses[i])} for i in range(len(xyxy))]

def generate_support_rois(detections, img_w, img_h, expand=0.20):
    supports = []
    for d in detections:
        x1,y1,x2,y2 = d["bbox"]
        w,h = x2-x1, y2-y1
        cx,cy = (x1+x2)/2, (y1+y2)/2
        nw,nh = w*(1+2*expand), h*(1+2*expand)
        supports.append({"bbox":[max(0,cx-nw/2),max(0,cy-nh/2),min(img_w,cx+nw/2),min(img_h,cy+nh/2)],
            "source_det":d["bbox"],"center":[cx,cy]})
    return supports

def compute_support_coverage(supports, gx1, gy1, gx2, gy2):
    grid_area = (gx2-gx1)*(gy2-gy1)
    if grid_area <= 0: return 0.0
    covered = 0
    for s in supports:
        sx1,sy1,sx2,sy2 = s["bbox"]
        ix1=max(gx1,sx1);iy1=max(gy1,sy1);ix2=min(gx2,sx2);iy2=min(gy2,sy2)
        if ix2>ix1 and iy2>iy1: covered += (ix2-ix1)*(iy2-iy1)
    return min(covered/grid_area, 1.0)

def _point_in_box(x, y, box):
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]

def generate_exploration(supports, img_w, img_h, alpha, beta, topk=TOPK):
    grid_w = img_w/GRID_COLS; grid_h = img_h/GRID_ROWS
    max_dist = (img_w**2 + img_h**2)**0.5
    candidates = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            gx1=int(c*grid_w);gy1=int(r*grid_h);gx2=int((c+1)*grid_w);gy2=int((r+1)*grid_h)
            gcx=(gx1+gx2)/2;gcy=(gy1+gy2)/2
            coverage = compute_support_coverage(supports,gx1,gy1,gx2,gy2)
            uncovered = 1.0 - coverage
            if supports:
                dist = min(((s["center"][0]-gcx)**2+(s["center"][1]-gcy)**2)**0.5 for s in supports)
            else:
                dist = max_dist
            proximity = 1.0 - dist/max_dist
            score = alpha*uncovered + beta*proximity
            candidates.append({"bbox":[float(gx1),float(gy1),float(gx2),float(gy2)],
                "score":round(score,3)})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:topk]

def main():
    with open(GT_ANNOTATION_PATH) as f:
        gt_coco = json.load(f)
    image_map = {img["id"]:img for img in gt_coco["images"]}
    ann_map = defaultdict(list)
    for ann in gt_coco["annotations"]:
        if ann.get("bbox") and ann["bbox"][2]>0 and ann["bbox"][3]>0:
            ann_map[ann["image_id"]].append(ann)
    image_list = list(image_map.values())
    
    print("[加载] YOLO模型...")
    model = YOLO(YOLO_MODEL_PATH)
    print(f"图片: {len(image_list)}, GT: {sum(len(v) for v in ann_map.values())}")
    print()
    
    # 先对所有图跑一次YOLO + Support，缓存结果
    print("[Step 1] YOLO + Support 生成中...")
    cached = {}
    total_support = 0; total_gt = 0; support_hit = 0; total_miss = 0
    miss_gt_list = []
    
    for img_info in tqdm(image_list):
        img_id = img_info["id"]
        img_path = os.path.join(IMAGE_DIR, img_info["file_name"])
        img = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w = img.shape[:2]
        
        dets = run_yolo(model, img_path)
        supports = generate_support_rois(dets, img_w, img_h)
        
        gt_anns = ann_map.get(img_id, [])
        total_gt += len(gt_anns)
        total_support += len(supports)
        
        # 记录每个GT是否被Support命中
        gt_status = []
        for ann in gt_anns:
            bx,by,bw,bh = ann["bbox"]
            gcx = bx+bw/2; gcy = by+bh/2
            hit_s = any(_point_in_box(gcx,gcy,s["bbox"]) for s in supports)
            gt_status.append({"cx":gcx, "cy":gcy, "hit_support":hit_s})
            if hit_s: support_hit += 1
            else:
                total_miss += 1
                miss_gt_list.append((img_id, gcx, gcy))
        
        cached[img_id] = {
            "img_w": img_w, "img_h": img_h,
            "supports": supports,
            "gt_status": gt_status,
        }
    
    print(f"  Support: {total_support}个")
    print(f"  Support命中: {support_hit}/{total_gt} ({support_hit/total_gt:.1%})")
    print(f"  漏检GT: {total_miss}")
    print()
    
    # 对每个variant计算Exploration命中
    results = {}
    for name, alpha, beta in VARIANTS:
        print(f"[Step 2] 计算 {name} (α={alpha}, β={beta})...")
        
        explore_hit = 0
        total_explore = 0
        explore_regions_hit = []  # 用于计算efficiency
        
        for img_id, data in tqdm(cached.items()):
            img_w = data["img_w"]; img_h = data["img_h"]
            supports = data["supports"]
            gt_status = data["gt_status"]
            
            explores = generate_exploration(supports, img_w, img_h, alpha, beta, TOPK)
            total_explore += len(explores)
            
            for gt in gt_status:
                if not gt["hit_support"]:
                    hit_e = any(_point_in_box(gt["cx"], gt["cy"], e["bbox"]) for e in explores)
                    if hit_e: explore_hit += 1
        
        miss_recovery = explore_hit / total_miss if total_miss > 0 else 0
        efficiency = explore_hit / total_explore if total_explore > 0 else 0
        
        results[name] = {
            "alpha": alpha, "beta": beta,
            "explore_regions": total_explore,
            "explore_hit": explore_hit,
            "total_miss": total_miss,
            "miss_recovery": round(miss_recovery, 4),
            "efficiency": round(efficiency, 6),
        }
        print(f"  Exploration命中: {explore_hit}/{total_miss} ({miss_recovery:.1%})")
        print(f"  效率: {efficiency:.6f} (recovered/regions)")
        print()
    
    # 汇总对比
    print("=" * 70)
    print(f"{'V1 权重消融对比':^70}")
    print("=" * 70)
    print(f"  {'版本':<8} {'α(U)':<6} {'β(P)':<6} {'Regions':<10} {'Recovered':<10} {'Recovery%':<10} {'Efficiency':<12}")
    print(f"  {'-'*68}")
    for name, alpha, beta in VARIANTS:
        r = results[name]
        print(f"  {name:<8} {alpha:<6} {beta:<6} {r['explore_regions']:<10} "
              f"{r['explore_hit']:<10} {r['miss_recovery']*100:<10.1f} {r['efficiency']:<12.6f}")
    print()
    print(f"  Support命中: {support_hit}/{total_gt} ({support_hit/total_gt:.1%})")
    print(f"  漏检GT: {total_miss}")
    print()
    
    with open("/app/Salience-DETR/experiments/step1_v1_ablation_result.json","w") as f:
        json.dump(results, f, indent=2)
    print(f"[保存] step1_v1_ablation_result.json")

if __name__ == "__main__":
    main()
