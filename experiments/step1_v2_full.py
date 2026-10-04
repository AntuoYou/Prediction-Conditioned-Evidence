#!/usr/bin/env python3
"""Step 1 V2 全量: coverage + texture + proximity"""
import json, os
import numpy as np
import cv2
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
ALPHA = 0.5; BETA = 0.3; GAMMA = 0.2

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
        w,h = x2-x1,y2-y1
        cx,cy = (x1+x2)/2,(y1+y2)/2
        nw,nh = w*(1+2*expand),h*(1+2*expand)
        supports.append({"bbox":[max(0,cx-nw/2),max(0,cy-nh/2),min(img_w,cx+nw/2),min(img_h,cy+nh/2)],
            "source_det":d["bbox"],"center":[cx,cy],"type":"support"})
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

def compute_texture_scores(image, img_w, img_h):
    grid_w = img_w/GRID_COLS; grid_h = img_h/GRID_ROWS
    scores = {}; raw = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            gx1=int(c*grid_w);gy1=int(r*grid_h);gx2=int((c+1)*grid_w);gy2=int((r+1)*grid_h)
            region = image[gy1:gy2, gx1:gx2]
            if region.size == 0: t = 0.0
            else:
                gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
                t = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            scores[(r,c)] = t; raw.append(t)
    if max(raw) > 0:
        mn = min(raw); rng = max(raw)-mn
        if rng > 0: scores = {k:(v-mn)/rng for k,v in scores.items()}
        else: scores = {k:0.5 for k in scores}
    else:
        scores = {k:0.0 for k in scores}
    return scores

def _point_in_box(x, y, box):
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]

def generate_exploration_rois(supports, texture_scores, img_w, img_h, topk=8):
    grid_w = img_w/GRID_COLS; grid_h = img_h/GRID_ROWS
    max_dist = (img_w**2 + img_h**2)**0.5
    candidates = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            gx1=int(c*grid_w);gy1=int(r*grid_h);gx2=int((c+1)*grid_w);gy2=int((r+1)*grid_h)
            gcx=(gx1+gx2)/2;gcy=(gy1+gy2)/2
            coverage = compute_support_coverage(supports,gx1,gy1,gx2,gy2)
            uncovered = 1.0 - coverage
            texture = texture_scores.get((r,c),0.0)
            if supports:
                dist = min(((s["center"][0]-gcx)**2+(s["center"][1]-gcy)**2)**0.5 for s in supports)
            else:
                dist = max_dist
            proximity = 1.0 - dist/max_dist
            score = ALPHA*uncovered + BETA*texture + GAMMA*proximity
            candidates.append({"bbox":[float(gx1),float(gy1),float(gx2),float(gy2)],
                "grid_row":r,"grid_col":c,"explore_score":round(score,3),"type":"exploration"})
    candidates.sort(key=lambda x: x["explore_score"], reverse=True)
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
    
    model = YOLO(YOLO_MODEL_PATH)
    
    total_support=0;total_explore=0;total_gt=0;support_hit=0;explore_hit=0;total_miss=0
    
    for img_info in tqdm(image_list, desc="V2"):
        img_id = img_info["id"]
        img_path = os.path.join(IMAGE_DIR, img_info["file_name"])
        img = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w = img.shape[:2]
        
        dets = run_yolo(model, img_path)
        supports = generate_support_rois(dets, img_w, img_h)
        texture_scores = compute_texture_scores(img, img_w, img_h)
        explores = generate_exploration_rois(supports, texture_scores, img_w, img_h, topk=8)
        
        total_support += len(supports); total_explore += len(explores)
        
        gt_anns = ann_map.get(img_id, [])
        total_gt += len(gt_anns)
        for ann in gt_anns:
            bx,by,bw,bh = ann["bbox"]
            gcx = bx+bw/2; gcy = by+bh/2
            hit_s = any(_point_in_box(gcx,gcy,s["bbox"]) for s in supports)
            if hit_s:
                support_hit += 1
            else:
                total_miss += 1
                hit_e = any(_point_in_box(gcx,gcy,e["bbox"]) for e in explores)
                if hit_e: explore_hit += 1
    
    print(f"\n{'='*60}")
    print(f"V2 全量结果 (coverage + texture + proximity)")
    print(f"{'='*60}")
    print(f"  图片: {len(image_list)}")
    print(f"  Support: {total_support}个, Exploration: {total_explore}个")
    print(f"  GT总数: {total_gt}")
    print(f"  Support命中: {support_hit}/{total_gt} ({support_hit/total_gt:.1%})")
    print(f"  漏检GT: {total_miss}")
    print(f"  Exploration命中: {explore_hit}/{total_miss} ({explore_hit/total_miss:.1%})" if total_miss > 0 else "  漏检: 0")
    print(f"  总命中率: {(support_hit+explore_hit)/total_gt:.1%}")
    
    with open("/app/Salience-DETR/experiments/step1_v2_result.json","w") as f:
        json.dump({"version":"V2","images":len(image_list),"support":total_support,
            "explore":total_explore,"gt":total_gt,"support_hit":support_hit,
            "explore_hit":explore_hit,"total_miss":total_miss}, f, indent=2)

if __name__ == "__main__":
    main()
