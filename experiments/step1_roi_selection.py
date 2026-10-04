#!/usr/bin/env python3
"""Step 1 V2: Support ROI + Exploration ROI 选择

Exploration Score = α*(1-coverage) + β*texture_score + γ*proximity_score
  - Uncovered: Support未覆盖的区域
  - Texture: Laplacian边缘强度，高纹理区域更可能包含目标
  - Proximity: 离Support近的区域更可能有漏检目标
"""
import json, os, argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw
from ultralytics import YOLO
from tqdm import tqdm
from collections import defaultdict

YOLO_MODEL_PATH = "/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt"
GT_ANNOTATION_PATH = "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"
IMAGE_DIR = "/app/data/seadronessee/compressed_version/val2017/"
OUTPUT_DIR = "/app/Salience-DETR/experiments/step1_output/"
YOLO_CONF = 0.01

SUPPORT_EXPAND = 0.20
GRID_ROWS = 8
GRID_COLS = 8
ALPHA = 0.5    # uncovered weight
BETA = 0.3     # texture weight
GAMMA = 0.2    # proximity weight

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR + "vis/", exist_ok=True)

def run_yolo(model, img_path, conf=YOLO_CONF):
    results = model(img_path, conf=conf, verbose=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0: return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clses = r.boxes.cls.cpu().numpy().astype(int)
    return [{"bbox":[float(xyxy[i][0]),float(xyxy[i][1]),float(xyxy[i][2]),float(xyxy[i][3])],
             "score":float(confs[i]),"class_id":int(clses[i])} for i in range(len(xyxy))]

def compute_iou_xyxy(b1, b2):
    x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1])
    x2=min(b1[2],b2[2]); y2=min(b1[3],b2[3])
    if x2<=x1 or y2<=y1: return 0.0
    inter=(x2-x1)*(y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if a1+a2-inter>0 else 0.0

def generate_support_rois(detections, img_w, img_h, expand=SUPPORT_EXPAND):
    supports = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1+x2)/2, (y1+y2)/2
        nw, nh = w * (1 + 2*expand), h * (1 + 2*expand)
        supports.append({
            "bbox": [max(0,cx-nw/2), max(0,cy-nh/2), min(img_w,cx+nw/2), min(img_h,cy+nh/2)],
            "source_det": d["bbox"], "score": d["score"],
            "class_id": d["class_id"], "center": [cx, cy], "type": "support"
        })
    return supports

def compute_support_coverage(supports, gx1, gy1, gx2, gy2):
    grid_area = (gx2 - gx1) * (gy2 - gy1)
    if grid_area <= 0: return 0.0
    covered = 0
    for s in supports:
        sx1, sy1, sx2, sy2 = s["bbox"]
        ix1=max(gx1,sx1); iy1=max(gy1,sy1); ix2=min(gx2,sx2); iy2=min(gy2,sy2)
        if ix2>ix1 and iy2>iy1: covered += (ix2-ix1)*(iy2-iy1)
    return min(covered/grid_area, 1.0)

def compute_texture_score(image, gx1, gy1, gx2, gy2):
    """Laplacian边缘强度作为纹理分数，归一化到[0,1]"""
    region = image[gy1:gy2, gx1:gx2]
    if region.size == 0: return 0.0
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # 用方差衡量纹理强度
    texture = laplacian.var()
    return float(texture)

def compute_all_texture_scores(image, grid_rows, grid_cols, img_w, img_h):
    """预计算所有网格的texture score，归一化"""
    grid_w = img_w / grid_cols
    grid_h = img_h / grid_rows
    scores = {}
    raw_values = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            gx1 = int(c * grid_w); gy1 = int(r * grid_h)
            gx2 = int((c+1) * grid_w); gy2 = int((r+1) * grid_h)
            raw = compute_texture_score(image, gx1, gy1, gx2, gy2)
            scores[(r, c)] = raw
            raw_values.append(raw)
    
    # 归一化到[0,1]
    if max(raw_values) > 0:
        min_v = min(raw_values)
        range_v = max(raw_values) - min_v
        if range_v > 0:
            scores = {k: (v - min_v) / range_v for k, v in scores.items()}
        else:
            scores = {k: 0.5 for k in scores}
    else:
        scores = {k: 0.0 for k in scores}
    return scores

def generate_exploration_rois(supports, image, img_w, img_h,
                              texture_scores,
                              grid_rows=GRID_ROWS, grid_cols=GRID_COLS,
                              topk=8, alpha=ALPHA, beta=BETA, gamma=GAMMA):
    """探索分数 = α*(1-coverage) + β*texture + γ*proximity"""
    grid_w = img_w / grid_cols
    grid_h = img_h / grid_rows
    max_dist = (img_w**2 + img_h**2)**0.5

    candidates = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            gx1 = int(c * grid_w); gy1 = int(r * grid_h)
            gx2 = int((c+1) * grid_w); gy2 = int((r+1) * grid_h)
            gcx = (gx1 + gx2) / 2; gcy = (gy1 + gy2) / 2

            coverage = compute_support_coverage(supports, gx1, gy1, gx2, gy2)
            uncovered = 1.0 - coverage

            texture = texture_scores.get((r, c), 0.0)

            if supports:
                dist = min(((s["center"][0]-gcx)**2+(s["center"][1]-gcy)**2)**0.5
                           for s in supports)
            else:
                dist = max_dist
            proximity = 1.0 - dist / max_dist

            score = alpha * uncovered + beta * texture + gamma * proximity

            candidates.append({
                "bbox": [float(gx1), float(gy1), float(gx2), float(gy2)],
                "grid_row": r, "grid_col": c,
                "support_coverage": round(coverage, 3),
                "uncovered_score": round(uncovered, 3),
                "texture_score": round(texture, 3),
                "proximity_score": round(proximity, 3),
                "explore_score": round(score, 3),
                "type": "exploration"
            })

    candidates.sort(key=lambda x: x["explore_score"], reverse=True)
    return candidates[:topk]

def visualize(img, supports, explores, save_path, gt_boxes=None):
    pil = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(pil)
    for e in explores:
        x1, y1, x2, y2 = e["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
        label = f"S={e['explore_score']:.2f} T={e['texture_score']:.2f}"
        draw.text((x1+5, y1+5), label, fill="red")
    for s in supports:
        x1, y1, x2, y2 = s["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        dx1, dy1, dx2, dy2 = s["source_det"]
        draw.rectangle([dx1, dy1, dx2, dy2], outline="lime", width=2)
    if gt_boxes:
        for gt in gt_boxes:
            draw.rectangle([gt[0], gt[1], gt[2], gt[3]], outline="blue", width=3)
    pil.save(save_path)

def _point_in_box(x, y, box):
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()

    print(f"[Step 1 V2] Support + Exploration ROI选择")
    print(f"  网格: {GRID_ROWS}x{GRID_COLS}, Top-K: {args.topk}")
    print(f"  分数: {ALPHA}*(1-cov) + {BETA}*texture + {GAMMA}*proximity")
    print()

    with open(GT_ANNOTATION_PATH, "r") as f:
        gt_coco = json.load(f)
    image_map = {img["id"]: img for img in gt_coco["images"]}
    ann_map = defaultdict(list)
    for ann in gt_coco["annotations"]:
        if ann.get("bbox") and ann["bbox"][2] > 0 and ann["bbox"][3] > 0:
            ann_map[ann["image_id"]].append(ann)

    image_list = list(image_map.values())[:args.max_images]
    model = YOLO(YOLO_MODEL_PATH)

    all_supports = {}
    all_explores = {}
    explore_hits = 0
    explore_misses = 0

    for img_info in tqdm(image_list, desc="Step 1 V2"):
        img_id = img_info["id"]
        img_name = img_info["file_name"]
        img_path = os.path.join(IMAGE_DIR, img_name)
        img = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w = img.shape[:2]

        dets = run_yolo(model, img_path)
        supports = generate_support_rois(dets, img_w, img_h)
        
        # 预计算所有网格的texture score
        texture_scores = compute_all_texture_scores(img, GRID_ROWS, GRID_COLS, img_w, img_h)
        
        explores = generate_exploration_rois(supports, img, img_w, img_h, texture_scores, topk=args.topk)

        img_key = img_name.replace(".jpg", "").replace(".png", "")
        all_supports[img_key] = supports
        all_explores[img_key] = explores

        gt_anns = ann_map.get(img_id, [])
        gt_boxes = []
        for ann in gt_anns:
            bx, by, bw, bh = ann["bbox"]
            gt_boxes.append([bx, by, bx+bw, by+bh])

        vis_path = os.path.join(OUTPUT_DIR, "vis", f"{img_key}_roi.png")
        visualize(img, supports, explores, vis_path, gt_boxes)

        support_area = sum((s["bbox"][2]-s["bbox"][0])*(s["bbox"][3]-s["bbox"][1]) for s in supports)
        explore_area = sum((e["bbox"][2]-e["bbox"][0])*(e["bbox"][3]-e["bbox"][1]) for e in explores)
        total_area = img_w * img_h

        support_hit = 0; explore_hit = 0
        for gt in gt_boxes:
            gcx = (gt[0] + gt[2]) / 2; gcy = (gt[1] + gt[3]) / 2
            hit_s = any(_point_in_box(gcx, gcy, s["bbox"]) for s in supports)
            if hit_s:
                support_hit += 1
            else:
                hit_e = any(_point_in_box(gcx, gcy, e["bbox"]) for e in explores)
                if hit_e:
                    explore_hit += 1; explore_hits += 1
                explore_misses += 1

        print(f"\n  {img_key}: {img_w}x{img_h}")
        print(f"    YOLO: {len(dets)}个, GT: {len(gt_boxes)}个")
        print(f"    Support: {len(supports)}个, 覆盖率: {support_area/total_area:.1%}")
        print(f"    Exploration: {len(explores)}个, 覆盖率: {explore_area/total_area:.1%}")
        print(f"    GT覆盖: Support={support_hit}/{len(gt_boxes)}, "
              f"Explore命中={explore_hit}/{explore_misses if explore_misses > 0 else 0}")
        for e in explores:
            print(f"      [{e['grid_row']},{e['grid_col']}] "
                  f"cov={e['support_coverage']:.2f} tex={e['texture_score']:.2f} "
                  f"prox={e['proximity_score']:.2f} score={e['explore_score']:.3f}")

    with open(os.path.join(OUTPUT_DIR, "support_boxes.json"), "w") as f:
        json.dump(all_supports, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "explore_boxes.json"), "w") as f:
        json.dump(all_explores, f, indent=2)

    print(f"\n[完成] Support: {sum(len(v) for v in all_supports.values())}个, "
          f"Exploration: {sum(len(v) for v in all_explores.values())}个")
    print(f"  Exploration命中漏检GT: {explore_hits}/{explore_misses}")

if __name__ == "__main__":
    main()
