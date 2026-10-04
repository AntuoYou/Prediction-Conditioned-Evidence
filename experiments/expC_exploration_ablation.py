#!/usr/bin/env python3
"""Experiment C: Exploration Ablation

C1: Top-K Budget (K=0,4,8,12,16) with V1-C scoring
C2: Scoring Function (Random, Coverage-only, Proximity-only, Proposed)

K=0 = Support only (已有全量结果参考)
K=8 + Proposed = Support+Exploration (已有全量结果参考)
"""
import json, os, io, argparse, random
import numpy as np
import cv2
import requests
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

YOLO_MODEL_PATH = "/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt"
CLOUD_URL = "http://localhost:8000/detect"
GT_PATH = "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"
IMG_DIR = "/app/data/seadronessee/compressed_version/val2017/"
EXPAND_RATIO = 0.1
LOW_RES_SCALE = 8
GRID_ROWS = 6
GRID_COLS = 10
ALPHA = 0.3
BETA = 0.7
TOPK_DEFAULT = 8

def send_cloud(img_np, conf=0.05):
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format='JPEG', quality=95)
    buf.seek(0)
    try:
        r = requests.post(CLOUD_URL,
            files={'file': ('img.jpg', buf, 'image/jpeg')},
            data={'conf_threshold': str(conf)}, timeout=60)
        if r.status_code == 200:
            return r.json().get('detections', [])
    except Exception as e:
        print(f"  云端错误: {e}")
    return []

def make_support_rois(img, dets, expand=EXPAND_RATIO):
    h, w = img.shape[:2]
    supports = []
    for det in dets:
        x1, y1, x2, y2 = det
        bw, bh = x2 - x1, y2 - y1
        sx1 = int(max(0, x1 - bw * expand))
        sy1 = int(max(0, y1 - bh * expand))
        sx2 = int(min(w, x2 + bw * expand))
        sy2 = int(min(h, y2 + bh * expand))
        if sx2 > sx1 and sy2 > sy1:
            supports.append((sx1, sy1, sx2, sy2))
    return supports

def compute_support_coverage(supports, gx1, gy1, gx2, gy2):
    grid_area = (gx2 - gx1) * (gy2 - gy1)
    if grid_area <= 0:
        return 1.0
    covered = 0
    for sx1, sy1, sx2, sy2 in supports:
        ix1 = max(gx1, sx1); iy1 = max(gy1, sy1)
        ix2 = min(gx2, sx2); iy2 = min(gy2, sy2)
        if ix2 > ix1 and iy2 > iy1:
            covered += (ix2 - ix1) * (iy2 - iy1)
    return min(1.0, covered / grid_area)

def get_grid_cells(img_w, img_h):
    grid_w = img_w / GRID_COLS
    grid_h = img_h / GRID_ROWS
    cells = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            gx1 = int(c * grid_w)
            gy1 = int(r * grid_h)
            gx2 = int((c + 1) * grid_w)
            gy2 = int((r + 1) * grid_h)
            gcx = (gx1 + gx2) / 2
            gcy = (gy1 + gy2) / 2
            cells.append({"bbox": (gx1, gy1, gx2, gy2), "gcx": gcx, "gcy": gcy})
    return cells

def score_cells(cells, supports, img_w, img_h, mode="proposed"):
    """根据评分模式给所有网格打分"""
    max_dist = (img_w**2 + img_h**2) ** 0.5
    
    if supports:
        sup_centers = [((s[0]+s[2])/2, (s[1]+s[3])/2) for s in supports]
    else:
        sup_centers = [(img_w/2, img_h/2)]
    
    for cell in cells:
        gx1, gy1, gx2, gy2 = cell["bbox"]
        coverage = compute_support_coverage(supports, gx1, gy1, gx2, gy2)
        dist = min(((sc[0]-cell["gcx"])**2 + (sc[1]-cell["gcy"])**2)**0.5 for sc in sup_centers)
        
        if mode == "proposed":
            cell["score"] = ALPHA * (1 - coverage) + BETA * (1 - dist / max_dist)
        elif mode == "coverage":
            cell["score"] = 1 - coverage
        elif mode == "proximity":
            cell["score"] = 1 - dist / max_dist
        elif mode == "random":
            cell["score"] = random.random()
        
        cell["coverage"] = coverage
        cell["dist"] = dist
    
    return cells

def select_topk(cells, topk):
    """选得分最高的topk个网格作为Exploration ROI"""
    sorted_cells = sorted(cells, key=lambda x: x["score"], reverse=True)
    return [c["bbox"] for c in sorted_cells[:topk]]

def make_lowres_canvas(img, rois):
    h, w = img.shape[:2]
    small_w = max(1, w // LOW_RES_SCALE)
    small_h = max(1, h // LOW_RES_SCALE)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    canvas = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    for rx1, ry1, rx2, ry2 in rois:
        if rx2 > rx1 and ry2 > ry1:
            canvas[ry1:ry2, rx1:rx2] = img[ry1:ry2, rx1:rx2]
    return canvas

def calc_payload(img, rois):
    h, w = img.shape[:2]
    total = 0
    small_w = max(1, w // LOW_RES_SCALE)
    small_h = max(1, h // LOW_RES_SCALE)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(small).save(buf, format='JPEG', quality=95)
    total += len(buf.getvalue())
    for rx1, ry1, rx2, ry2 in rois:
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        crop = img[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            continue
        buf = io.BytesIO()
        Image.fromarray(crop).save(buf, format='JPEG', quality=95)
        total += len(buf.getvalue())
    return total

def evaluate(results_list, name, img_ids_used):
    if not results_list:
        return {"AP50": 0, "AP": 0, "AR": 0, "APsmall": 0, "APmedium": 0, "APlarge": 0, "AP75": 0, "num_dets": 0}
    tmp = f"/tmp/expC_{name}.json"
    with open(tmp, "w") as f:
        json.dump(results_list, f)
    try:
        coco_gt = COCO(GT_PATH)
        coco_dt = coco_gt.loadRes(tmp)
        e = COCOeval(coco_gt, coco_dt, 'bbox')
        e.params.imgIds = list(img_ids_used)
        e.evaluate()
        e.accumulate()
        e.summarize()
        return {
            "AP50": float(e.stats[1]),
            "AP": float(e.stats[0]),
            "AP75": float(e.stats[2]),
            "APsmall": float(e.stats[3]),
            "APmedium": float(e.stats[4]),
            "APlarge": float(e.stats[5]),
            "AR": float(e.stats[8]),
            "ARsmall": float(e.stats[11]),
            "num_dets": len(results_list),
        }
    except Exception as ex:
        print(f"  {name} 评估失败: {ex}")
        return {"AP50": 0, "AP": 0, "AR": 0, "APsmall": 0, "APmedium": 0, "APlarge": 0, "AP75": 0, "num_dets": len(results_list)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=200)
    args = parser.parse_args()
    
    print("=" * 70)
    print("Experiment C: Exploration Ablation")
    print("=" * 70)
    
    with open(GT_PATH) as f:
        gt_coco = json.load(f)
    images = gt_coco["images"][:args.max_images]
    print(f"图片数: {len(images)}")
    
    model = YOLO(YOLO_MODEL_PATH)
    
    # C1: Top-K Budget
    topk_list = [0, 4, 8, 12, 16]
    # C2: Scoring Function (all K=8)
    scoring_modes = ["random", "coverage", "proximity", "proposed"]
    
    # 用key标识每个变体
    variants = {}
    for k in topk_list:
        variants[f"topk_{k}"] = []
    for mode in scoring_modes:
        variants[f"score_{mode}"] = []
    
    # topk_0 和 score_proposed 是重复的(Support only)，但分开跑确保一致性
    # topk_8 和 score_proposed 应该一样
    
    img_ids_used = set()
    total_payloads = {k: 0 for k in variants}
    
    for img_info in tqdm(images, desc="处理中"):
        img_id = img_info["id"]
        fname = img_info["file_name"]
        img_path = os.path.join(IMG_DIR, fname)
        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]
        
        results_yolo = model(img_path, conf=0.01, verbose=False)
        r = results_yolo[0]
        
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().tolist()
            dets = [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in xyxy]
        else:
            dets = []
        
        supports = make_support_rois(img, dets)
        cells = get_grid_cells(w, h)
        
        # === C1: Top-K Budget (V1-C scoring) ===
        scored_proposed = score_cells(cells, supports, w, h, "proposed")
        
        for k in topk_list:
            if k == 0:
                rois = supports
            else:
                exp_rois = select_topk(scored_proposed, k)
                rois = supports + exp_rois
            
            canvas = make_lowres_canvas(img, rois)
            cloud_dets = send_cloud(canvas, 0.05)
            total_payloads[f"topk_{k}"] += calc_payload(img, rois)
            
            for d in cloud_dets:
                b = d["bbox"]
                variants[f"topk_{k}"].append({
                    "image_id": int(img_id),
                    "category_id": int(d.get("class_id", 1)),
                    "bbox": [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])],
                    "score": float(d.get("confidence", 0)),
                })
        
        # === C2: Scoring Function (K=8) ===
        for mode in scoring_modes:
            scored = score_cells(cells, supports, w, h, mode)
            exp_rois = select_topk(scored, TOPK_DEFAULT)
            rois = supports + exp_rois
            
            canvas = make_lowres_canvas(img, rois)
            cloud_dets = send_cloud(canvas, 0.05)
            total_payloads[f"score_{mode}"] += calc_payload(img, rois)
            
            for d in cloud_dets:
                b = d["bbox"]
                variants[f"score_{mode}"].append({
                    "image_id": int(img_id),
                    "category_id": int(d.get("class_id", 1)),
                    "bbox": [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])],
                    "score": float(d.get("confidence", 0)),
                })
        
        img_ids_used.add(int(img_id))
    
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    
    # 评估所有变体
    metrics = {}
    for name in variants:
        print(f"\n--- {name} ---")
        metrics[name] = evaluate(variants[name], name, img_ids_used)
    
    print("\n" + "=" * 70)
    print("C1: Top-K Budget (V1-C scoring)")
    print("=" * 70)
    print(f"\n{'K':<8} {'AP50':<8} {'AP':<8} {'AP75':<8} {'APsmall':<8} {'APmed':<8} {'APlarge':<8} {'AR':<8} {'ARsmall':<8} {'检测数':<8} {'传输量':<10}")
    print("-" * 105)
    for k in topk_list:
        m = metrics[f"topk_{k}"]
        avg_p = total_payloads[f"topk_{k}"] / len(img_ids_used)
        print(f"K={k:<5} {m['AP50']:<8.4f} {m['AP']:<8.4f} {m['AP75']:<8.4f} {m['APsmall']:<8.4f} {m['APmedium']:<8.4f} {m['APlarge']:<8.4f} {m['AR']:<8.4f} {m['ARsmall']:<8.4f} {m['num_dets']:<8} {avg_p:<10.0f}")
    
    print(f"\n{'K':<8} {'AP50相对K=0':<15} {'APsmall相对K=0':<18} {'AR相对K=0':<15}")
    print("-" * 60)
    base_ap50 = metrics["topk_0"]["AP50"]
    base_apsmall = metrics["topk_0"]["APsmall"]
    base_ar = metrics["topk_0"]["AR"]
    for k in topk_list:
        m = metrics[f"topk_{k}"]
        if base_ap50 > 0:
            print(f"K={k:<5} {m['AP50']/base_ap50*100:<14.1f}% {m['APsmall']/base_apsmall*100:<17.1f}% {m['AR']/base_ar*100:<14.1f}%")
    
    print("\n" + "=" * 70)
    print("C2: Scoring Function (K=8)")
    print("=" * 70)
    print(f"\n{'策略':<15} {'AP50':<8} {'AP':<8} {'AP75':<8} {'APsmall':<8} {'APmed':<8} {'APlarge':<8} {'AR':<8} {'ARsmall':<8} {'检测数':<8} {'传输量':<10}")
    print("-" * 110)
    for mode in scoring_modes:
        m = metrics[f"score_{mode}"]
        avg_p = total_payloads[f"score_{mode}"] / len(img_ids_used)
        print(f"{mode:<15} {m['AP50']:<8.4f} {m['AP']:<8.4f} {m['AP75']:<8.4f} {m['APsmall']:<8.4f} {m['APmedium']:<8.4f} {m['APlarge']:<8.4f} {m['AR']:<8.4f} {m['ARsmall']:<8.4f} {m['num_dets']:<8} {avg_p:<10.0f}")
    
    # 保存
    with open("/app/Salience-DETR/experiments/expC_exploration_ablation.json", "w") as f:
        json.dump({
            "C1_topk": {f"K={k}": metrics[f"topk_{k}"] for k in topk_list},
            "C2_scoring": {mode: metrics[f"score_{mode}"] for mode in scoring_modes},
            "payloads": total_payloads,
            "num_images": len(img_ids_used),
            "config": {
                "expand_ratio": EXPAND_RATIO,
                "low_res_scale": LOW_RES_SCALE,
                "alpha": ALPHA,
                "beta": BETA,
                "grid_rows": GRID_ROWS,
                "grid_cols": GRID_COLS,
                "topk_default": TOPK_DEFAULT,
            },
        }, f, indent=2)
    
    print(f"\n结果已保存: expC_exploration_ablation.json")

if __name__ == "__main__":
    main()
