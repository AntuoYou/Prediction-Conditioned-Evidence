#!/usr/bin/env python3
"""验证: Support ROI + Exploration ROI 完整双路径 (低分辨率背景)

对比三组:
1. 50% ROI (Baseline)
2. Support only (expand=0.1 + lowres)
3. Support + Exploration (expand=0.1 + lowres)

Exploration使用V1-C: score = 0.3*(1-coverage) + 0.7*(1-dist/max_dist)
"""
import json, os, io, argparse
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
ALPHA = 0.3  # coverage weight
BETA = 0.7   # proximity weight
TOPK_EXPLORE = 8

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

def compute_roi_50(dets, img_w, img_h):
    if not dets:
        return [0, 0, min(img_w, 800), min(img_h, 800)]
    x1s = [d[0] for d in dets]; y1s = [d[1] for d in dets]
    x2s = [d[2] for d in dets]; y2s = [d[3] for d in dets]
    cx = (min(x1s) + max(x2s)) / 2
    cy = (min(y1s) + max(y2s)) / 2
    ss = np.sqrt(0.5)
    cw, ch = int(img_w * ss), int(img_h * ss)
    x1 = int(max(0, cx - cw/2)); y1 = int(max(0, cy - ch/2))
    x2, y2 = x1 + cw, y1 + ch
    if x2 > img_w: x2 = img_w; x1 = max(0, x2 - cw)
    if y2 > img_h: y2 = img_h; y1 = max(0, y2 - ch)
    return [int(x1), int(y1), int(x2), int(y2)]

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
    """计算grid cell被support覆盖的比例"""
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

def generate_exploration_rois(supports, img_w, img_h, topk=TOPK_EXPLORE):
    """V1-C: score = alpha*(1-coverage) + beta*(1-dist/max_dist)"""
    grid_w = img_w / GRID_COLS
    grid_h = img_h / GRID_ROWS
    max_dist = (img_w**2 + img_h**2) ** 0.5
    
    # Support中心点
    if supports:
        sup_centers = [((s[0]+s[2])/2, (s[1]+s[3])/2) for s in supports]
    else:
        sup_centers = [(img_w/2, img_h/2)]
    
    candidates = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            gx1 = int(c * grid_w)
            gy1 = int(r * grid_h)
            gx2 = int((c + 1) * grid_w)
            gy2 = int((r + 1) * grid_h)
            
            coverage = compute_support_coverage(supports, gx1, gy1, gx2, gy2)
            
            gcx = (gx1 + gx2) / 2
            gcy = (gy1 + gy2) / 2
            min_dist = min(((sc[0]-gcx)**2 + (sc[1]-gcy)**2)**0.5 for sc in sup_centers)
            
            score = ALPHA * (1 - coverage) + BETA * (1 - min_dist / max_dist)
            candidates.append({
                "bbox": [gx1, gy1, gx2, gy2],
                "score": score,
                "coverage": coverage,
                "dist": min_dist,
            })
    
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return [(c["bbox"][0], c["bbox"][1], c["bbox"][2], c["bbox"][3]) for c in candidates[:topk]]

def make_lowres_canvas(img, rois):
    """低分辨率背景 + ROI叠加"""
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

def calc_payload_roi50(crop):
    buf = io.BytesIO()
    Image.fromarray(crop).save(buf, format='JPEG', quality=95)
    return len(buf.getvalue())

def evaluate(results_list, name, img_ids_used):
    if not results_list:
        return {"AP50": 0, "AP": 0, "AR": 0, "APsmall": 0, "APmedium": 0, "APlarge": 0, "AP75": 0, "num_dets": 0}
    tmp = f"/tmp/final_{name}.json"
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
    parser.add_argument("--max-images", type=int, default=1600)
    args = parser.parse_args()
    
    print("=" * 70)
    print("验证: Support + Exploration (低分辨率背景)")
    print("=" * 70)
    
    with open(GT_PATH) as f:
        gt_coco = json.load(f)
    images = gt_coco["images"][:args.max_images]
    print(f"图片数: {len(images)}")
    print(f"Support: expand={EXPAND_RATIO}")
    print(f"Exploration: V1-C (alpha={ALPHA}, beta={BETA}, topk={TOPK_EXPLORE})")
    
    model = YOLO(YOLO_MODEL_PATH)
    
    results_roi50 = []
    results_support = []
    results_support_explore = []
    img_ids_used = set()
    
    total_p_roi50 = 0
    total_p_support = 0
    total_p_sup_exp = 0
    
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
        
        # 1. 50% ROI
        roi = compute_roi_50(dets, w, h)
        rx1, ry1, rx2, ry2 = roi
        crop_50 = img[ry1:ry2, rx1:rx2]
        total_p_roi50 += calc_payload_roi50(crop_50)
        cloud_50 = send_cloud(crop_50, 0.05)
        for d in cloud_50:
            b = d["bbox"]
            results_roi50.append({
                "image_id": int(img_id),
                "category_id": int(d.get("class_id", 1)),
                "bbox": [float(b[0]+rx1), float(b[1]+ry1), float(b[2]-b[0]), float(b[3]-b[1])],
                "score": float(d.get("confidence", 0)),
            })
        
        # 2. Support only
        supports = make_support_rois(img, dets)
        canvas_sup = make_lowres_canvas(img, supports)
        total_p_support += calc_payload(img, supports)
        cloud_sup = send_cloud(canvas_sup, 0.05)
        for d in cloud_sup:
            b = d["bbox"]
            results_support.append({
                "image_id": int(img_id),
                "category_id": int(d.get("class_id", 1)),
                "bbox": [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])],
                "score": float(d.get("confidence", 0)),
            })
        
        # 3. Support + Exploration
        explorations = generate_exploration_rois(supports, w, h)
        all_rois = supports + explorations
        canvas_sup_exp = make_lowres_canvas(img, all_rois)
        total_p_sup_exp += calc_payload(img, all_rois)
        cloud_sup_exp = send_cloud(canvas_sup_exp, 0.05)
        for d in cloud_sup_exp:
            b = d["bbox"]
            results_support_explore.append({
                "image_id": int(img_id),
                "category_id": int(d.get("class_id", 1)),
                "bbox": [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])],
                "score": float(d.get("confidence", 0)),
            })
        
        img_ids_used.add(int(img_id))
    
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    
    print("\n--- 50% ROI ---")
    m_roi50 = evaluate(results_roi50, "roi50", img_ids_used)
    
    print("\n--- Support only ---")
    m_support = evaluate(results_support, "support", img_ids_used)
    
    print("\n--- Support + Exploration ---")
    m_sup_exp = evaluate(results_support_explore, "sup_exp", img_ids_used)
    
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    
    print(f"\n{'方案':<30} {'AP50':<8} {'AP':<8} {'AP75':<8} {'APsmall':<8} {'APmed':<8} {'APlarge':<8} {'AR':<8} {'检测数':<8}")
    print("-" * 100)
    print(f"{'50% ROI':<30} {m_roi50['AP50']:<8.4f} {m_roi50['AP']:<8.4f} {m_roi50['AP75']:<8.4f} {m_roi50['APsmall']:<8.4f} {m_roi50['APmedium']:<8.4f} {m_roi50['APlarge']:<8.4f} {m_roi50['AR']:<8.4f} {m_roi50['num_dets']:<8}")
    print(f"{'Support only':<30} {m_support['AP50']:<8.4f} {m_support['AP']:<8.4f} {m_support['AP75']:<8.4f} {m_support['APsmall']:<8.4f} {m_support['APmedium']:<8.4f} {m_support['APlarge']:<8.4f} {m_support['AR']:<8.4f} {m_support['num_dets']:<8}")
    print(f"{'Support + Exploration':<30} {m_sup_exp['AP50']:<8.4f} {m_sup_exp['AP']:<8.4f} {m_sup_exp['AP75']:<8.4f} {m_sup_exp['APsmall']:<8.4f} {m_sup_exp['APmedium']:<8.4f} {m_sup_exp['APlarge']:<8.4f} {m_sup_exp['AR']:<8.4f} {m_sup_exp['num_dets']:<8}")
    
    print(f"\n传输量统计:")
    n = len(img_ids_used)
    print(f"  50% ROI:              {total_p_roi50:>14,} bytes (avg {total_p_roi50/n:.0f} bytes/img)")
    print(f"  Support only:         {total_p_support:>14,} bytes (avg {total_p_support/n:.0f} bytes/img)")
    print(f"  Support+Exploration:   {total_p_sup_exp:>14,} bytes (avg {total_p_sup_exp/n:.0f} bytes/img)")
    
    if m_roi50['AP50'] > 0:
        print(f"\n相对 50% ROI:")
        print(f"  Support:       AP50={m_support['AP50']/m_roi50['AP50']*100:.1f}%  mAP={m_support['AP']/m_roi50['AP']*100:.1f}%  AR={m_support['AR']/m_roi50['AR']*100:.1f}%")
        print(f"  Support+Exp:  AP50={m_sup_exp['AP50']/m_roi50['AP50']*100:.1f}%  mAP={m_sup_exp['AP']/m_roi50['AP']*100:.1f}%  AR={m_sup_exp['AR']/m_roi50['AR']*100:.1f}%")
    
    if m_support['AP50'] > 0:
        print(f"\nSupport+Exp 相对 Support only:")
        print(f"  AP50提升: {m_sup_exp['AP50'] - m_support['AP50']:+.4f} ({(m_sup_exp['AP50']/m_support['AP50'] - 1)*100:+.1f}%)")
        print(f"  AR提升:  {m_sup_exp['AR'] - m_support['AR']:+.4f} ({(m_sup_exp['AR']/m_support['AR'] - 1)*100:+.1f}%)")
    
    with open("/app/Salience-DETR/experiments/support_exploration_result.json", "w") as f:
        json.dump({
            "roi50": m_roi50,
            "support": m_support,
            "support_exploration": m_sup_exp,
            "payloads": {
                "roi50": total_p_roi50,
                "support": total_p_support,
                "support_exploration": total_p_sup_exp,
            },
            "num_images": n,
            "config": {
                "expand_ratio": EXPAND_RATIO,
                "low_res_scale": LOW_RES_SCALE,
                "alpha": ALPHA,
                "beta": BETA,
                "topk_explore": TOPK_EXPLORE,
                "grid_rows": GRID_ROWS,
                "grid_cols": GRID_COLS,
            },
        }, f, indent=2)
    
    print(f"\n结果已保存: support_exploration_result.json")

if __name__ == "__main__":
    main()
