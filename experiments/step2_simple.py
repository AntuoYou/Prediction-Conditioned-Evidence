#!/usr/bin/env python3
"""Step 2简化版: 对比 ROI原图 vs 100%patch重建 vs 模糊背景patch

直接用pycocotools评估，确保image_id正确。
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
PATCH_SIZE = 16

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
    except:
        pass
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
    return [x1, y1, x2, y2]

def patch_reconstruct_all(crop):
    """切patch再拼回去（100%，无损）"""
    return crop.copy()  # 全部patch都在，相当于原图

def patch_reconstruct_blur(crop, patch_size=PATCH_SIZE):
    """模糊背景 + 所有patch（验证背景填充）"""
    h, w = crop.shape[:2]
    background = cv2.GaussianBlur(crop, (31, 31), 0)
    canvas = background.copy()
    
    n_rows = (h + patch_size - 1) // patch_size
    n_cols = (w + patch_size - 1) // patch_size
    
    for r in range(n_rows):
        for c in range(n_cols):
            px1 = c * patch_size
            py1 = r * patch_size
            px2 = min(px1 + patch_size, w)
            py2 = min(py1 + patch_size, h)
            canvas[py1:py2, px1:px2] = crop[py1:py2, px1:px2]
    
    return canvas

def evaluate(results_subset, name):
    """用pycocotools评估"""
    if not results_subset:
        print(f"  {name}: 无结果")
        return
    
    # 写临时JSON
    tmp = f"/tmp/step2_{name}.json"
    with open(tmp, "w") as f:
        json.dump(results_subset, f)
    
    try:
        coco_gt = COCO(GT_PATH)
        coco_dt = coco_gt.loadRes(tmp)
        e = COCOeval(coco_gt, coco_dt, 'bbox')
        e.evaluate()
        e.accumulate()
        e.summarize()
        return {
            "AP50": float(e.stats[1]),
            "AP": float(e.stats[0]),
            "AR": float(e.stats[8]),
        }
    except Exception as ex:
        print(f"  {name} 评估失败: {ex}")
        return {"AP50": 0, "AP": 0, "AR": 0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=20)
    args = parser.parse_args()
    
    print("=" * 60)
    print("Step 2: Patch级传输验证 (简化版)")
    print("=" * 60)
    
    with open(GT_PATH) as f:
        gt_coco = json.load(f)
    images = gt_coco["images"][:args.max_images]
    print(f"图片数: {len(images)}")
    
    model = YOLO(YOLO_MODEL_PATH)
    
    coco_results_a = []  # ROI原图
    coco_results_b = []  # 100% patch (无损)
    coco_results_c = []  # 模糊背景 + patch
    
    for img_info in tqdm(images, desc="Step 2"):
        img_id = img_info["id"]
        fname = img_info["file_name"]
        img_path = os.path.join(IMG_DIR, fname)
        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]
        
        # YOLO检测
        results = model(img_path, conf=0.01, verbose=False)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy().tolist()
        dets = [[float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])] for xy in xyxy]
        
        # 计算50% ROI
        roi = compute_roi_50(dets, w, h)
        rx1, ry1, rx2, ry2 = roi
        crop = img[ry1:ry2, rx1:rx2]
        
        # A: ROI原图
        cloud_a = send_cloud(crop, 0.05)
        for d in cloud_a:
            b = d["bbox"]
            coco_results_a.append({
                "image_id": img_id,
                "category_id": d.get("class_id", 0),
                "bbox": [b[0]+rx1, b[1]+ry1, b[2]-b[0], b[3]-b[1]],
                "score": d.get("confidence", 0),
            })
        
        # B: 100% patch无损重建（就是原图，验证一致性）
        recon_b = patch_reconstruct_all(crop)
        cloud_b = send_cloud(recon_b, 0.05)
        for d in cloud_b:
            b = d["bbox"]
            coco_results_b.append({
                "image_id": img_id,
                "category_id": d.get("class_id", 0),
                "bbox": [b[0]+rx1, b[1]+ry1, b[2]-b[0], b[3]-b[1]],
                "score": d.get("confidence", 0),
            })
        
        # C: 模糊背景 + patch
        recon_c = patch_reconstruct_blur(crop)
        cloud_c = send_cloud(recon_c, 0.05)
        for d in cloud_c:
            b = d["bbox"]
            coco_results_c.append({
                "image_id": img_id,
                "category_id": d.get("class_id", 0),
                "bbox": [b[0]+rx1, b[1]+ry1, b[2]-b[0], b[3]-b[1]],
                "score": d.get("confidence", 0),
            })
    
    # 评估
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    
    metrics_a = evaluate(coco_results_a, "A_ROI")
    metrics_b = evaluate(coco_results_b, "B_patch_all")
    metrics_c = evaluate(coco_results_c, "C_patch_blur")
    
    print(f"\n{'方法':<25} {'AP50':<10} {'AP':<10} {'AR':<10}")
    print("-" * 55)
    print(f"{'A: ROI原图 (Baseline)':<25} {metrics_a['AP50']:<10.4f} {metrics_a['AP']:<10.4f} {metrics_a['AR']:<10.4f}")
    print(f"{'B: 100% Patch (无损)':<25} {metrics_b['AP50']:<10.4f} {metrics_b['AP']:<10.4f} {metrics_b['AR']:<10.4f}")
    print(f"{'C: Patch + 模糊背景':<25} {metrics_c['AP50']:<10.4f} {metrics_c['AP']:<10.4f} {metrics_c['AR']:<10.4f}")
    
    if metrics_a['AP50'] > 0:
        print(f"\n相对Baseline保持率:")
        print(f"  B: {metrics_b['AP50']/metrics_a['AP50']*100:.1f}%")
        print(f"  C: {metrics_c['AP50']/metrics_a['AP50']*100:.1f}%")
    
    with open("/app/Salience-DETR/experiments/step2_result.json", "w") as f:
        json.dump({"A": metrics_a, "B": metrics_b, "C": metrics_c}, f, indent=2)

if __name__ == "__main__":
    main()
