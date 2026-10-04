#!/usr/bin/env python3
"""全量验证: 最优Support ROI (expand=0.1 + 低分辨率背景) vs 50% ROI基线"""
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
        supports.append((sx1, sy1, sx2, sy2))
    return supports

def make_lowres_support_canvas(img, supports):
    h, w = img.shape[:2]
    small_w = max(1, w // LOW_RES_SCALE)
    small_h = max(1, h // LOW_RES_SCALE)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    canvas = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    for sx1, sy1, sx2, sy2 in supports:
        if sx2 <= sx1 or sy2 <= sy1:
            continue
        canvas[sy1:sy2, sx1:sx2] = img[sy1:sy2, sx1:sx2]
    return canvas

def calc_payload_support(img, supports):
    h, w = img.shape[:2]
    total = 0
    small_w = max(1, w // LOW_RES_SCALE)
    small_h = max(1, h // LOW_RES_SCALE)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(small).save(buf, format='JPEG', quality=95)
    total += len(buf.getvalue())
    for sx1, sy1, sx2, sy2 in supports:
        if sx2 <= sx1 or sy2 <= sy1:
            continue
        crop = img[sy1:sy2, sx1:sx2]
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
    print("全量验证: Support ROI (expand=0.1 + lowres) vs 50% ROI")
    print("=" * 70)
    
    with open(GT_PATH) as f:
        gt_coco = json.load(f)
    images = gt_coco["images"][:args.max_images]
    print(f"图片数: {len(images)}")
    
    model = YOLO(YOLO_MODEL_PATH)
    
    results_roi50 = []
    results_support = []
    img_ids_used = set()
    total_payload_roi50 = 0
    total_payload_support = 0
    
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
        
        # 方案1: 50% ROI
        roi = compute_roi_50(dets, w, h)
        rx1, ry1, rx2, ry2 = roi
        crop_50 = img[ry1:ry2, rx1:rx2]
        total_payload_roi50 += calc_payload_roi50(crop_50)
        
        cloud_50 = send_cloud(crop_50, 0.05)
        for d in cloud_50:
            b = d["bbox"]
            results_roi50.append({
                "image_id": int(img_id),
                "category_id": int(d.get("class_id", 1)),
                "bbox": [float(b[0]+rx1), float(b[1]+ry1), float(b[2]-b[0]), float(b[3]-b[1])],
                "score": float(d.get("confidence", 0)),
            })
        
        # 方案2: Support ROI + lowres
        supports = make_support_rois(img, dets)
        canvas = make_lowres_support_canvas(img, supports)
        total_payload_support += calc_payload_support(img, supports)
        
        cloud_sup = send_cloud(canvas, 0.05)
        for d in cloud_sup:
            b = d["bbox"]
            results_support.append({
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
    
    print("\n--- Support ROI (expand=0.1 + lowres) ---")
    m_support = evaluate(results_support, "support", img_ids_used)
    
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    
    print(f"\n{'方案':<30} {'AP50':<8} {'AP':<8} {'AP75':<8} {'APsmall':<8} {'APmed':<8} {'APlarge':<8} {'AR':<8} {'检测数':<8}")
    print("-" * 100)
    print(f"{'50% ROI':<30} {m_roi50['AP50']:<8.4f} {m_roi50['AP']:<8.4f} {m_roi50['AP75']:<8.4f} {m_roi50['APsmall']:<8.4f} {m_roi50['APmedium']:<8.4f} {m_roi50['APlarge']:<8.4f} {m_roi50['AR']:<8.4f} {m_roi50['num_dets']:<8}")
    print(f"{'Support+lowres (exp=0.1)':<30} {m_support['AP50']:<8.4f} {m_support['AP']:<8.4f} {m_support['AP75']:<8.4f} {m_support['APsmall']:<8.4f} {m_support['APmedium']:<8.4f} {m_support['APlarge']:<8.4f} {m_support['AR']:<8.4f} {m_support['num_dets']:<8}")
    
    print(f"\n传输量统计:")
    print(f"  50% ROI:       {total_payload_roi50:>14,} bytes (avg {total_payload_roi50/len(img_ids_used):.0f} bytes/img)")
    print(f"  Support+lowres: {total_payload_support:>14,} bytes (avg {total_payload_support/len(img_ids_used):.0f} bytes/img)")
    print(f"  压缩比:         {total_payload_roi50/total_payload_support:.1f}x")
    
    if m_roi50['AP50'] > 0:
        print(f"\nSupport ROI 相对 50% ROI:")
        print(f"  AP50保持率: {m_support['AP50']/m_roi50['AP50']*100:.1f}%")
        print(f"  mAP保持率: {m_support['AP']/m_roi50['AP']*100:.1f}%")
        print(f"  AR保持率:  {m_support['AR']/m_roi50['AR']*100:.1f}%")
    
    with open("/app/Salience-DETR/experiments/final_support_result.json", "w") as f:
        json.dump({
            "roi50": m_roi50,
            "support_lowres": m_support,
            "payload_roi50_bytes": total_payload_roi50,
            "payload_support_bytes": total_payload_support,
            "num_images": len(img_ids_used),
            "config": {
                "expand_ratio": EXPAND_RATIO,
                "low_res_scale": LOW_RES_SCALE,
            },
        }, f, indent=2)
    
    print(f"\n结果已保存: final_support_result.json")

if __name__ == "__main__":
    main()
