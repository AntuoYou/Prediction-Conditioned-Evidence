#!/usr/bin/env python3
"""Experiment A: Context Ablation
保持Support ROI位置、大小不变，只改变未传输区域的背景处理方式。

A1: Support + 灰色背景
A2: Support + 高斯模糊背景
A3: Support + 低分辨率背景
A4: Support + 原图背景 (上界)
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
EXPAND_RATIO = 0.2
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

def apply_support(img, supports, bg_mode="gray"):
    """根据背景策略生成发送给云端的图像"""
    if bg_mode == "full":
        return img.copy()
    
    h, w = img.shape[:2]
    
    if bg_mode == "gray":
        canvas = np.full_like(img, 114)
    elif bg_mode == "blur":
        canvas = cv2.GaussianBlur(img, (31, 31), 0)
    elif bg_mode == "lowres":
        small_w = max(1, w // LOW_RES_SCALE)
        small_h = max(1, h // LOW_RES_SCALE)
        small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
        canvas = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError(f"Unknown bg_mode: {bg_mode}")
    
    for sx1, sy1, sx2, sy2 in supports:
        canvas[sy1:sy2, sx1:sx2] = img[sy1:sy2, sx1:sx2]
    
    return canvas

def calc_payload_bytes(img, supports, bg_mode):
    """计算传输量 (bytes)"""
    h, w = img.shape[:2]
    
    if bg_mode == "full":
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format='JPEG', quality=95)
        return len(buf.getvalue())
    
    if bg_mode == "gray":
        total = 0
        for sx1, sy1, sx2, sy2 in supports:
            crop = img[sy1:sy2, sx1:sx2]
            buf = io.BytesIO()
            Image.fromarray(crop).save(buf, format='JPEG', quality=95)
            total += len(buf.getvalue())
        return total
    
    if bg_mode == "blur":
        canvas = apply_support(img, supports, "blur")
        buf = io.BytesIO()
        Image.fromarray(canvas).save(buf, format='JPEG', quality=95)
        return len(buf.getvalue())
    
    if bg_mode == "lowres":
        total = 0
        small_w = max(1, w // LOW_RES_SCALE)
        small_h = max(1, h // LOW_RES_SCALE)
        small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
        buf = io.BytesIO()
        Image.fromarray(small).save(buf, format='JPEG', quality=95)
        total += len(buf.getvalue())
        for sx1, sy1, sx2, sy2 in supports:
            crop = img[sy1:sy2, sx1:sx2]
            buf = io.BytesIO()
            Image.fromarray(crop).save(buf, format='JPEG', quality=95)
            total += len(buf.getvalue())
        return total
    
    return 0

def evaluate(results_list, name, img_ids_used):
    if not results_list:
        return {"AP50": 0, "AP": 0, "AR": 0, "APsmall": 0, "APmedium": 0, "APlarge": 0, "num_dets": 0}
    
    tmp = f"/tmp/expA_{name}.json"
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
            "AR": float(e.stats[8]),
            "APsmall": float(e.stats[3]),
            "APmedium": float(e.stats[4]),
            "APlarge": float(e.stats[5]),
            "num_dets": len(results_list),
        }
    except Exception as ex:
        print(f"  {name} 评估失败: {ex}")
        return {"AP50": 0, "AP": 0, "AR": 0, "APsmall": 0, "APmedium": 0, "APlarge": 0, "num_dets": len(results_list)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=200)
    args = parser.parse_args()
    
    print("=" * 70)
    print("Experiment A: Context Ablation - 背景策略对比")
    print("=" * 70)
    
    with open(GT_PATH) as f:
        gt_coco = json.load(f)
    images = gt_coco["images"][:args.max_images]
    print(f"图片数: {len(images)}")
    print(f"背景策略: A1=gray, A2=blur, A3=lowres, A4=full")
    print(f"Support扩展: {EXPAND_RATIO}, 低分辨率缩放: 1/{LOW_RES_SCALE}")
    
    model = YOLO(YOLO_MODEL_PATH)
    
    strategies = ["gray", "blur", "lowres", "full"]
    results = {s: [] for s in strategies}
    img_ids_used = set()
    total_payloads = {s: 0 for s in strategies}
    
    for img_info in tqdm(images, desc="处理中"):
        img_id = img_info["id"]
        fname = img_info["file_name"]
        img_path = os.path.join(IMG_DIR, fname)
        img = np.array(Image.open(img_path).convert("RGB"))
        
        results_yolo = model(img_path, conf=0.01, verbose=False)
        r = results_yolo[0]
        
        has_dets = r.boxes is not None and len(r.boxes) > 0
        
        if has_dets:
            xyxy = r.boxes.xyxy.cpu().numpy().tolist()
            dets = [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in xyxy]
            supports = make_support_rois(img, dets)
        else:
            supports = []
        
        for s in strategies:
            canvas = apply_support(img, supports, s)
            cloud_dets = send_cloud(canvas, 0.05)
            
            for d in cloud_dets:
                b = d["bbox"]
                results[s].append({
                    "image_id": int(img_id),
                    "category_id": int(d.get("class_id", 1)),
                    "bbox": [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])],
                    "score": float(d.get("confidence", 0)),
                })
            
            total_payloads[s] += calc_payload_bytes(img, supports, s)
        
        img_ids_used.add(int(img_id))
    
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    
    metrics = {}
    for i, s in enumerate(strategies):
        print(f"\n--- A{i+1}: Support + {s} ---")
        metrics[s] = evaluate(results[s], s, img_ids_used)
    
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    
    print(f"\n{'方案':<25} {'AP50':<8} {'AP':<8} {'APsmall':<8} {'APmed':<8} {'APlarge':<8} {'AR':<8} {'检测数':<8}")
    print("-" * 85)
    for i, s in enumerate(strategies):
        m = metrics[s]
        label = f"A{i+1}: Support+{s}"
        print(f"{label:<25} {m['AP50']:<8.4f} {m['AP']:<8.4f} {m['APsmall']:<8.4f} {m['APmedium']:<8.4f} {m['APlarge']:<8.4f} {m['AR']:<8.4f} {m['num_dets']:<8}")
    
    full_bytes = total_payloads["full"]
    print(f"\n传输量统计:")
    for i, s in enumerate(strategies):
        pct = total_payloads[s] / full_bytes * 100 if full_bytes > 0 else 0
        print(f"  A{i+1} ({s:>6}): {total_payloads[s]:>10,} bytes ({pct:5.1f}% of full)")
    
    with open("/app/Salience-DETR/experiments/expA_context_ablation.json", "w") as f:
        json.dump({
            "strategies": strategies,
            "metrics": metrics,
            "payloads_bytes": total_payloads,
            "num_images": len(img_ids_used),
            "expand_ratio": EXPAND_RATIO,
            "low_res_scale": LOW_RES_SCALE,
        }, f, indent=2)
    
    print(f"\n结果已保存: expA_context_ablation.json")

if __name__ == "__main__":
    main()
