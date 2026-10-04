#!/usr/bin/env python3
"""Experiment B: Expansion Ablation
在低分辨率背景下，测试不同Support扩展比例。

B1: expand=0.1
B2: expand=0.2 (当前)
B3: expand=0.3
B4: expand=0.5
B5: adaptive (按目标大小自适应)
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

def make_support_rois(img, dets, expand_mode="0.2"):
    h, w = img.shape[:2]
    supports = []
    for det in dets:
        x1, y1, x2, y2 = det
        bw, bh = x2 - x1, y2 - y1
        
        if expand_mode == "adaptive":
            # 按目标大小自适应: 小目标扩展更多
            area = bw * bh
            if area < 32*32:
                ratio = 0.5
            elif area < 96*96:
                ratio = 0.35
            else:
                ratio = 0.2
        else:
            ratio = float(expand_mode)
        
        sx1 = int(max(0, x1 - bw * ratio))
        sy1 = int(max(0, y1 - bh * ratio))
        sx2 = int(min(w, x2 + bw * ratio))
        sy2 = int(min(h, y2 + bh * ratio))
        supports.append((sx1, sy1, sx2, sy2))
    return supports

def apply_lowres_support(img, supports):
    """低分辨率背景 + Support ROI"""
    h, w = img.shape[:2]
    small_w = max(1, w // LOW_RES_SCALE)
    small_h = max(1, h // LOW_RES_SCALE)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    canvas = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    for sx1, sy1, sx2, sy2 in supports:
        canvas[sy1:sy2, sx1:sx2] = img[sy1:sy2, sx1:sx2]
    
    return canvas

def calc_payload(img, supports):
    """计算传输量: 低分辨率全局 + Support ROI"""
    h, w = img.shape[:2]
    total = 0
    
    # 低分辨率全局
    small_w = max(1, w // LOW_RES_SCALE)
    small_h = max(1, h // LOW_RES_SCALE)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(small).save(buf, format='JPEG', quality=95)
    total += len(buf.getvalue())
    
    # Support ROI
    for sx1, sy1, sx2, sy2 in supports:
        crop = img[sy1:sy2, sx1:sx2]
        buf = io.BytesIO()
        Image.fromarray(crop).save(buf, format='JPEG', quality=95)
        total += len(buf.getvalue())
    
    return total

def evaluate(results_list, name, img_ids_used):
    if not results_list:
        return {"AP50": 0, "AP": 0, "AR": 0, "APsmall": 0, "APmedium": 0, "APlarge": 0, "num_dets": 0}
    
    tmp = f"/tmp/expB_{name}.json"
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
    print("Experiment B: Expansion Ablation (低分辨率背景)")
    print("=" * 70)
    
    with open(GT_PATH) as f:
        gt_coco = json.load(f)
    images = gt_coco["images"][:args.max_images]
    print(f"图片数: {len(images)}")
    
    model = YOLO(YOLO_MODEL_PATH)
    
    expand_modes = ["0.1", "0.2", "0.3", "0.5", "0.7", "adaptive"]
    results = {m: [] for m in expand_modes}
    img_ids_used = set()
    total_payloads = {m: 0 for m in expand_modes}
    
    for img_info in tqdm(images, desc="处理中"):
        img_id = img_info["id"]
        fname = img_info["file_name"]
        img_path = os.path.join(IMG_DIR, fname)
        img = np.array(Image.open(img_path).convert("RGB"))
        
        results_yolo = model(img_path, conf=0.01, verbose=False)
        r = results_yolo[0]
        
        if r.boxes is None or len(r.boxes) == 0:
            img_ids_used.add(int(img_id))
            continue
        
        xyxy = r.boxes.xyxy.cpu().numpy().tolist()
        dets = [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in xyxy]
        
        for mode in expand_modes:
            supports = make_support_rois(img, dets, mode)
            canvas = apply_lowres_support(img, supports)
            cloud_dets = send_cloud(canvas, 0.05)
            
            for d in cloud_dets:
                b = d["bbox"]
                results[mode].append({
                    "image_id": int(img_id),
                    "category_id": int(d.get("class_id", 1)),
                    "bbox": [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])],
                    "score": float(d.get("confidence", 0)),
                })
            
            total_payloads[mode] += calc_payload(img, supports)
        
        img_ids_used.add(int(img_id))
    
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    
    metrics = {}
    for i, mode in enumerate(expand_modes):
        print(f"\n--- B{i+1}: expand={mode} ---")
        metrics[mode] = evaluate(results[mode], mode, img_ids_used)
    
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    
    print(f"\n{'方案':<20} {'AP50':<8} {'AP':<8} {'APsmall':<8} {'APmed':<8} {'APlarge':<8} {'AR':<8} {'检测数':<8}")
    print("-" * 80)
    for i, mode in enumerate(expand_modes):
        m = metrics[mode]
        label = f"B{i+1}: expand={mode}"
        print(f"{label:<20} {m['AP50']:<8.4f} {m['AP']:<8.4f} {m['APsmall']:<8.4f} {m['APmedium']:<8.4f} {m['APlarge']:<8.4f} {m['AR']:<8.4f} {m['num_dets']:<8}")
    
    print(f"\n传输量统计:")
    for i, mode in enumerate(expand_modes):
        avg_bytes = total_payloads[mode] / len(img_ids_used) if img_ids_used else 0
        print(f"  B{i+1} (expand={mode:>8}): {total_payloads[mode]:>12,} bytes (avg {avg_bytes:.0f} bytes/img)")
    
    with open("/app/Salience-DETR/experiments/expB_expansion_ablation.json", "w") as f:
        json.dump({
            "expand_modes": expand_modes,
            "metrics": metrics,
            "payloads_bytes": total_payloads,
            "num_images": len(img_ids_used),
            "low_res_scale": LOW_RES_SCALE,
        }, f, indent=2)
    
    print(f"\n结果已保存: expB_expansion_ablation.json")

if __name__ == "__main__":
    main()
