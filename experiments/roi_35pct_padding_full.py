#!/usr/bin/env python3
"""
ROI 35% + Padding 全量测试 (1547 张图)
========================================
在全量验证集上测试最佳策略：
  - 裁剪 35% 面积（以所有 GT bbox 整体中心为中心，每图只裁一次）
  - 灰底 padding 到短边 800px（保持宽高比，crop 不缩放）
  - 上传云端 Salience-DETR 推理
  - 坐标映射回 GT 空间
  - Subset GT COCO 评估

坐标空间转换链：
  GT空间(3840×2160) → 实际图片空间 → 裁剪局部空间 → Canvas空间
                                                    ↓ 云端检测
  GT空间 ← 实际图片空间 ← 裁剪局部空间 ← Canvas空间
"""

import json
import os
import time
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
from collections import defaultdict

# ---------- 配置 ----------
CLOUD_SERVER_URL = os.environ.get("CLOUD_SERVER_URL", "http://localhost:8000")
GT_ANNOTATION_PATH = os.environ.get(
    "GT_ANNOTATION_PATH",
    "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json",
)
IMAGE_DIR = os.environ.get(
    "IMAGE_DIR",
    "/app/data/seadronessee/compressed_version/val2017/",
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "/app/Salience-DETR/experiments/roi_35pct_padding_full_results/",
)
CONF_THRESHOLD = 0.05
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "0"))
SCALE_FACTOR = 0.35
TARGET_SHORT_SIDE = 800
CHECKPOINT_INTERVAL = 100

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
    img = Image.open(image_path).convert("RGB")
    return np.array(img)

def gt_bbox_to_actual(bbox_xywh, gt_w, gt_h, actual_w, actual_h):
    if gt_w == actual_w and gt_h == actual_h:
        return list(bbox_xywh)
    sx = actual_w / gt_w
    sy = actual_h / gt_h
    return [bbox_xywh[0]*sx, bbox_xywh[1]*sy, bbox_xywh[2]*sx, bbox_xywh[3]*sy]

def scale_bbox_to_gt(bbox_xywh, actual_w, actual_h, gt_w, gt_h):
    if actual_w == gt_w and actual_h == gt_h:
        return list(bbox_xywh)
    sx = gt_w / actual_w
    sy = gt_h / actual_h
    return [bbox_xywh[0]*sx, bbox_xywh[1]*sy, bbox_xywh[2]*sx, bbox_xywh[3]*sy]

def create_padded_crop(img_array, bboxes_actual, scale_factor=SCALE_FACTOR,
                        target_short_side=TARGET_SHORT_SIDE):
    h, w = img_array.shape[:2]
    all_x1 = [b[0] for b in bboxes_actual]
    all_y1 = [b[1] for b in bboxes_actual]
    all_x2 = [b[0]+b[2] for b in bboxes_actual]
    all_y2 = [b[1]+b[3] for b in bboxes_actual]
    cx = (min(all_x1) + max(all_x2)) / 2
    cy = (min(all_y1) + max(all_y2)) / 2
    scale_sqrt = np.sqrt(scale_factor)
    crop_w = int(w * scale_sqrt)
    crop_h = int(h * scale_sqrt)
    x1 = int(cx - crop_w / 2)
    y1 = int(cy - crop_h / 2)
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    if x2 > w:
        x2 = w
        x1 = max(0, x2 - crop_w)
    if y2 > h:
        y2 = h
        y1 = max(0, y2 - crop_h)
    crop = img_array[y1:y2, x1:x2].copy()
    crop_h_actual, crop_w_actual = crop.shape[:2]
    short_side = min(crop_w_actual, crop_h_actual)
    if short_side < target_short_side:
        pad_scale = target_short_side / short_side
        padded_w = int(crop_w_actual * pad_scale)
        padded_h = int(crop_h_actual * pad_scale)
    else:
        padded_w = crop_w_actual
        padded_h = crop_h_actual
    mean_val = img_array.mean(axis=(0, 1)).astype(np.uint8)
    canvas = np.full((padded_h, padded_w, 3), mean_val, dtype=np.uint8)
    pad_left = (padded_w - crop_w_actual) // 2
    pad_top = (padded_h - crop_h_actual) // 2
    canvas[pad_top:pad_top+crop_h_actual, pad_left:pad_left+crop_w_actual] = crop
    return canvas, (x1, y1, x2, y2), (pad_left, pad_top)

def image_to_bytes(img_array):
    img = Image.fromarray(img_array)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()

def send_to_cloud(img_array, conf_threshold=CONF_THRESHOLD, max_retries=3):
    img_bytes = image_to_bytes(img_array)
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{CLOUD_SERVER_URL}/detect",
                files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                data={"conf_threshold": str(conf_threshold)},
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                detections = []
                if "results" in data:
                    for item in data["results"]:
                        if "detections" in item:
                            for det in item["detections"]:
                                detections.append(det)
                elif "detections" in data:
                    detections = data["detections"]
                elif isinstance(data, list):
                    detections = data
                return detections
            else:
                if attempt < max_retries - 1:
                    time.sleep(1)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"  [失败] {e}")
                return []
    return []

def parse_bbox(bbox):
    if isinstance(bbox, list):
        return bbox[0], bbox[1], bbox[2], bbox[3]
    elif isinstance(bbox, dict):
        return bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    raise ValueError(f"Unknown bbox format: {type(bbox)}")

def evaluate_coco_subset(gt_coco, results):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if not results:
        return {"AP50": 0.0, "AP": 0.0, "AR": 0.0, "stats": [0]*12}
    pred_image_ids = {d["image_id"] for d in results}
    subset_images = [img for img in gt_coco["images"] if img["id"] in pred_image_ids]
    subset_anns = [ann for ann in gt_coco["annotations"] if ann["image_id"] in pred_image_ids]
    subset_gt = {"images": subset_images, "annotations": subset_anns, "categories": gt_coco["categories"]}
    print(f"\n  Subset GT: {len(subset_images)} 张图片, {len(subset_anns)} 个标注")
    coco_gt = COCO()
    coco_gt.dataset = subset_gt
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {
        "AP50": float(coco_eval.stats[1]),
        "AP": float(coco_eval.stats[0]),
        "AP75": float(coco_eval.stats[2]),
        "AR": float(coco_eval.stats[8]),
        "AR1": float(coco_eval.stats[6]),
        "AR10": float(coco_eval.stats[7]),
        "AP_small": float(coco_eval.stats[3]),
        "AP_medium": float(coco_eval.stats[4]),
        "AP_large": float(coco_eval.stats[5]),
        "AR_small": float(coco_eval.stats[9]),
        "AR_medium": float(coco_eval.stats[10]),
        "AR_large": float(coco_eval.stats[11]),
        "stats": [float(s) for s in coco_eval.stats],
        "subset_images": len(subset_images),
        "subset_anns": len(subset_anns),
    }

def save_checkpoint(results, output_dir, processed_count):
    ckpt_path = os.path.join(output_dir, "checkpoint.json")
    with open(ckpt_path, "w") as f:
        json.dump({"processed": processed_count, "results": results}, f)

def run_experiment(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gt_coco, image_map, ann_map = load_gt_annotations(GT_ANNOTATION_PATH)
    image_ids = sorted(ann_map.keys())
    if MAX_IMAGES > 0:
        image_ids = image_ids[:MAX_IMAGES]
    print(f"[实验] 处理 {len(image_ids)} 张图片 | 裁剪比例: {SCALE_FACTOR:.0%} | "
          f"Padding: {TARGET_SHORT_SIDE}px | 置信度阈值: {CONF_THRESHOLD}\n")

    results = []
    processed_count = 0
    fail_count = 0
    total_inference_time = 0
    total_upload_bytes = 0
    total_full_bytes = 0

    for img_id in tqdm(image_ids, desc="处理图片"):
        img_info = image_map[img_id]
        img_path = os.path.join(IMAGE_DIR, img_info["file_name"])
        if not os.path.exists(img_path):
            fail_count += 1
            continue
        try:
            img_array = load_image(img_path)
        except Exception as e:
            print(f"  [跳过] {img_info['file_name']}: {e}")
            fail_count += 1
            continue
        actual_h, actual_w = img_array.shape[:2]
        gt_w = img_info["width"]
        gt_h = img_info["height"]
        annotations = ann_map[img_id]
        bboxes_gt = [ann["bbox"] for ann in annotations]
        bboxes_actual = [gt_bbox_to_actual(b, gt_w, gt_h, actual_w, actual_h) for b in bboxes_gt]
        canvas, crop_pos, pad_offset = create_padded_crop(
            img_array, bboxes_actual, SCALE_FACTOR, TARGET_SHORT_SIDE)
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_pos
        pad_left, pad_top = pad_offset
        upload_bytes = len(image_to_bytes(canvas))
        full_bytes = len(image_to_bytes(img_array))
        total_upload_bytes += upload_bytes
        total_full_bytes += full_bytes
        t0 = time.time()
        detections = send_to_cloud(canvas)
        infer_time = time.time() - t0
        total_inference_time += infer_time
        for det in detections:
            bx1, by1, bx2, by2 = parse_bbox(det["bbox"])
            score = det.get("score", det.get("confidence", 0.0))
            cls_id = det.get("class_id", det.get("category_id", 0))
            if score < CONF_THRESHOLD:
                continue
            crop_local_x1 = bx1 - pad_left
            crop_local_y1 = by1 - pad_top
            crop_local_x2 = bx2 - pad_left
            crop_local_y2 = by2 - pad_top
            actual_x1 = crop_x1 + crop_local_x1
            actual_y1 = crop_y1 + crop_local_y1
            actual_x2 = crop_x1 + crop_local_x2
            actual_y2 = crop_y1 + crop_local_y2
            gt_bbox = scale_bbox_to_gt(
                [actual_x1, actual_y1, actual_x2-actual_x1, actual_y2-actual_y1],
                actual_w, actual_h, gt_w, gt_h)
            if gt_bbox[2] <= 0 or gt_bbox[3] <= 0:
                continue
            results.append({
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": gt_bbox,
                "score": float(score),
            })
        processed_count += 1
        if processed_count % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(results, OUTPUT_DIR, processed_count)

    print("\n" + "=" * 70)
    print(f"ROI 35% + Padding 全量测试 — 评估结果")
    print(f"裁剪比例: {SCALE_FACTOR:.0%} | Padding: {TARGET_SHORT_SIDE}px | 置信度阈值: {CONF_THRESHOLD}")
    print("=" * 70)
    metrics = evaluate_coco_subset(gt_coco, results)
    print(f"\n{'─'*50}")
    print(f"  处理图片数:       {processed_count}")
    print(f"  失败/跳过:        {fail_count}")
    print(f"  总检测数:         {len(results)}")
    print(f"  总推理时间:       {total_inference_time:.1f}s")
    print(f"  平均推理时间:     {total_inference_time/max(processed_count,1):.3f}s/图")
    print(f"  Subset GT 图片:   {metrics.get('subset_images', '?')}")
    print(f"  Subset GT 标注:   {metrics.get('subset_anns', '?')}")
    print(f"{'─'*50}")
    print(f"  AP   @[0.50:0.95 | all]:    {metrics['AP']:.4f}")
    print(f"  AP50 @[0.50      | all]:    {metrics['AP50']:.4f}")
    print(f"  AP75 @[0.75      | all]:    {metrics['AP75']:.4f}")
    print(f"  AR   @[0.50:0.95 | maxDets=1]:   {metrics['AR1']:.4f}")
    print(f"  AR   @[0.50:0.95 | maxDets=10]:  {metrics['AR10']:.4f}")
    print(f"  AR   @[0.50:0.95 | maxDets=100]: {metrics['AR']:.4f}")
    print(f"{'─'*50}")
    print(f"  AP   [small]:     {metrics['AP_small']:.4f}")
    print(f"  AP   [medium]:    {metrics['AP_medium']:.4f}")
    print(f"  AP   [large]:     {metrics['AP_large']:.4f}")
    print(f"  AR   [small]:     {metrics['AR_small']:.4f}")
    print(f"  AR   [medium]:    {metrics['AR_medium']:.4f}")
    print(f"  AR   [large]:     {metrics['AR_large']:.4f}")
    print(f"{'─'*50}")
    sample_50_ap50 = 0.8337
    print(f"\n  与 50 图小样本对比:")
    print(f"     50图 AP50:   {sample_50_ap50:.4f}")
    print(f"     全量 AP50:   {metrics['AP50']:.4f}")
    diff = metrics['AP50'] - sample_50_ap50
    print(f"     变化:        {'↑' if diff>=0 else '↓'} {abs(diff):.4f}")
    full_ap50 = 0.858
    print(f"\n  与云端全图基线对比:")
    print(f"     全图 AP50:   {full_ap50:.4f}")
    print(f"     ROI AP50:    {metrics['AP50']:.4f}")
    recovery = metrics['AP50'] / full_ap50 if full_ap50 > 0 else 0
    print(f"     恢复率:      {recovery:.1%}")
    avg_upload = total_upload_bytes / max(processed_count, 1)
    avg_full = total_full_bytes / max(processed_count, 1)
    bw_ratio = avg_upload / avg_full if avg_full > 0 else 0
    print(f"\n  带宽分析:")
    print(f"     平均上传 (35%+Pad): {avg_upload/1024:.1f} KB")
    print(f"     平均全图:           {avg_full/1024:.1f} KB")
    print(f"     带宽比例:           {bw_ratio:.1%}")
    print(f"     带宽节省:           {(1-bw_ratio):.1%}")
    output_path = os.path.join(OUTPUT_DIR, "roi_35pct_padding_full_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "metrics": {k: v for k, v in metrics.items() if k != "stats"},
            "coco_stats": metrics.get("stats", []),
            "config": {"scale_factor": SCALE_FACTOR, "target_short_side": TARGET_SHORT_SIDE,
                       "conf_threshold": CONF_THRESHOLD, "max_images": MAX_IMAGES,
                       "processed_images": processed_count, "server": CLOUD_SERVER_URL},
            "stats": {"total_detections": len(results), "total_inference_time": total_inference_time,
                      "avg_inference_time": total_inference_time/max(processed_count,1),
                      "avg_upload_kb": avg_upload/1024, "avg_full_kb": avg_full/1024,
                      "bandwidth_ratio": bw_ratio, "bandwidth_saving": 1-bw_ratio},
        }, f, indent=2)
    print(f"\n[保存] 结果 → {output_path}")
    pred_path = os.path.join(OUTPUT_DIR, "predictions_35pct_padding.json")
    with open(pred_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[保存] 预测结果 → {pred_path}")
    ckpt_path = os.path.join(OUTPUT_DIR, "checkpoint.json")
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"[清理] checkpoint 已删除")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROI 35% + Padding 全量测试")
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES, help="最多处理图片数 (0=全部)")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="置信度阈值")
    parser.add_argument("--server", type=str, default=CLOUD_SERVER_URL, help="云端服务器地址")
    args = parser.parse_args()
    MAX_IMAGES = args.max_images
    CONF_THRESHOLD = args.conf
    CLOUD_SERVER_URL = args.server
    run_experiment(args)
