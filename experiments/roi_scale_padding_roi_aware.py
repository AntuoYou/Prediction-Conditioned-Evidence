#!/usr/bin/env python3
"""
ROI 裁剪比例 + Padding 对比实验 — ROI-aware 微调后模型 (全量/子集)
==================================================================
在全量验证集上测试 10%-40% 裁剪比例，对比有无 Padding 的效果。

实验设计：
  - 裁剪比例: 10%, 15%, 20%, 25%, 30%, 35%, 40%
  - 两种模式: 无 Padding / 有 Padding (灰底 pad 到短边 800px)
  - GT 决定裁剪位置（以所有 GT bbox 整体中心为中心）
  - 使用 ROI-aware 微调后的云端模型

坐标空间转换链：
  GT空间 → 实际图片空间 → 裁剪局部空间 → [Canvas空间 (仅padding模式)]
                                                ↓ 云端检测
  GT空间 ← 实际图片空间 ← 裁剪局部空间 ← [Canvas空间]

用法：
  python roi_scale_padding_roi_aware.py --server http://localhost:8000
  python roi_scale_padding_roi_aware.py --server http://localhost:8000 --max-images 50  # 快速测试

依赖：
  pip install requests pycocotools pillow numpy tqdm --break-system-packages
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
CLOUD_SERVER_URL = "http://localhost:8000"
GT_ANNOTATION_PATH = "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"
IMAGE_DIR = "/app/data/seadronessee/compressed_version/val2017/"
OUTPUT_DIR = "/app/Salience-DETR/experiments/roi_scale_padding_roi_aware_results/"

CONF_THRESHOLD = 0.05
TARGET_SHORT_SIDE = 800  # Padding 目标短边
CHECKPOINT_INTERVAL = 100

# 裁剪比例档位
SCALE_FACTORS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

# 微调前 50 图基线（用于对比）
BEFORE_FINETUNE_50IMG = {
    "no_pad": {
        0.10: 0.614, 0.15: 0.684, 0.20: 0.706, 0.25: 0.678,
        0.30: 0.699, 0.35: 0.805, 0.40: 0.839,
    },
    "with_pad": {
        0.10: 0.676, 0.15: 0.752, 0.20: 0.770, 0.25: 0.778,
        0.30: 0.786, 0.35: 0.834, 0.40: 0.821,
    },
}

# ---------- 工具函数 ----------

def load_gt_annotations(gt_path):
    print(f"[加载] GT 标注: {gt_path}")
    with open(gt_path, "r") as f:
        coco = json.load(f)
    image_map = {img["id"]: img for img in coco["images"]}
    ann_map = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("bbox") and ann["bbox"][2] > 0 and ann["bbox"][3] > 0:
            ann_map[ann["image_id"]].append(ann)
    print(f"  -> {len(image_map)} 张图片, {sum(len(v) for v in ann_map.values())} 个标注")
    return coco, image_map, ann_map

def load_image(image_path):
    return np.array(Image.open(image_path).convert("RGB"))

def gt_bbox_to_actual(bbox_xywh, gt_w, gt_h, actual_w, actual_h):
    """GT 空间 → 实际图片空间"""
    if gt_w == actual_w and gt_h == actual_h:
        return list(bbox_xywh)
    sx = actual_w / gt_w
    sy = actual_h / gt_h
    return [bbox_xywh[0] * sx, bbox_xywh[1] * sy, bbox_xywh[2] * sx, bbox_xywh[3] * sy]

def scale_bbox_to_gt(bbox_xywh, actual_w, actual_h, gt_w, gt_h):
    """实际图片空间 → GT 空间"""
    if actual_w == gt_w and actual_h == gt_h:
        return list(bbox_xywh)
    sx = gt_w / actual_w
    sy = gt_h / actual_h
    return [bbox_xywh[0] * sx, bbox_xywh[1] * sy, bbox_xywh[2] * sx, bbox_xywh[3] * sy]

def xyxy_to_coco_xywh(x1, y1, x2, y2):
    return [x1, y1, x2 - x1, y2 - y1]

def coco_to_xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]

def compute_crop_region(bboxes_xyxy, w, h, scale_factor):
    """以所有 bbox 整体中心为中心，裁剪 scale_factor 面积子图（保持宽高比）"""
    all_x1 = [b[0] for b in bboxes_xyxy]
    all_y1 = [b[1] for b in bboxes_xyxy]
    all_x2 = [b[2] for b in bboxes_xyxy]
    all_y2 = [b[3] for b in bboxes_xyxy]
    cx = (min(all_x1) + max(all_x2)) / 2
    cy = (min(all_y1) + max(all_y2)) / 2
    ss = np.sqrt(scale_factor)
    cw, ch = int(w * ss), int(h * ss)
    x1, y1 = int(cx - cw / 2), int(cy - ch / 2)
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    x2, y2 = x1 + cw, y1 + ch
    if x2 > w:
        x2 = w
        x1 = max(0, x2 - cw)
    if y2 > h:
        y2 = h
        y1 = max(0, y2 - ch)
    return x1, y1, x2, y2

def create_padded_canvas(crop, img_array):
    """
    将裁剪图用灰底 padding 到短边 800px。
    crop 本身不缩放，只是放在更大的灰底画布中。

    返回:
      canvas: padding 后的图片
      pad_left, pad_top: 裁剪图在 canvas 中的偏移
    """
    crop_h, crop_w = crop.shape[:2]
    short_side = min(crop_w, crop_h)

    if short_side < TARGET_SHORT_SIDE:
        pad_scale = TARGET_SHORT_SIDE / short_side
        padded_w = int(crop_w * pad_scale)
        padded_h = int(crop_h * pad_scale)
    else:
        padded_w = crop_w
        padded_h = crop_h

    # 灰底画布（用原图均值填充，模拟训练分布）
    mean_val = img_array.mean(axis=(0, 1)).astype(np.uint8)
    canvas = np.full((padded_h, padded_w, 3), mean_val, dtype=np.uint8)

    # 居中放置裁剪图（不缩放，原始分辨率）
    pad_left = (padded_w - crop_w) // 2
    pad_top = (padded_h - crop_h) // 2
    canvas[pad_top:pad_top + crop_h, pad_left:pad_left + crop_w] = crop

    return canvas, pad_left, pad_top

def image_to_bytes(img_array):
    buf = BytesIO()
    Image.fromarray(img_array).save(buf, format="JPEG", quality=95)
    return buf.getvalue()

def send_to_cloud(img_array, conf=CONF_THRESHOLD, retries=3):
    ib = image_to_bytes(img_array)
    for a in range(retries):
        try:
            r = requests.post(
                f"{CLOUD_SERVER_URL}/detect",
                files={"file": ("image.jpg", ib, "image/jpeg")},
                data={"conf_threshold": str(conf)},
                timeout=60,
            )
            if r.status_code == 200:
                d = r.json()
                if "results" in d:
                    return [det for item in d["results"] for det in item.get("detections", [])]
                return d.get("detections", d if isinstance(d, list) else [])
            if a < retries - 1:
                time.sleep(1)
        except Exception as e:
            if a < retries - 1:
                time.sleep(2)
            else:
                print(f"  [云端失败] {e}")
                return []
    return []

def parse_bbox(b):
    if isinstance(b, list):
        return b[0], b[1], b[2], b[3]
    if isinstance(b, dict):
        return b["x1"], b["y1"], b["x2"], b["y2"]
    raise ValueError(str(type(b)))

def evaluate_coco_subset(gt_coco, results):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not results:
        return {
            "AP50": 0.0, "AP": 0.0, "AP75": 0.0, "AR": 0.0,
            "AR1": 0.0, "AR10": 0.0,
            "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0,
            "AR_small": 0.0, "AR_medium": 0.0, "AR_large": 0.0,
            "stats": [0] * 12,
            "subset_images": 0, "subset_anns": 0,
        }

    pred_image_ids = {d["image_id"] for d in results}
    subset_images = [img for img in gt_coco["images"] if img["id"] in pred_image_ids]
    subset_anns = [ann for ann in gt_coco["annotations"] if ann["image_id"] in pred_image_ids]
    subset_gt = {
        "images": subset_images,
        "annotations": subset_anns,
        "categories": gt_coco["categories"],
    }

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

# ---------- 单次实验 ----------

def run_single_experiment(gt_coco, image_map, ann_map, scale_factor, use_padding,
                          max_images=0):
    """
    测试单个 (裁剪比例, padding模式) 组合。

    Args:
        scale_factor: 裁剪面积比例 (如 0.10)
        use_padding: 是否使用灰底 padding
        max_images: 最多处理图片数 (0=全部)
    """
    mode = "with_pad" if use_padding else "no_pad"
    image_ids = sorted(ann_map.keys())
    if max_images > 0:
        image_ids = image_ids[:max_images]

    results = []
    total_upload = 0
    total_full = 0
    inference_times = []
    processed = 0

    for img_id in tqdm(image_ids, desc=f"{scale_factor:.0%}_{mode}"):
        img_info = image_map[img_id]
        img_path = os.path.join(IMAGE_DIR, img_info["file_name"])
        if not os.path.exists(img_path):
            continue
        try:
            img_array = load_image(img_path)
        except:
            continue

        actual_h, actual_w = img_array.shape[:2]
        gt_w, gt_h = img_info["width"], img_info["height"]
        gt_anns = ann_map[img_id]

        # GT bbox → 实际图片空间 xyxy
        gt_xyxy_actual = []
        for ann in gt_anns:
            b = gt_bbox_to_actual(ann["bbox"], gt_w, gt_h, actual_w, actual_h)
            gt_xyxy_actual.append([b[0], b[1], b[0] + b[2], b[1] + b[3]])

        # 裁剪区域
        cx1, cy1, cx2, cy2 = compute_crop_region(gt_xyxy_actual, actual_w, actual_h, scale_factor)
        crop = img_array[cy1:cy2, cx1:cx2].copy()

        if use_padding:
            # 灰底 padding 到短边 800px
            canvas, pad_left, pad_top = create_padded_canvas(crop, img_array)
            send_img = canvas
        else:
            # 直接裁剪上传
            send_img = crop
            pad_left, pad_top = 0, 0

        # 上传大小统计
        total_upload += len(image_to_bytes(send_img))
        total_full += len(image_to_bytes(img_array))

        # 云端推理
        t0 = time.time()
        cloud_dets = send_to_cloud(send_img, conf=CONF_THRESHOLD)
        t1 = time.time()
        inference_times.append(t1 - t0)

        # 坐标映射: 发送图空间 → 裁剪局部 → 实际图片 → GT空间
        for det in cloud_dets:
            bx1, by1, bx2, by2 = parse_bbox(det["bbox"])
            score = det.get("score", det.get("confidence", 0.0))
            cls_id = det.get("class_id", det.get("category_id", 0))
            if score < CONF_THRESHOLD:
                continue

            # 发送图空间 → 裁剪局部（减去 padding 偏移）
            local_x1 = bx1 - pad_left
            local_y1 = by1 - pad_top
            local_x2 = bx2 - pad_left
            local_y2 = by2 - pad_top

            # 裁剪局部 → 实际图片空间（加上裁剪起点）
            actual_x1 = cx1 + local_x1
            actual_y1 = cy1 + local_y1
            actual_x2 = cx1 + local_x2
            actual_y2 = cy1 + local_y2

            # 实际图片空间 → GT 空间
            xywh = xyxy_to_coco_xywh(actual_x1, actual_y1, actual_x2, actual_y2)
            gt_bbox = scale_bbox_to_gt(xywh, actual_w, actual_h, gt_w, gt_h)
            if gt_bbox[2] <= 0 or gt_bbox[3] <= 0:
                continue

            results.append({
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": gt_bbox,
                "score": float(score),
            })

        processed += 1

    # 评估
    metrics = evaluate_coco_subset(gt_coco, results)
    metrics["avg_upload_kb"] = total_upload / max(processed, 1) / 1024
    metrics["avg_full_kb"] = total_full / max(processed, 1) / 1024
    metrics["bandwidth_save"] = 1 - metrics["avg_upload_kb"] / max(metrics["avg_full_kb"], 1)
    metrics["avg_inference_time"] = float(np.mean(inference_times)) if inference_times else 0
    metrics["scale_factor"] = scale_factor
    metrics["use_padding"] = use_padding
    metrics["num_results"] = len(results)
    metrics["num_images"] = processed

    return metrics, results

# ---------- 主函数 ----------

def main():
    global CLOUD_SERVER_URL

    parser = argparse.ArgumentParser(description="ROI 裁剪比例 + Padding 对比 (ROI-aware 微调后)")
    parser.add_argument("--server", default=CLOUD_SERVER_URL, help="云端服务器地址")
    parser.add_argument("--max-images", type=int, default=0, help="最多处理图片数 (0=全部)")
    args = parser.parse_args()

    CLOUD_SERVER_URL = args.server

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gt_coco, image_map, ann_map = load_gt_annotations(GT_ANNOTATION_PATH)

    all_results = {}

    print(f"\n[实验] ROI-aware 微调后 — 裁剪比例 + Padding 对比测试")
    print(f"  裁剪比例: {[f'{s:.0%}' for s in SCALE_FACTORS]}")
    print(f"  模式: 无Padding / 有Padding (灰底 pad 到 {TARGET_SHORT_SIDE}px)")
    print(f"  图片数: {'全部' if args.max_images == 0 else args.max_images}\n")

    for sf in SCALE_FACTORS:
        for use_pad in [False, True]:
            mode = "with_pad" if use_pad else "no_pad"
            key = f"{sf:.2f}_{mode}"

            print(f"\n{'='*70}")
            print(f"测试: 裁剪 {sf:.0%} | {'有Padding' if use_pad else '无Padding'}")
            print(f"{'='*70}")

            metrics, _ = run_single_experiment(
                gt_coco, image_map, ann_map, sf, use_pad, args.max_images
            )
            all_results[key] = metrics

            print(f"\n  裁剪: {sf:.0%} | {'有Pad' if use_pad else '无Pad'}")
            print(f"  AP50: {metrics['AP50']:.4f} | AP: {metrics['AP']:.4f} | AR: {metrics['AR']:.4f}")
            print(f"  AP small/medium/large: {metrics['AP_small']:.4f} / {metrics['AP_medium']:.4f} / {metrics['AP_large']:.4f}")
            print(f"  上传: {metrics['avg_upload_kb']:.1f}KB vs 全图 {metrics['avg_full_kb']:.1f}KB | 节省: {metrics['bandwidth_save']:.1%}")
            print(f"  推理: {metrics['avg_inference_time']:.3f}s/图")

    # ===== 汇总表 =====
    print(f"\n{'='*70}")
    print(f"汇总 — ROI-aware 微调后: 裁剪比例 × Padding 对比")
    print(f"{'='*70}")

    # 表1: 无 Padding vs 有 Padding
    print(f"\n  {'裁剪':>6} | {'无Pad AP50':>10} {'无Pad AP':>9} {'无Pad AR':>8} | "
          f"{'有Pad AP50':>10} {'有Pad AP':>9} {'有Pad AR':>8} | {'Pad增益':>8}")
    print(f"  {'-'*80}")
    for sf in SCALE_FACTORS:
        no_pad = all_results[f"{sf:.2f}_no_pad"]
        with_pad = all_results[f"{sf:.2f}_with_pad"]
        gain = with_pad["AP50"] - no_pad["AP50"]
        print(f"  {sf:>5.0%} | {no_pad['AP50']:>10.4f} {no_pad['AP']:>9.4f} {no_pad['AR']:>8.4f} | "
              f"{with_pad['AP50']:>10.4f} {with_pad['AP']:>9.4f} {with_pad['AR']:>8.4f} | {gain:>+8.4f}")

    # 表2: 按目标尺寸分解 (无Pad)
    print(f"\n  按目标尺寸分解 (无Padding):")
    print(f"  {'裁剪':>6} | {'AP_small':>9} {'AP_medium':>10} {'AP_large':>9} | "
          f"{'AR_small':>9} {'AR_medium':>10} {'AR_large':>9}")
    print(f"  {'-'*70}")
    for sf in SCALE_FACTORS:
        m = all_results[f"{sf:.2f}_no_pad"]
        print(f"  {sf:>5.0%} | {m['AP_small']:>9.4f} {m['AP_medium']:>10.4f} {m['AP_large']:>9.4f} | "
              f"{m['AR_small']:>9.4f} {m['AR_medium']:>10.4f} {m['AR_large']:>9.4f}")

    # 表3: 按目标尺寸分解 (有Pad)
    print(f"\n  按目标尺寸分解 (有Padding):")
    print(f"  {'裁剪':>6} | {'AP_small':>9} {'AP_medium':>10} {'AP_large':>9} | "
          f"{'AR_small':>9} {'AR_medium':>10} {'AR_large':>9}")
    print(f"  {'-'*70}")
    for sf in SCALE_FACTORS:
        m = all_results[f"{sf:.2f}_with_pad"]
        print(f"  {sf:>5.0%} | {m['AP_small']:>9.4f} {m['AP_medium']:>10.4f} {m['AP_large']:>9.4f} | "
              f"{m['AR_small']:>9.4f} {m['AR_medium']:>10.4f} {m['AR_large']:>9.4f}")

    # 表4: 带宽对比
    print(f"\n  带宽对比:")
    print(f"  {'裁剪':>6} | {'无Pad上传KB':>12} {'无Pad节省':>9} | {'有Pad上传KB':>12} {'有Pad节省':>9} | {'推理时间':>8}")
    print(f"  {'-'*75}")
    for sf in SCALE_FACTORS:
        no_pad = all_results[f"{sf:.2f}_no_pad"]
        with_pad = all_results[f"{sf:.2f}_with_pad"]
        print(f"  {sf:>5.0%} | {no_pad['avg_upload_kb']:>12.1f} {no_pad['bandwidth_save']:>8.1%} | "
              f"{with_pad['avg_upload_kb']:>12.1f} {with_pad['bandwidth_save']:>8.1%} | "
              f"{no_pad['avg_inference_time']:>7.3f}s")

    # 表5: 对比微调前 (50图小样本)
    print(f"\n  对比微调前 (50图小样本 AP50):")
    print(f"  {'裁剪':>6} | {'微调前无Pad':>10} {'微调后无Pad':>10} {'提升':>8} | "
          f"{'微调前有Pad':>10} {'微调后有Pad':>10} {'提升':>8}")
    print(f"  {'-'*75}")
    for sf in SCALE_FACTORS:
        no_pad = all_results[f"{sf:.2f}_no_pad"]
        with_pad = all_results[f"{sf:.2f}_with_pad"]
        before_no = BEFORE_FINETUNE_50IMG["no_pad"].get(sf, 0)
        before_pad = BEFORE_FINETUNE_50IMG["with_pad"].get(sf, 0)
        diff_no = no_pad["AP50"] - before_no
        diff_pad = with_pad["AP50"] - before_pad
        print(f"  {sf:>5.0%} | {before_no:>10.4f} {no_pad['AP50']:>10.4f} {diff_no:>+8.4f} | "
              f"{before_pad:>10.4f} {with_pad['AP50']:>10.4f} {diff_pad:>+8.4f}")

    # 保存
    output_path = os.path.join(OUTPUT_DIR, "scale_padding_roi_aware.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[保存] -> {output_path}")

if __name__ == "__main__":
    main()
