#!/usr/bin/env python3
"""
Adaptive ROI 云边协同 Pipeline v2 — ROI-aware 微调后模型 (全量 1547 张)
=====================================================================
v2 增强功能:
  1. 自适应裁剪比例 (35%-50%) + 自适应 Padding (resize > 1.2x 时 pad)
  2. YOLO 低置信度 Fallback 机制 (云端漏检时保留 YOLO 低置信检测)
  3. 自适应云端置信度阈值 (小裁剪+多检测时提高阈值减少 FP)
  4. 类别感知的裁剪策略 (swimmer 多 -> 小裁剪+pad, boat 多 -> 大裁剪)
  5. 双模式融合对比 (with/without fallback)

裁剪比例依据 (ROI-aware 微调后全量数据):
  35% = AP50 0.752 (小目标集中场景最优)
  40% = AP50 0.782 (训练分布内最优平衡, 默认)
  45% ≈ 0.785 (大目标多/检测多)
  50% = AP50 0.794 (分散场景必须大裁剪)

Pipeline:
  1. YOLO 边缘推理全图 (conf=0.01)
  2. 置信度分流: >=0.7 直接输出 (边缘), <0.7 自适应裁剪+可选Padding -> 云端
  3. 云端 DETR 检测 -> 坐标映射回 GT 空间
  4. NMS 融合 (IoU=0.5) -- 两种模式:
     a. No-Fallback:  高置信 + 云端 -> NMS  (v1 策略)
     b. With-Fallback: 高置信 + 低置信 + 云端 -> NMS  (v2 推荐)
  5. COCO 评估 (Subset GT)

用法:
  python adaptive_roi_pipeline.py --server http://localhost:8000
  python adaptive_roi_pipeline.py --server http://localhost:8000 --max-images 50  # 快速测试
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
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt")
CLOUD_SERVER_URL = os.environ.get("CLOUD_SERVER_URL", "http://localhost:8000")
GT_ANNOTATION_PATH = os.environ.get("GT_ANNOTATION_PATH", "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json")
IMAGE_DIR = os.environ.get("IMAGE_DIR", "/app/data/seadronessee/compressed_version/val2017/")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/Salience-DETR/experiments/adaptive_roi_v2_results/")

SPLIT_THRESHOLD = 0.7
YOLO_CONF = 0.01
CLOUD_CONF = 0.05
EVAL_CONF = 0.05
NMS_IOU = 0.5
CHECKPOINT_INTERVAL = 100

MIN_SCALE = 0.35
MAX_SCALE = 0.50
SMALL_BBOX_THRESHOLD = 32

TARGET_SHORT_SIDE = 800
PADDING_RESIZE_THRESHOLD = 1.2

SMALL_CLASSES = {0}
LARGE_CLASSES = {1, 2}

EDGE_BASELINE_AP50 = 0.735
CLOUD_FULL_AP50 = 0.858
CLOUD_FULL_AP50_FT = 0.868
ORIGINAL_C_AP50 = 0.544
FUSED_50PCT_AP50 = 0.854

# ========================================================================
#  自适应裁剪比例策略 (v2 增强)
# ========================================================================

def adaptive_scale_factor(low_conf_dets, actual_w, actual_h):
    """
    根据低置信度检测结果自适应决定裁剪比例、padding 和云端置信度阈值。

    决策因素:
      1. 目标分散度 (spread_ratio): 低置信检测框的包围盒面积 / 原图面积
      2. 小目标占比 (small_ratio): 短边 < 32px 的检测框比例
      3. 检测数量 (n)
      4. 类别分布: swimmer 多 -> 小目标场景; boat 多 -> 大目标场景

    裁剪比例依据 (ROI-aware 微调后全量数据):
      35% = AP50 0.752 (小目标集中场景最优)
      40% = AP50 0.782 (训练分布内最优平衡, 默认)
      45% ≈ 0.785 (大目标多/检测多)
      50% = AP50 0.794 (分散场景必须大裁剪)

    Returns:
        scale_factor: float -- 裁剪面积比例 [0.35, 0.50]
        use_padding: bool   -- 是否启用灰底 padding
        cloud_conf: float   -- 云端置信度阈值
        reason: str         -- 决策原因
    """
    if not low_conf_dets:
        return 0.40, True, CLOUD_CONF, "default_no_dets"

    n = len(low_conf_dets)

    all_x1 = [d["bbox"][0] for d in low_conf_dets]
    all_y1 = [d["bbox"][1] for d in low_conf_dets]
    all_x2 = [d["bbox"][2] for d in low_conf_dets]
    all_y2 = [d["bbox"][3] for d in low_conf_dets]
    spread_w = max(all_x2) - min(all_x1)
    spread_h = max(all_y2) - min(all_y1)
    spread_ratio = (spread_w * spread_h) / (actual_w * actual_h)

    small_count = sum(1 for d in low_conf_dets
                      if min(d["bbox"][2] - d["bbox"][0],
                             d["bbox"][3] - d["bbox"][1]) < SMALL_BBOX_THRESHOLD)
    small_ratio = small_count / n

    class_ids = [d.get("class_id", -1) for d in low_conf_dets if "class_id" in d]
    small_class_count = sum(1 for c in class_ids if c in SMALL_CLASSES)
    large_class_count = sum(1 for c in class_ids if c in LARGE_CLASSES)
    has_class_info = len(class_ids) > 0

    if spread_ratio > 0.6:
        scale = 0.50
        use_padding = False
        cloud_conf = CLOUD_CONF
        reason = "spread_high"
    elif spread_ratio > 0.4:
        if small_ratio > 0.5:
            scale = 0.40
            use_padding = True
            cloud_conf = 0.08
            reason = "spread_mid_small"
        else:
            scale = 0.45
            use_padding = False
            cloud_conf = CLOUD_CONF
            reason = "spread_mid_large"
    elif small_ratio > 0.6:
        scale = 0.35
        use_padding = True
        cloud_conf = 0.10
        reason = "small_concentrated"
    elif n > 10:
        scale = 0.45
        use_padding = False
        cloud_conf = 0.08
        reason = "det_count_high"
    else:
        scale = 0.40
        use_padding = True
        cloud_conf = CLOUD_CONF
        reason = "default"

    if has_class_info and small_class_count > large_class_count * 2:
        if scale > 0.40:
            scale = max(MIN_SCALE, scale - 0.05)
            use_padding = True
            reason += "_class_small"
    elif has_class_info and large_class_count > small_class_count * 2:
        if scale < 0.45:
            scale = min(MAX_SCALE, scale + 0.05)
            use_padding = False
            reason += "_class_large"

    return scale, use_padding, cloud_conf, reason

# ========================================================================
#  Padding 函数
# ========================================================================

def create_padded_canvas(crop, img_array, target_short_side=TARGET_SHORT_SIDE):
    crop_h, crop_w = crop.shape[:2]
    short_side = min(crop_w, crop_h)

    if short_side < target_short_side:
        pad_scale = target_short_side / short_side
        padded_w = int(crop_w * pad_scale)
        padded_h = int(crop_h * pad_scale)
    else:
        padded_w = crop_w
        padded_h = crop_h

    mean_val = img_array.mean(axis=(0, 1)).astype(np.uint8)
    canvas = np.full((padded_h, padded_w, 3), mean_val, dtype=np.uint8)

    pad_left = (padded_w - crop_w) // 2
    pad_top = (padded_h - crop_h) // 2
    canvas[pad_top:pad_top + crop_h, pad_left:pad_left + crop_w] = crop

    return canvas, pad_left, pad_top

# ========================================================================
#  工具函数
# ========================================================================

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

def scale_bbox_to_gt(bbox_xywh, actual_w, actual_h, gt_w, gt_h):
    if actual_w == gt_w and actual_h == gt_h:
        return list(bbox_xywh)
    sx, sy = gt_w / actual_w, gt_h / actual_h
    return [bbox_xywh[0] * sx, bbox_xywh[1] * sy,
            bbox_xywh[2] * sx, bbox_xywh[3] * sy]

def xyxy_to_coco_xywh(x1, y1, x2, y2):
    return [x1, y1, x2 - x1, y2 - y1]

def compute_iou(box1, box2):
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[0] + box1[2], box2[0] + box2[2])
    yb = min(box1[1] + box1[3], box2[1] + box2[3])
    if xb <= xa or yb <= ya:
        return 0.0
    inter = (xb - xa) * (yb - ya)
    a1 = box1[2] * box1[3]
    a2 = box2[2] * box2[3]
    return inter / (a1 + a2 - inter)

def apply_nms(detections, iou_threshold=NMS_IOU):
    if not detections:
        return []
    by_class = defaultdict(list)
    for det in detections:
        by_class[det["category_id"]].append(det)
    result = []
    for cls_id, dets in by_class.items():
        dets.sort(key=lambda x: -x["score"])
        kept = []
        for det in dets:
            is_dup = False
            for k in kept:
                if compute_iou(det["bbox"], k["bbox"]) > iou_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(det)
        result.extend(kept)
    return result

def run_yolo(model, img_path, conf=YOLO_CONF):
    results = model(img_path, conf=conf, verbose=False)
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clses = result.boxes.cls.cpu().numpy().astype(int)
    dets = []
    for i in range(len(xyxy)):
        dets.append({
            "bbox": [float(xyxy[i][0]), float(xyxy[i][1]),
                     float(xyxy[i][2]), float(xyxy[i][3])],
            "score": float(confs[i]),
            "class_id": int(clses[i]),
        })
    return dets

def compute_crop_region(actual_w, actual_h, low_conf_dets, scale_factor):
    all_x1 = [d["bbox"][0] for d in low_conf_dets]
    all_y1 = [d["bbox"][1] for d in low_conf_dets]
    all_x2 = [d["bbox"][2] for d in low_conf_dets]
    all_y2 = [d["bbox"][3] for d in low_conf_dets]
    cx = (min(all_x1) + max(all_x2)) / 2
    cy = (min(all_y1) + max(all_y2)) / 2
    ss = np.sqrt(scale_factor)
    cw, ch = int(actual_w * ss), int(actual_h * ss)
    x1, y1 = int(cx - cw / 2), int(cy - ch / 2)
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    x2, y2 = x1 + cw, y1 + ch
    if x2 > actual_w:
        x2 = actual_w
        x1 = max(0, x2 - cw)
    if y2 > actual_h:
        y2 = actual_h
        y1 = max(0, y2 - ch)
    return x1, y1, x2, y2

def image_to_bytes(img_array):
    buf = BytesIO()
    Image.fromarray(img_array).save(buf, format="JPEG", quality=95)
    return buf.getvalue()

def send_to_cloud(img_array, conf=CLOUD_CONF, retries=3):
    ib = image_to_bytes(img_array)
    for a in range(retries):
        try:
            r = requests.post(f"{CLOUD_SERVER_URL}/detect",
                files={"file": ("image.jpg", ib, "image/jpeg")},
                data={"conf_threshold": str(conf)}, timeout=60)
            if r.status_code == 200:
                d = r.json()
                if "results" in d:
                    return [det for item in d["results"]
                            for det in item.get("detections", [])]
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
    raise ValueError(f"Unknown bbox: {type(b)}")

def evaluate_coco_subset(gt_coco, results, label=""):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if not results:
        return {"AP50": 0, "AP": 0, "AP75": 0, "AR": 0, "AR1": 0, "AR10": 0,
                "AP_small": 0, "AP_medium": 0, "AP_large": 0,
                "AR_small": 0, "AR_medium": 0, "AR_large": 0,
                "stats": [0] * 12, "subset_images": 0, "subset_anns": 0}
    pids = {d["image_id"] for d in results}
    si = [img for img in gt_coco["images"] if img["id"] in pids]
    sa = [ann for ann in gt_coco["annotations"] if ann["image_id"] in pids]
    sgt = {"images": si, "annotations": sa, "categories": gt_coco["categories"]}
    if label:
        print(f"\n  [{label}] Subset GT: {len(si)} 张图片, {len(sa)} 个标注")
    cg = COCO()
    cg.dataset = sgt
    cg.createIndex()
    cd = cg.loadRes(results)
    ce = COCOeval(cg, cd, "bbox")
    ce.evaluate()
    ce.accumulate()
    ce.summarize()
    return {
        "AP50": float(ce.stats[1]), "AP": float(ce.stats[0]),
        "AP75": float(ce.stats[2]), "AR": float(ce.stats[8]),
        "AR1": float(ce.stats[6]), "AR10": float(ce.stats[7]),
        "AP_small": float(ce.stats[3]), "AP_medium": float(ce.stats[4]),
        "AP_large": float(ce.stats[5]),
        "AR_small": float(ce.stats[9]), "AR_medium": float(ce.stats[10]),
        "AR_large": float(ce.stats[11]),
        "stats": [float(s) for s in ce.stats],
        "subset_images": len(si), "subset_anns": len(sa),
    }

# ========================================================================
#  主 Pipeline
# ========================================================================

def run_pipeline(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[加载] YOLO 模型: {YOLO_MODEL_PATH}")
    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL_PATH)

    gt_coco, image_map, ann_map = load_gt_annotations(GT_ANNOTATION_PATH)
    image_ids = sorted(ann_map.keys())
    if args.max_images > 0:
        image_ids = image_ids[:args.max_images]
    print(f"[实验] Adaptive ROI v2 云边协同 Pipeline")
    print(f"  处理 {len(image_ids)} 张图片")
    print(f"  分流阈值: {SPLIT_THRESHOLD} | 自适应裁剪: {MIN_SCALE:.0%}-{MAX_SCALE:.0%} | NMS IoU: {NMS_IOU}")
    print(f"  Padding: 短边 < {TARGET_SHORT_SIDE}px 且 resize > {PADDING_RESIZE_THRESHOLD}x 时启用")
    print(f"  Fallback: 低置信度 YOLO 检测作为 fallback 保留\n")

    all_yolo = []
    edge_only = []
    fused_no_fb = []
    fused_with_fb = []

    processed = 0
    fail_count = 0
    img_with_low_conf = 0
    img_no_detection = 0
    total_high_conf = 0
    total_low_conf = 0
    total_cloud_dets = 0
    total_nms_removed_no_fb = 0
    total_nms_removed_with_fb = 0
    total_yolo_time = 0.0
    total_cloud_time = 0.0
    total_upload_bytes = 0
    total_full_bytes = 0
    total_pad_count = 0
    per_class_high = defaultdict(int)
    per_class_low = defaultdict(int)
    per_class_cloud = defaultdict(int)

    scale_distribution = defaultdict(int)
    reason_distribution = defaultdict(int)
    padding_distribution = defaultdict(int)
    scale_per_image = []
    spread_ratios = []
    small_ratios = []
    low_conf_counts = []
    per_scale_upload = defaultdict(list)
    per_scale_cloud_dets = defaultdict(int)
    per_scale_pad = defaultdict(int)
    per_reason_stats = defaultdict(lambda: {
        "count": 0, "upload_kb": [], "cloud_dets": 0, "padded": 0
    })

    for img_id in tqdm(image_ids, desc="Adaptive ROI v2"):
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
        gt_w, gt_h = img_info["width"], img_info["height"]

        # ===== 1. YOLO 边缘推理 =====
        t0 = time.time()
        yolo_dets = run_yolo(model, img_path, conf=YOLO_CONF)
        total_yolo_time += time.time() - t0

        if len(yolo_dets) == 0:
            img_no_detection += 1
            processed += 1
            continue

        yolo_coco = []
        for d in yolo_dets:
            x1, y1, x2, y2 = d["bbox"]
            xywh = xyxy_to_coco_xywh(x1, y1, x2, y2)
            gt_bbox = scale_bbox_to_gt(xywh, actual_w, actual_h, gt_w, gt_h)
            if gt_bbox[2] <= 0 or gt_bbox[3] <= 0:
                continue
            coco_det = {
                "image_id": img_id,
                "category_id": d["class_id"] + 1,
                "bbox": gt_bbox,
                "score": d["score"],
            }
            yolo_coco.append(coco_det)
            if d["score"] >= EVAL_CONF:
                all_yolo.append(coco_det)

        # ===== 2. 置信度分流 =====
        high_conf = [d for d in yolo_coco if d["score"] >= SPLIT_THRESHOLD]
        low_conf = [d for d in yolo_coco
                    if d["score"] < SPLIT_THRESHOLD and d["score"] >= EVAL_CONF]

        for d in high_conf:
            edge_only.append(d)
            per_class_high[d["category_id"]] += 1
        for d in low_conf:
            per_class_low[d["category_id"]] += 1

        total_high_conf += len(high_conf)
        total_low_conf += len(low_conf)

        # ===== 3. 低置信度 -> 自适应裁剪 (+Padding) -> 云端 =====
        cloud_dets_coco = []
        if len(low_conf) > 0:
            img_with_low_conf += 1

            low_conf_actual = []
            for d in low_conf:
                gt_bbox = d["bbox"]
                sx, sy = actual_w / gt_w, actual_h / gt_h
                actual_bbox = [gt_bbox[0] * sx, gt_bbox[1] * sy,
                               gt_bbox[2] * sx, gt_bbox[3] * sy]
                low_conf_actual.append({
                    "bbox": [actual_bbox[0], actual_bbox[1],
                             actual_bbox[0] + actual_bbox[2],
                             actual_bbox[1] + actual_bbox[3]],
                    "class_id": d["category_id"] - 1,
                    "score": d["score"],
                })

            scale_factor, use_padding, cloud_conf, reason = adaptive_scale_factor(
                low_conf_actual, actual_w, actual_h)

            sf_key = f"{scale_factor:.2f}"
            scale_distribution[sf_key] += 1
            reason_distribution[reason] += 1
            padding_distribution[use_padding] += 1
            scale_per_image.append(scale_factor)
            low_conf_counts.append(len(low_conf_actual))

            all_x1 = [d["bbox"][0] for d in low_conf_actual]
            all_y1 = [d["bbox"][1] for d in low_conf_actual]
            all_x2 = [d["bbox"][2] for d in low_conf_actual]
            all_y2 = [d["bbox"][3] for d in low_conf_actual]
            sr = ((max(all_x2) - min(all_x1)) * (max(all_y2) - min(all_y1))
                  / (actual_w * actual_h))
            spread_ratios.append(sr)

            sc = sum(1 for d in low_conf_actual
                     if min(d["bbox"][2] - d["bbox"][0],
                            d["bbox"][3] - d["bbox"][1]) < SMALL_BBOX_THRESHOLD)
            small_ratios.append(sc / len(low_conf_actual))

            crop_region = compute_crop_region(
                actual_w, actual_h, low_conf_actual, scale_factor)
            cx1, cy1, cx2, cy2 = crop_region
            crop = img_array[cy1:cy2, cx1:cx2].copy()

            crop_h, crop_w = crop.shape[:2]
            short_side = min(crop_h, crop_w)
            actual_use_pad = use_padding and short_side < TARGET_SHORT_SIDE

            if actual_use_pad:
                total_pad_count += 1
                per_scale_pad[sf_key] += 1
                cloud_input, pad_left, pad_top = create_padded_canvas(
                    crop, img_array, TARGET_SHORT_SIDE)
            else:
                cloud_input = crop
                pad_left, pad_top = 0, 0

            upload_bytes = len(image_to_bytes(cloud_input))
            total_upload_bytes += upload_bytes
            total_full_bytes += len(image_to_bytes(img_array))
            per_scale_upload[sf_key].append(upload_bytes / 1024)
            per_reason_stats[reason]["count"] += 1
            per_reason_stats[reason]["upload_kb"].append(upload_bytes / 1024)
            per_reason_stats[reason]["padded"] += (1 if actual_use_pad else 0)

            t0 = time.time()
            cloud_dets = send_to_cloud(cloud_input, conf=cloud_conf)
            total_cloud_time += time.time() - t0

            for det in cloud_dets:
                bx1, by1, bx2, by2 = parse_bbox(det["bbox"])
                score = det.get("score", det.get("confidence", 0.0))
                cls_id = det.get("class_id", det.get("category_id", 0))
                if score < cloud_conf:
                    continue

                crop_x1 = bx1 - pad_left
                crop_y1 = by1 - pad_top
                crop_x2 = bx2 - pad_left
                crop_y2 = by2 - pad_top

                actual_x1 = cx1 + crop_x1
                actual_y1 = cy1 + crop_y1
                actual_x2 = cx1 + crop_x2
                actual_y2 = cy1 + crop_y2

                xywh = xyxy_to_coco_xywh(
                    actual_x1, actual_y1, actual_x2, actual_y2)
                gt_bbox = scale_bbox_to_gt(
                    xywh, actual_w, actual_h, gt_w, gt_h)
                if gt_bbox[2] <= 0 or gt_bbox[3] <= 0:
                    continue
                coco_det = {
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": gt_bbox,
                    "score": float(score),
                }
                cloud_dets_coco.append(coco_det)
                per_class_cloud[cls_id] += 1

            total_cloud_dets += len(cloud_dets_coco)
            per_scale_cloud_dets[sf_key] += len(cloud_dets_coco)
            per_reason_stats[reason]["cloud_dets"] += len(cloud_dets_coco)

        # ===== 4. NMS 融合 (两种模式) =====

        combined_no_fb = list(high_conf) + list(cloud_dets_coco)
        pre_nms_no_fb = len(combined_no_fb)
        fused_no_fb_dets = apply_nms(combined_no_fb, NMS_IOU)
        fused_no_fb_dets = [d for d in fused_no_fb_dets if d["score"] >= EVAL_CONF]
        total_nms_removed_no_fb += pre_nms_no_fb - len(fused_no_fb_dets)
        fused_no_fb.extend(fused_no_fb_dets)

        combined_with_fb = list(high_conf) + list(low_conf) + list(cloud_dets_coco)
        pre_nms_with_fb = len(combined_with_fb)
        fused_with_fb_dets = apply_nms(combined_with_fb, NMS_IOU)
        fused_with_fb_dets = [d for d in fused_with_fb_dets if d["score"] >= EVAL_CONF]
        total_nms_removed_with_fb += pre_nms_with_fb - len(fused_with_fb_dets)
        fused_with_fb.extend(fused_with_fb_dets)

        processed += 1
        if processed % CHECKPOINT_INTERVAL == 0:
            with open(os.path.join(OUTPUT_DIR, "checkpoint.json"), "w") as f:
                json.dump({
                    "processed": processed,
                    "all_yolo": all_yolo, "edge_only": edge_only,
                    "fused_no_fb": fused_no_fb, "fused_with_fb": fused_with_fb,
                }, f)

    # ===== 5. 评估 =====
    print("\n" + "=" * 70)
    print(f"Adaptive ROI v2 云边协同 Pipeline -- {processed} 张图片")
    print(f"分流: {SPLIT_THRESHOLD} | 裁剪: {MIN_SCALE:.0%}-{MAX_SCALE:.0%} | NMS IoU: {NMS_IOU}")
    print(f"Padding: {total_pad_count}/{img_with_low_conf} 张启用 | Fallback: 已启用")
    print("=" * 70)

    print("\n--- (1) All YOLO: 全部 YOLO 检测 ---")
    m_yolo = evaluate_coco_subset(gt_coco, all_yolo, "All YOLO")

    print("\n--- (2) Edge-only: 仅高置信度 YOLO ---")
    m_edge = evaluate_coco_subset(gt_coco, edge_only, "Edge-only")

    print("\n--- (3) Fused-NoFB: 高置信 + 自适应云端 (无 Fallback) ---")
    m_no_fb = evaluate_coco_subset(gt_coco, fused_no_fb, "No-Fallback")

    print("\n--- (4) Fused-FB: 高置信 + 低置信 + 自适应云端 (有 Fallback) ---")
    m_fb = evaluate_coco_subset(gt_coco, fused_with_fb, "With-Fallback")

    # ===== 6. 对比输出 =====
    print(f"\n{'=' * 70}")
    print("五组对比")
    print(f"{'=' * 70}")
    hdr = (f"  {'指标':<20} {'AllYOLO':>9} {'Edge':>9} {'NoFB':>9} {'WithFB':>9} {'固定50%':>9}")
    print(hdr)
    print(f"  {'-' * 55}")
    for k, label in [("AP50", "AP50"), ("AP", "AP"), ("AP75", "AP75"),
                      ("AR", "AR"), ("AR1", "AR1"), ("AR10", "AR10")]:
        fixed_val = f"{FUSED_50PCT_AP50:.4f}" if k == "AP50" else "--"
        print(f"  {label:<20} {m_yolo[k]:>9.4f} {m_edge[k]:>9.4f} "
              f"{m_no_fb[k]:>9.4f} {m_fb[k]:>9.4f} {fixed_val:>9}")

    print(f"\n  {'指标':<20} {'AllYOLO':>9} {'Edge':>9} {'NoFB':>9} {'WithFB':>9}")
    print(f"  {'-' * 47}")
    for k, label in [("AP_small", "AP small"), ("AP_medium", "AP medium"),
                      ("AP_large", "AP large"), ("AR_small", "AR small"),
                      ("AR_medium", "AR medium"), ("AR_large", "AR large")]:
        print(f"  {label:<20} {m_yolo[k]:>9.4f} {m_edge[k]:>9.4f} "
              f"{m_no_fb[k]:>9.4f} {m_fb[k]:>9.4f}")

    print(f"\n{'=' * 70}")
    print("与基线对比 (AP50)")
    print(f"{'=' * 70}")
    print(f"  A组 纯边缘 YOLO:         {EDGE_BASELINE_AP50:.4f}")
    print(f"  Edge-only (>=0.7):        {m_edge['AP50']:.4f}")
    print(f"  Fused-NoFB (v1策略):      {m_no_fb['AP50']:.4f}")
    print(f"  Fused-FB (v2推荐):        {m_fb['AP50']:.4f}  ★")
    print(f"  Fused-50% (微调后):       {FUSED_50PCT_AP50:.4f}  (固定50%对比)")
    print(f"  B组 纯云端 (微调后):      {CLOUD_FULL_AP50_FT:.4f}")
    print(f"  C组 原始云边协同:         {ORIGINAL_C_AP50:.4f}")

    fb_vs_nofb = m_fb["AP50"] - m_no_fb["AP50"]
    fb_vs_edge = m_fb["AP50"] - m_edge["AP50"]
    fb_vs_a = m_fb["AP50"] - EDGE_BASELINE_AP50
    fb_vs_50 = m_fb["AP50"] - FUSED_50PCT_AP50
    fb_vs_c = m_fb["AP50"] - ORIGINAL_C_AP50
    recovery = m_fb["AP50"] / CLOUD_FULL_AP50_FT

    print(f"\n  Fused-FB vs NoFB:         {'+' if fb_vs_nofb >= 0 else ''}{fb_vs_nofb:.4f} (fallback 增益)")
    print(f"  Fused-FB vs Edge-only:    {'+' if fb_vs_edge >= 0 else ''}{fb_vs_edge:.4f} (云端增益)")
    print(f"  Fused-FB vs A组边缘:      {'+' if fb_vs_a >= 0 else ''}{fb_vs_a:.4f} ({'超过' if fb_vs_a >= 0 else '低于'}边缘端)")
    print(f"  Fused-FB vs 固定50%:      {'+' if fb_vs_50 >= 0 else ''}{fb_vs_50:.4f} ({'优于' if fb_vs_50 >= 0 else '劣于'}固定50%)")
    print(f"  Fused-FB vs C组原始:      {'+' if fb_vs_c >= 0 else ''}{fb_vs_c:.4f} (改善幅度)")
    print(f"  恢复率 (vs 纯云端):       {recovery:.1%}")

    # ===== 7. Adaptive ROI 策略分析 =====
    print(f"\n{'=' * 70}")
    print("Adaptive ROI 策略分析")
    print(f"{'=' * 70}")

    print(f"\n  裁剪比例分布:")
    print(f"  {'比例':>8} {'图片数':>8} {'占比':>8} {'Padding':>8} {'平均上传KB':>12} {'云端检测/图':>12}")
    print(f"  {'-' * 60}")
    for sf in sorted(scale_distribution.keys()):
        count = scale_distribution[sf]
        pct = count / max(img_with_low_conf, 1) * 100
        avg_up = np.mean(per_scale_upload[sf]) if per_scale_upload[sf] else 0
        pad_cnt = per_scale_pad.get(sf, 0)
        avg_dets = per_scale_cloud_dets.get(sf, 0) / max(count, 1)
        print(f"  {float(sf):>7.0%} {count:>8} {pct:>7.1f}% {pad_cnt:>8} {avg_up:>12.1f} {avg_dets:>12.1f}")

    avg_scale = np.mean(scale_per_image) if scale_per_image else 0
    print(f"\n  平均裁剪比例: {avg_scale:.1%} (固定50%对比: 节省 {(1 - avg_scale / 0.50) * 100:.1f}% 裁剪面积)")
    print(f"  Padding 启用: {total_pad_count}/{img_with_low_conf} ({total_pad_count / max(img_with_low_conf, 1) * 100:.1f}%)")

    print(f"\n  决策原因分布:")
    print(f"  {'原因':>26} {'图片数':>8} {'占比':>8} {'平均上传KB':>12} {'Padding':>8} {'云端检测/图':>12}")
    print(f"  {'-' * 70}")
    for reason in sorted(per_reason_stats.keys()):
        stats = per_reason_stats[reason]
        count = stats["count"]
        pct = count / max(img_with_low_conf, 1) * 100
        avg_up = np.mean(stats["upload_kb"]) if stats["upload_kb"] else 0
        avg_dets = stats["cloud_dets"] / max(count, 1)
        print(f"  {reason:>26} {count:>8} {pct:>7.1f}% {avg_up:>12.1f} {stats['padded']:>8} {avg_dets:>12.1f}")

    if spread_ratios:
        print(f"\n  低置信度检测特征统计:")
        print(f"    分散度 (spread):  mean={np.mean(spread_ratios):.3f}  "
              f"median={np.median(spread_ratios):.3f}  "
              f"min={np.min(spread_ratios):.3f}  max={np.max(spread_ratios):.3f}")
        print(f"    小目标占比:       mean={np.mean(small_ratios):.3f}  "
              f"median={np.median(small_ratios):.3f}")
        print(f"    低置信检测数:     mean={np.mean(low_conf_counts):.1f}  "
              f"median={np.median(low_conf_counts):.0f}  "
              f"max={np.max(low_conf_counts):.0f}")

    # ===== 8. Pipeline 统计 =====
    print(f"\n{'=' * 70}")
    print("Pipeline 统计")
    print(f"{'=' * 70}")
    print(f"  处理图片:              {processed}")
    print(f"  失败/跳过:             {fail_count}")
    print(f"  无检测图片:            {img_no_detection} ({img_no_detection / max(processed, 1) * 100:.1f}%)")
    print(f"  有低置信度图片:        {img_with_low_conf} ({img_with_low_conf / max(processed, 1) * 100:.1f}%)")
    print(f"  高置信度检测:          {total_high_conf}")
    print(f"  低置信度检测:          {total_low_conf} (上传云端)")
    print(f"  云端返回检测:          {total_cloud_dets}")
    print(f"  NMS 去重 (无Fallback): {total_nms_removed_no_fb}")
    print(f"  NMS 去重 (有Fallback): {total_nms_removed_with_fb}")
    print(f"  融合后检测 (无FB):     {len(fused_no_fb)}")
    print(f"  融合后检测 (有FB):     {len(fused_with_fb)}")
    cloud_ratio = total_low_conf / max(total_high_conf + total_low_conf, 1) * 100
    print(f"  云端分流比:            {cloud_ratio:.1f}%")

    cat_names = {1: "swimmer", 2: "boat", 3: "jetski", 4: "life_saving", 5: "buoy"}
    print(f"\n  按类别统计:")
    print(f"  {'类别':<15} {'高置信':>8} {'低置信':>8} {'云端':>8}")
    print(f"  {'-' * 40}")
    for cid in sorted(per_class_high.keys() | per_class_low.keys() | per_class_cloud.keys()):
        name = cat_names.get(cid, f"cls{cid}")
        print(f"  {name:<15} {per_class_high.get(cid, 0):>8} "
              f"{per_class_low.get(cid, 0):>8} {per_class_cloud.get(cid, 0):>8}")

    print(f"\n  带宽 (仅低置信度图片):")
    if img_with_low_conf > 0:
        avg_upload = total_upload_bytes / img_with_low_conf / 1024
        avg_full = total_full_bytes / img_with_low_conf / 1024
        bw_ratio = total_upload_bytes / max(total_full_bytes, 1)
        fixed_50_upload = 863.5
        print(f"    Adaptive v2 平均上传:  {avg_upload:.1f} KB")
        print(f"    固定50% 平均上传:     {fixed_50_upload:.1f} KB")
        print(f"    全图平均大小:          {avg_full:.1f} KB")
        print(f"    带宽比例 (vs全图):     {bw_ratio:.1%}")
        print(f"    带宽节省 (vs全图):     {(1 - bw_ratio):.1%}")
        bw_save_vs_50 = (fixed_50_upload - avg_upload) / fixed_50_upload * 100
        print(f"    带宽节省 (vs 50%):     {bw_save_vs_50:.1f}% ({fixed_50_upload - avg_upload:.1f} KB/图)")

    print(f"\n  延迟:")
    print(f"    YOLO 总时间:  {total_yolo_time:.1f}s ({total_yolo_time / max(processed, 1) * 1000:.1f}ms/图)")
    print(f"    云端总时间:  {total_cloud_time:.1f}s ({total_cloud_time / max(img_with_low_conf, 1) * 1000:.1f}ms/需云端图)")

    # ===== 9. 保存 =====
    output_path = os.path.join(OUTPUT_DIR, "adaptive_roi_v2_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "all_yolo": {k: v for k, v in m_yolo.items() if k != "stats"},
            "edge_only": {k: v for k, v in m_edge.items() if k != "stats"},
            "fused_no_fallback": {k: v for k, v in m_no_fb.items() if k != "stats"},
            "fused_with_fallback": {k: v for k, v in m_fb.items() if k != "stats"},
            "config": {
                "split_threshold": SPLIT_THRESHOLD,
                "min_scale": MIN_SCALE, "max_scale": MAX_SCALE,
                "nms_iou": NMS_IOU,
                "yolo_conf": YOLO_CONF, "cloud_conf": CLOUD_CONF,
                "eval_conf": EVAL_CONF,
                "target_short_side": TARGET_SHORT_SIDE,
                "padding_resize_threshold": PADDING_RESIZE_THRESHOLD,
                "processed": processed,
            },
            "adaptive_stats": {
                "scale_distribution": dict(scale_distribution),
                "reason_distribution": dict(reason_distribution),
                "padding_count": total_pad_count,
                "padding_ratio": total_pad_count / max(img_with_low_conf, 1),
                "avg_scale": float(avg_scale) if scale_per_image else 0,
                "avg_spread_ratio": float(np.mean(spread_ratios)) if spread_ratios else 0,
                "avg_small_ratio": float(np.mean(small_ratios)) if small_ratios else 0,
                "avg_low_conf_count": float(np.mean(low_conf_counts)) if low_conf_counts else 0,
            },
            "pipeline_stats": {
                "img_with_low_conf": img_with_low_conf,
                "img_no_detection": img_no_detection,
                "total_high_conf": total_high_conf,
                "total_low_conf": total_low_conf,
                "total_cloud_dets": total_cloud_dets,
                "total_nms_removed_no_fb": total_nms_removed_no_fb,
                "total_nms_removed_with_fb": total_nms_removed_with_fb,
                "total_fused_no_fb": len(fused_no_fb),
                "total_fused_with_fb": len(fused_with_fb),
                "cloud_ratio": cloud_ratio,
                "yolo_time_s": total_yolo_time,
                "cloud_time_s": total_cloud_time,
                "total_pad_count": total_pad_count,
            },
            "baselines": {
                "edge": EDGE_BASELINE_AP50,
                "cloud_full": CLOUD_FULL_AP50,
                "cloud_full_finetuned": CLOUD_FULL_AP50_FT,
                "original_c": ORIGINAL_C_AP50,
                "fused_50pct": FUSED_50PCT_AP50,
            },
            "comparisons": {
                "fb_vs_nofb": fb_vs_nofb,
                "fb_vs_edge": fb_vs_edge,
                "fb_vs_a": fb_vs_a,
                "fb_vs_50pct": fb_vs_50,
                "fb_vs_c": fb_vs_c,
                "recovery_rate": recovery,
            },
        }, f, indent=2, default=str)
    print(f"\n[保存] 结果 -> {output_path}")

    for name, res in [("all_yolo", all_yolo), ("edge_only", edge_only),
                      ("fused_no_fb", fused_no_fb),
                      ("fused_with_fb", fused_with_fb)]:
        p = os.path.join(OUTPUT_DIR, f"predictions_{name}.json")
        with open(p, "w") as f:
            json.dump(res, f)
    print(f"[保存] 预测结果 -> {OUTPUT_DIR}/predictions_*.json")

    ckpt = os.path.join(OUTPUT_DIR, "checkpoint.json")
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print("[清理] checkpoint 已删除")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive ROI v2 云边协同 Pipeline")
    parser.add_argument("--max-images", type=int, default=0,
                        help="最多处理图片数 (0=全部)")
    parser.add_argument("--server", type=str, default=CLOUD_SERVER_URL,
                        help="云端服务器地址")
    parser.add_argument("--split", type=float, default=SPLIT_THRESHOLD,
                        help="置信度分流阈值")
    args = parser.parse_args()

    CLOUD_SERVER_URL = args.server
    SPLIT_THRESHOLD = args.split

    run_pipeline(args)
