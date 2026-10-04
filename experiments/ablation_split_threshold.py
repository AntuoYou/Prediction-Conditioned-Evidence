#!/usr/bin/env python3
"""
A7 消融实验：分流阈值灵敏度 (Split Threshold Ablation)
=====================================================
测试不同置信度分流阈值 (0.5, 0.6, 0.7, 0.8) 对云边协同性能的影响。

优化策略: YOLO 推理只跑一次，缓存全部检测结果，
         然后对每个分流阈值分别做云端推理 + NMS 融合 + COCO 评估。

消融逻辑:
  - 分流阈值决定多少检测走云端 (阈值越低 -> 高置信越多, 云端负载越轻)
  - 阈值影响: 云端分流比、带宽、延迟、融合精度
  - 已有 0.7 的数据 (Adaptive ROI v2 Fused-FB AP50=0.8466)

用法:
  python ablation_split_threshold.py --server http://localhost:8000
  python ablation_split_threshold.py --server http://localhost:8000 --max-images 50  # 快速测试

依赖:
  pip install requests pycocotools pillow numpy tqdm ultralytics --break-system-packages
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
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/Salience-DETR/experiments/ablation_split_threshold/")

# 消融实验: 分流阈值列表
SPLIT_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]

YOLO_CONF = 0.01          # YOLO 最小输出置信度
CLOUD_CONF = 0.05         # 云端默认置信度阈值
EVAL_CONF = 0.05          # 评估时最小置信度
NMS_IOU = 0.5             # NMS IoU 阈值

# 自适应裁剪比例范围
MIN_SCALE = 0.35
MAX_SCALE = 0.50
SMALL_BBOX_THRESHOLD = 32

# Padding 配置
TARGET_SHORT_SIDE = 800
PADDING_RESIZE_THRESHOLD = 1.2

# 类别映射
SMALL_CLASSES = {0}        # swimmer
LARGE_CLASSES = {1, 2}     # boat, jetski

# 基线
EDGE_BASELINE_AP50 = 0.735
CLOUD_FULL_AP50_FT = 0.868
FUSED_50PCT_AP50 = 0.854

# ========================================================================

#  自适应裁剪比例策略 (与 adaptive_roi_pipeline.py 一致)

# ========================================================================

def adaptive_scale_factor(low_conf_dets, actual_w, actual_h):
    """根据低置信度检测结果自适应决定裁剪比例、padding 和云端置信度阈值。"""
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

#  工具函数 (与 adaptive_roi_pipeline.py 一致)

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

#  阶段 1: YOLO 推理 (只跑一次，缓存结果)

# ========================================================================

def run_yolo_cache(model, gt_coco, image_map, ann_map, max_images=0):
    """
    对所有图片跑一次 YOLO 推理，缓存检测结果。
    返回: yolo_cache = {img_id: {"dets": [...], "actual_w": w, "actual_h": h, "gt_w": w, "gt_h": h}}
    """
    image_ids = sorted(ann_map.keys())
    if max_images > 0:
        image_ids = image_ids[:max_images]

    yolo_cache = {}
    total_yolo_time = 0.0

    for img_id in tqdm(image_ids, desc="YOLO 推理 (缓存)"):
        img_info = image_map[img_id]
        img_path = os.path.join(IMAGE_DIR, img_info["file_name"])
        if not os.path.exists(img_path):
            continue
        try:
            img_array = load_image(img_path)
        except Exception:
            continue

        actual_h, actual_w = img_array.shape[:2]
        gt_w, gt_h = img_info["width"], img_info["height"]

        t0 = time.time()
        yolo_dets = run_yolo(model, img_path, conf=YOLO_CONF)
        total_yolo_time += time.time() - t0

        # 转换为 GT 空间 COCO 格式
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

        yolo_cache[img_id] = {
            "dets": yolo_coco,
            "actual_w": actual_w,
            "actual_h": actual_h,
            "gt_w": gt_w,
            "gt_h": gt_h,
            "img_array": img_array,  # 保留图片数组供云端裁剪使用
        }

    print(f"\n  YOLO 缓存完成: {len(yolo_cache)} 张图片, "
          f"总时间 {total_yolo_time:.1f}s "
          f"({total_yolo_time / max(len(yolo_cache), 1) * 1000:.1f} ms/图)")
    return yolo_cache, total_yolo_time

# ========================================================================

#  阶段 2: 对每个分流阈值做云端推理 + 融合 + 评估

# ========================================================================

def run_single_threshold(yolo_cache, gt_coco, split_threshold, yolo_time_total):
    """
    对单个分流阈值运行云端推理 + NMS 融合 + COCO 评估。
    使用 With-Fallback 模式 (v2 推荐)。
    """
    print(f"\n{'=' * 70}")
    print(f"分流阈值 = {split_threshold}")
    print(f"{'=' * 70}")

    # 结果收集
    all_yolo = []
    edge_only = []
    fused_with_fb = []

    # 统计
    processed = 0
    img_with_low_conf = 0
    total_high_conf = 0
    total_low_conf = 0
    total_cloud_dets = 0
    total_nms_removed = 0
    total_cloud_time = 0.0
    total_upload_bytes = 0
    total_full_bytes = 0
    total_pad_count = 0
    per_class_high = defaultdict(int)
    per_class_low = defaultdict(int)
    per_class_cloud = defaultdict(int)

    # Adaptive ROI 统计
    scale_distribution = defaultdict(int)
    reason_distribution = defaultdict(int)

    for img_id in tqdm(sorted(yolo_cache.keys()), desc=f"split={split_threshold}"):
        cached = yolo_cache[img_id]
        yolo_coco = cached["dets"]
        actual_w = cached["actual_w"]
        actual_h = cached["actual_h"]
        gt_w = cached["gt_w"]
        gt_h = cached["gt_h"]
        img_array = cached["img_array"]

        if len(yolo_coco) == 0:
            processed += 1
            continue

        # 全部 YOLO 检测 (用于 All YOLO 评估)
        for d in yolo_coco:
            if d["score"] >= EVAL_CONF:
                all_yolo.append(d)

        # ===== 置信度分流 =====
        high_conf = [d for d in yolo_coco if d["score"] >= split_threshold]
        low_conf = [d for d in yolo_coco
                    if d["score"] < split_threshold and d["score"] >= EVAL_CONF]

        for d in high_conf:
            edge_only.append(d)
            per_class_high[d["category_id"]] += 1
        for d in low_conf:
            per_class_low[d["category_id"]] += 1

        total_high_conf += len(high_conf)
        total_low_conf += len(low_conf)

        # ===== 低置信度 -> 自适应裁剪 -> 云端 =====
        cloud_dets_coco = []
        if len(low_conf) > 0:
            img_with_low_conf += 1

            # 转回实际图片空间坐标
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

            # 自适应裁剪比例决策
            scale_factor, use_padding, cloud_conf, reason = adaptive_scale_factor(
                low_conf_actual, actual_w, actual_h)

            sf_key = f"{scale_factor:.2f}"
            scale_distribution[sf_key] += 1
            reason_distribution[reason] += 1

            # 裁剪区域
            crop_region = compute_crop_region(
                actual_w, actual_h, low_conf_actual, scale_factor)
            cx1, cy1, cx2, cy2 = crop_region
            crop = img_array[cy1:cy2, cx1:cx2].copy()

            # Padding 决策
            crop_h, crop_w = crop.shape[:2]
            short_side = min(crop_h, crop_w)
            actual_use_pad = use_padding and short_side < TARGET_SHORT_SIDE

            if actual_use_pad:
                total_pad_count += 1
                cloud_input, pad_left, pad_top = create_padded_canvas(
                    crop, img_array, TARGET_SHORT_SIDE)
            else:
                cloud_input = crop
                pad_left, pad_top = 0, 0

            # 带宽统计
            upload_bytes = len(image_to_bytes(cloud_input))
            total_upload_bytes += upload_bytes
            total_full_bytes += len(image_to_bytes(img_array))

            # 云端推理
            t0 = time.time()
            cloud_dets = send_to_cloud(cloud_input, conf=cloud_conf)
            total_cloud_time += time.time() - t0

            # 坐标映射
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

        # ===== NMS 融合 (With-Fallback: 高置信 + 低置信 + 云端) =====
        combined = list(high_conf) + list(low_conf) + list(cloud_dets_coco)
        pre_nms = len(combined)
        fused_dets = apply_nms(combined, NMS_IOU)
        fused_dets = [d for d in fused_dets if d["score"] >= EVAL_CONF]
        total_nms_removed += pre_nms - len(fused_dets)
        fused_with_fb.extend(fused_dets)

        processed += 1

    # ===== 评估 =====
    print(f"\n--- split={split_threshold} 评估 ---")
    m_yolo = evaluate_coco_subset(gt_coco, all_yolo, f"All YOLO (split={split_threshold})")
    m_edge = evaluate_coco_subset(gt_coco, edge_only, f"Edge-only (split={split_threshold})")
    m_fused = evaluate_coco_subset(gt_coco, fused_with_fb, f"Fused-FB (split={split_threshold})")

    # 带宽统计
    avg_upload = total_upload_bytes / max(img_with_low_conf, 1) / 1024
    avg_full = total_full_bytes / max(img_with_low_conf, 1) / 1024
    cloud_ratio = total_low_conf / max(total_high_conf + total_low_conf, 1) * 100

    # 打印摘要
    print(f"\n  split={split_threshold} 摘要:")
    print(f"    AP50:  All YOLO={m_yolo['AP50']:.4f}  Edge={m_edge['AP50']:.4f}  Fused-FB={m_fused['AP50']:.4f}")
    print(f"    高置信={total_high_conf}  低置信={total_low_conf}  云端返回={total_cloud_dets}")
    print(f"    云端分流比={cloud_ratio:.1f}%  带宽={avg_upload:.1f}KB  云端延迟={total_cloud_time/max(img_with_low_conf,1)*1000:.1f}ms/图")
    print(f"    裁剪比例分布: {dict(scale_distribution)}")

    return {
        "split_threshold": split_threshold,
        "metrics": {
            "all_yolo": {k: v for k, v in m_yolo.items() if k != "stats"},
            "edge_only": {k: v for k, v in m_edge.items() if k != "stats"},
            "fused_fb": {k: v for k, v in m_fused.items() if k != "stats"},
        },
        "stats": {
            "processed": processed,
            "img_with_low_conf": img_with_low_conf,
            "total_high_conf": total_high_conf,
            "total_low_conf": total_low_conf,
            "total_cloud_dets": total_cloud_dets,
            "total_nms_removed": total_nms_removed,
            "total_fused": len(fused_with_fb),
            "cloud_ratio": cloud_ratio,
            "avg_upload_kb": avg_upload,
            "avg_full_kb": avg_full,
            "bw_save_pct": (1 - total_upload_bytes / max(total_full_bytes, 1)) * 100,
            "yolo_time_s": yolo_time_total,
            "cloud_time_s": total_cloud_time,
            "cloud_time_per_img_ms": total_cloud_time / max(img_with_low_conf, 1) * 1000,
            "total_pad_count": total_pad_count,
            "scale_distribution": dict(scale_distribution),
            "reason_distribution": dict(reason_distribution),
            "per_class": {
                "high": dict(per_class_high),
                "low": dict(per_class_low),
                "cloud": dict(per_class_cloud),
            },
        },
    }

# ========================================================================

#  主函数

# ========================================================================

def main():
    global CLOUD_SERVER_URL

    parser = argparse.ArgumentParser(description="A7 消融实验: 分流阈值灵敏度")
    parser.add_argument("--server", default=CLOUD_SERVER_URL, help="云端服务器地址")
    parser.add_argument("--max-images", type=int, default=0, help="最多处理图片数 (0=全部)")
    parser.add_argument("--thresholds", type=str, default="0.5,0.6,0.7,0.8",
                        help="逗号分隔的分流阈值列表")
    args = parser.parse_args()

    CLOUD_SERVER_URL = args.server
    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载 YOLO
    print(f"[加载] YOLO 模型: {YOLO_MODEL_PATH}")
    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL_PATH)

    # 加载 GT
    gt_coco, image_map, ann_map = load_gt_annotations(GT_ANNOTATION_PATH)

    print(f"\n{'=' * 70}")
    print(f"A7 消融实验: 分流阈值灵敏度")
    print(f"  分流阈值: {thresholds}")
    print(f"  YOLO 推理只跑一次，然后对每个阈值分别做云端推理+融合+评估")
    print(f"{'=' * 70}")

    # ===== 阶段 1: YOLO 推理 (只跑一次) =====
    print(f"\n[阶段 1] YOLO 推理缓存...")
    yolo_cache, yolo_time_total = run_yolo_cache(
        model, gt_coco, image_map, ann_map, args.max_images)

    # ===== 阶段 2: 对每个分流阈值做云端推理 + 融合 + 评估 =====
    all_results = []
    for thr in thresholds:
        print(f"\n[阶段 2] 分流阈值 = {thr}")
        result = run_single_threshold(yolo_cache, gt_coco, thr, yolo_time_total)
        all_results.append(result)

        # 每个阈值的结果单独保存
        thr_dir = os.path.join(OUTPUT_DIR, f"split_{thr}")
        os.makedirs(thr_dir, exist_ok=True)
        with open(os.path.join(thr_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)

    # ===== 汇总对比表 =====
    print(f"\n{'=' * 70}")
    print("A7 消融实验汇总: 分流阈值灵敏度")
    print(f"{'=' * 70}")

    # 主指标对比
    print(f"\n  {'指标':<20} ", end="")
    for r in all_results:
        print(f"{'split='+str(r['split_threshold']):>12}", end="")
    print()
    print(f"  {'-' * (20 + 12 * len(all_results))}")

    for metric_key, metric_label in [
        ("AP50", "Fused AP50"), ("AP", "Fused AP"), ("AP75", "Fused AP75"),
        ("AR", "Fused AR"), ("AR1", "Fused AR1"), ("AR10", "Fused AR10"),
    ]:
        print(f"  {metric_label:<20} ", end="")
        for r in all_results:
            val = r["metrics"]["fused_fb"][metric_key]
            print(f"{val:>12.4f}", end="")
        print()

    print()
    for metric_key, metric_label in [
        ("AP50", "Edge AP50"), ("AP50", "All YOLO AP50"),
    ]:
        src = "edge_only" if "Edge" in metric_label else "all_yolo"
        print(f"  {metric_label:<20} ", end="")
        for r in all_results:
            val = r["metrics"][src][metric_key]
            print(f"{val:>12.4f}", end="")
        print()

    # 按目标尺寸
    print(f"\n  {'Fused 按尺寸':<20} ", end="")
    for r in all_results:
        print(f"{'split='+str(r['split_threshold']):>12}", end="")
    print()
    print(f"  {'-' * (20 + 12 * len(all_results))}")
    for metric_key, metric_label in [
        ("AP_small", "AP small"), ("AP_medium", "AP medium"), ("AP_large", "AP large"),
        ("AR_small", "AR small"), ("AR_medium", "AR medium"), ("AR_large", "AR large"),
    ]:
        print(f"  {metric_label:<20} ", end="")
        for r in all_results:
            val = r["metrics"]["fused_fb"][metric_key]
            print(f"{val:>12.4f}", end="")
        print()

    # Pipeline 统计
    print(f"\n  {'Pipeline 统计':<20} ", end="")
    for r in all_results:
        print(f"{'split='+str(r['split_threshold']):>12}", end="")
    print()
    print(f"  {'-' * (20 + 12 * len(all_results))}")
    for stat_key, stat_label in [
        ("total_high_conf", "高置信检测"), ("total_low_conf", "低置信检测"),
        ("total_cloud_dets", "云端返回"), ("total_fused", "融合后检测"),
        ("cloud_ratio", "云端分流比%"), ("avg_upload_kb", "带宽KB"),
        ("bw_save_pct", "带宽节省%"), ("cloud_time_per_img_ms", "云端延迟ms"),
    ]:
        print(f"  {stat_label:<20} ", end="")
        for r in all_results:
            val = r["stats"][stat_key]
            if "ratio" in stat_key or "save" in stat_key:
                print(f"{val:>12.1f}", end="")
            elif "ms" in stat_key:
                print(f"{val:>12.1f}", end="")
            else:
                print(f"{int(val):>12}", end="")
        print()

    # 与基线对比
    print(f"\n  {'vs 基线':<20} ", end="")
    for r in all_results:
        print(f"{'split='+str(r['split_threshold']):>12}", end="")
    print()
    print(f"  {'-' * (20 + 12 * len(all_results))}")
    print(f"  {'vs Edge baseline':<20} ", end="")
    for r in all_results:
        val = r["metrics"]["fused_fb"]["AP50"] - EDGE_BASELINE_AP50
        print(f"{'+' if val >= 0 else ''}{val:>11.4f}", end="")
    print()
    print(f"  {'vs Cloud (FT)':<20} ", end="")
    for r in all_results:
        val = r["metrics"]["fused_fb"]["AP50"] - CLOUD_FULL_AP50_FT
        print(f"{'+' if val >= 0 else ''}{val:>11.4f}", end="")
    print()
    print(f"  {'恢复率':<20} ", end="")
    for r in all_results:
        val = r["metrics"]["fused_fb"]["AP50"] / CLOUD_FULL_AP50_FT * 100
        print(f"{val:>11.1f}%", end="")
    print()

    # 保存汇总
    summary_path = os.path.join(OUTPUT_DIR, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "experiment": "A7: Split Threshold Ablation",
            "thresholds": thresholds,
            "results": all_results,
            "baselines": {
                "edge": EDGE_BASELINE_AP50,
                "cloud_full_ft": CLOUD_FULL_AP50_FT,
                "fused_50pct": FUSED_50PCT_AP50,
            },
        }, f, indent=2, default=str)
    print(f"\n[保存] 汇总结果 -> {summary_path}")

if __name__ == "__main__":
    main()
