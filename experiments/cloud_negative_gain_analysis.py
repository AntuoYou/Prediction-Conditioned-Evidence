#!/usr/bin/env python3
"""
Cloud Negative Gain Error Analysis (1547 张全量)
=================================================

对云边协同 Pipeline 中云端 DETR 的检测做逐目标 Error Analysis：
  - Correct (TP): 云端检测命中 GT (IoU≥0.5, 类别正确)
  - Miss (FN):   GT 在裁剪区域内但云端未检出
  - FP:          云端检测无 GT 匹配
  - ClassErr:    云端检测位置命中 GT 但类别错误

同时分析"云端增益/负增益"：
  - Gain:  云端检出了 YOLO 低置信或漏检的 GT → 云端帮忙
  - Neutral: 云端检出与 YOLO 高置信重复 → NMS 去重
  - Harm:  云端 FP 进入融合结果 → 云端添乱

输出：
  1. 云端检测质量总表 (TP/FP/Miss/ClassErr)
  2. 按类别分解
  3. 按目标尺寸分解
  4. 云端增益 vs 负增益
  5. YOLO 低置信检测的正确性分析
  6. 逐图统计

用法：
  python cloud_negative_gain_analysis.py --server http://localhost:8000
  python cloud_negative_gain_analysis.py --server http://localhost:8000 --max-images 50  # 快速测试
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
YOLO_MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH",
    "/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt",
)
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
    "/app/Salience-DETR/experiments/cloud_negative_gain_analysis/",
)

SPLIT_THRESHOLD = 0.7
YOLO_CONF = 0.01
CLOUD_CONF = 0.05
EVAL_CONF = 0.05
SCALE_FACTOR = 0.50
NMS_IOU = 0.5
MATCH_IOU = 0.5  # TP 匹配阈值
CHECKPOINT_INTERVAL = 100

# COCO 面积阈值 (与 pycocotools 一致)
AREA_SMALL = 32 ** 2      # 1024 px² (在 3840×2160 空间)
AREA_LARGE = 96 ** 2      # 9216 px²

CAT_NAMES = {
    1: "swimmer",
    2: "boat",
    3: "jetski",
    4: "life_saving",
    5: "buoy",
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

def scale_bbox_to_gt(bbox_xywh, actual_w, actual_h, gt_w, gt_h):
    if actual_w == gt_w and actual_h == gt_h:
        return list(bbox_xywh)
    sx, sy = gt_w / actual_w, gt_h / actual_h
    return [bbox_xywh[0] * sx, bbox_xywh[1] * sy, bbox_xywh[2] * sx, bbox_xywh[3] * sy]

def gt_bbox_to_actual(bbox_xywh, gt_w, gt_h, actual_w, actual_h):
    if gt_w == actual_w and gt_h == actual_h:
        return list(bbox_xywh)
    sx, sy = actual_w / gt_w, actual_h / gt_h
    return [bbox_xywh[0] * sx, bbox_xywh[1] * sy, bbox_xywh[2] * sx, bbox_xywh[3] * sy]

def xyxy_to_coco_xywh(x1, y1, x2, y2):
    return [x1, y1, x2 - x1, y2 - y1]

def coco_to_xyxy(bbox_xywh):
    return [bbox_xywh[0], bbox_xywh[1],
            bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]]

def compute_iou_xyxy(box1, box2):
    """两个 [x1,y1,x2,y2] 格式 bbox 的 IoU"""
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    if xb <= xa or yb <= ya:
        return 0.0
    inter = (xb - xa) * (yb - ya)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter)

def get_size_category(area):
    if area < AREA_SMALL:
        return "small"
    elif area < AREA_LARGE:
        return "medium"
    else:
        return "large"

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

def compute_crop_region(actual_w, actual_h, low_conf_dets, scale_factor=SCALE_FACTOR):
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
    raise ValueError(f"Unknown bbox: {type(b)}")

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
                if compute_iou_xyxy(coco_to_xyxy(det["bbox"]), coco_to_xyxy(k["bbox"])) > iou_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(det)
        result.extend(kept)
    return result

# ---------- 匹配与统计 ----------

def match_detections_to_gt(detections, gt_anns, match_iou=MATCH_IOU):
    """
    将检测结果与 GT 匹配，返回每个检测的匹配状态。
    """
    gt_xyxy = {}
    for ann in gt_anns:
        gt_xyxy[ann["id"]] = ann["bbox_xyxy"] if "bbox_xyxy" in ann else coco_to_xyxy(ann["bbox"])

    matched_gt_ids = set()
    results = []

    sorted_dets = sorted(detections, key=lambda x: -x["score"])

    for det in sorted_dets:
        det_xyxy = det["bbox"] if len(det["bbox"]) == 4 and det["bbox"][2] > det["bbox"][0] else coco_to_xyxy(det["bbox"])
        det_cls = det["class_id"]

        best_iou = 0
        best_gt_id = None
        best_gt_cls = None

        for ann in gt_anns:
            if ann["id"] in matched_gt_ids:
                continue
            iou = compute_iou_xyxy(det_xyxy, gt_xyxy[ann["id"]])
            if iou > best_iou:
                best_iou = iou
                best_gt_id = ann["id"]
                best_gt_cls = ann["category_id"]

        if best_iou >= match_iou and best_gt_id is not None:
            if best_gt_cls == det_cls:
                match_type = "correct"
            else:
                match_type = "class_err"
            matched_gt_ids.add(best_gt_id)
        else:
            match_type = "fp"

        results.append({
            **det,
            "match_type": match_type,
            "matched_gt_id": best_gt_id if match_type in ("correct", "class_err") else None,
            "iou": best_iou,
        })

    return results, matched_gt_ids

def analyze_image(img_id, img_info, img_array, gt_anns,
                  yolo_dets, cloud_dets_coco, crop_region,
                  high_conf_yolo, low_conf_yolo):
    """
    对单张图片做完整的 Error Analysis。
    """
    actual_h, actual_w = img_array.shape[:2]
    gt_w, gt_h = img_info["width"], img_info["height"]

    # GT bbox 转为 xyxy (GT 空间)
    gt_list = []
    for ann in gt_anns:
        gt_list.append({
            "id": ann["id"],
            "bbox_xyxy": coco_to_xyxy(ann["bbox"]),
            "bbox_xywh": ann["bbox"],
            "category_id": ann["category_id"],
            "area": ann["area"],
            "size": get_size_category(ann["area"]),
        })

    # ---- 1. 云端检测匹配 GT ----
    cloud_for_match = []
    for d in cloud_dets_coco:
        cloud_for_match.append({
            "bbox": coco_to_xyxy(d["bbox"]),
            "score": d["score"],
            "class_id": d["category_id"],
        })

    cloud_matched, cloud_matched_gt_ids = match_detections_to_gt(
        cloud_for_match, gt_list, MATCH_IOU
    )

    cloud_tp = sum(1 for d in cloud_matched if d["match_type"] == "correct")
    cloud_fp = sum(1 for d in cloud_matched if d["match_type"] == "fp")
    cloud_class_err = sum(1 for d in cloud_matched if d["match_type"] == "class_err")

    # ---- 2. 云端 Miss: GT 在裁剪区域内但未被云端检出 ----
    cx1, cy1, cx2, cy2 = crop_region
    sx, sy = gt_w / actual_w, gt_h / actual_h
    gt_crop_x1, gt_crop_y1 = cx1 * sx, cy1 * sy
    gt_crop_x2, gt_crop_y2 = cx2 * sx, cy2 * sy

    gt_in_crop = []
    gt_in_crop_unmatched = []
    for g in gt_list:
        gx1, gy1, gx2, gy2 = g["bbox_xyxy"]
        gcx = (gx1 + gx2) / 2
        gcy = (gy1 + gy2) / 2
        if gt_crop_x1 <= gcx <= gt_crop_x2 and gt_crop_y1 <= gcy <= gt_crop_y2:
            gt_in_crop.append(g)
            if g["id"] not in cloud_matched_gt_ids:
                gt_in_crop_unmatched.append(g)

    cloud_miss = len(gt_in_crop_unmatched)

    # ---- 3. YOLO 高置信检测匹配 GT ----
    high_for_match = []
    for d in high_conf_yolo:
        high_for_match.append({
            "bbox": coco_to_xyxy(d["bbox"]),
            "score": d["score"],
            "class_id": d["category_id"],
        })

    high_matched, high_matched_gt_ids = match_detections_to_gt(
        high_for_match, gt_list, MATCH_IOU
    )

    # ---- 4. YOLO 低置信检测匹配 GT ----
    low_for_match = []
    for d in low_conf_yolo:
        low_for_match.append({
            "bbox": coco_to_xyxy(d["bbox"]),
            "score": d["score"],
            "class_id": d["category_id"],
        })

    low_matched, low_matched_gt_ids = match_detections_to_gt(
        low_for_match, gt_list, MATCH_IOU
    )

    yolo_low_tp = sum(1 for d in low_matched if d["match_type"] == "correct")
    yolo_low_fp = sum(1 for d in low_matched if d["match_type"] == "fp")

    # ---- 5. 云端增益/负增益分析 ----
    cloud_gain = 0
    cloud_confirm = 0
    cloud_rescue = 0
    cloud_redundant = 0
    cloud_harm = 0

    for d in cloud_matched:
        if d["match_type"] == "correct":
            matched_gt = d["matched_gt_id"]
            if matched_gt in high_matched_gt_ids:
                cloud_redundant += 1
            elif matched_gt in low_matched_gt_ids:
                cloud_confirm += 1
            else:
                cloud_rescue += 1
            cloud_gain += 1
        elif d["match_type"] == "fp":
            cloud_harm += 1

    return {
        "img_id": img_id,
        "cloud_total": len(cloud_matched),
        "cloud_tp": cloud_tp,
        "cloud_fp": cloud_fp,
        "cloud_class_err": cloud_class_err,
        "cloud_miss": cloud_miss,
        "gt_in_crop": len(gt_in_crop),
        "cloud_gain": cloud_gain,
        "cloud_confirm": cloud_confirm,
        "cloud_rescue": cloud_rescue,
        "cloud_redundant": cloud_redundant,
        "cloud_harm": cloud_harm,
        "yolo_high_total": len(high_conf_yolo),
        "yolo_high_tp": sum(1 for d in high_matched if d["match_type"] == "correct"),
        "yolo_low_total": len(low_conf_yolo),
        "yolo_low_tp": yolo_low_tp,
        "yolo_low_fp": yolo_low_fp,
        "cloud_tp_by_cat": defaultdict(int),
        "cloud_fp_by_cat": defaultdict(int),
        "cloud_miss_by_cat": defaultdict(int),
        "cloud_tp_by_size": defaultdict(int),
        "cloud_fp_by_size": defaultdict(int),
        "cloud_miss_by_size": defaultdict(int),
        "cloud_matched_detail": cloud_matched,
        "gt_in_crop_detail": [g["id"] for g in gt_in_crop],
        "gt_in_crop_unmatched_detail": [g for g in gt_in_crop_unmatched],
    }

# ---------- 主函数 ----------

def run_analysis(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[加载] YOLO 模型: {YOLO_MODEL_PATH}")
    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL_PATH)

    gt_coco, image_map, ann_map = load_gt_annotations(GT_ANNOTATION_PATH)
    image_ids = sorted(ann_map.keys())
    if MAX_IMAGES > 0:
        image_ids = image_ids[:MAX_IMAGES]
    print(f"[实验] 处理 {len(image_ids)} 张图片")
    print(f"  分流阈值: {SPLIT_THRESHOLD} | 裁剪比例: {SCALE_FACTOR:.0%} | 匹配 IoU: {MATCH_IOU}\n")

    global_stats = {
        "total_images": 0,
        "images_with_cloud": 0,
        "cloud_total": 0,
        "cloud_tp": 0,
        "cloud_fp": 0,
        "cloud_class_err": 0,
        "cloud_miss": 0,
        "gt_in_crop": 0,
        "cloud_gain": 0,
        "cloud_confirm": 0,
        "cloud_rescue": 0,
        "cloud_redundant": 0,
        "cloud_harm": 0,
        "yolo_low_total": 0,
        "yolo_low_tp": 0,
        "yolo_low_fp": 0,
        "yolo_high_total": 0,
        "yolo_high_tp": 0,
    }

    cloud_tp_by_cat = defaultdict(int)
    cloud_fp_by_cat = defaultdict(int)
    cloud_miss_by_cat = defaultdict(int)
    gt_in_crop_by_cat = defaultdict(int)

    cloud_tp_by_size = defaultdict(int)
    cloud_fp_by_size = defaultdict(int)
    cloud_miss_by_size = defaultdict(int)
    gt_in_crop_by_size = defaultdict(int)

    per_image_results = []

    for img_id in tqdm(image_ids, desc="Error Analysis"):
        img_info = image_map[img_id]
        img_path = os.path.join(IMAGE_DIR, img_info["file_name"])
        if not os.path.exists(img_path):
            continue
        try:
            img_array = load_image(img_path)
        except Exception as e:
            print(f"  [跳过] {img_info['file_name']}: {e}")
            continue

        actual_h, actual_w = img_array.shape[:2]
        gt_w, gt_h = img_info["width"], img_info["height"]
        gt_anns = ann_map[img_id]

        # YOLO 推理
        yolo_dets = run_yolo(model, img_path, conf=YOLO_CONF)
        if len(yolo_dets) == 0:
            global_stats["total_images"] += 1
            continue

        # 转 GT 空间 COCO 格式
        yolo_coco = []
        for d in yolo_dets:
            x1, y1, x2, y2 = d["bbox"]
            xywh = xyxy_to_coco_xywh(x1, y1, x2, y2)
            gt_bbox = scale_bbox_to_gt(xywh, actual_w, actual_h, gt_w, gt_h)
            if gt_bbox[2] <= 0 or gt_bbox[3] <= 0:
                continue
            yolo_coco.append({
                "image_id": img_id,
                "category_id": d["class_id"] + 1,
                "bbox": gt_bbox,
                "score": d["score"],
            })

        # 置信度分流
        high_conf = [d for d in yolo_coco if d["score"] >= SPLIT_THRESHOLD and d["score"] >= EVAL_CONF]
        low_conf = [d for d in yolo_coco if d["score"] < SPLIT_THRESHOLD and d["score"] >= EVAL_CONF]

        cloud_dets_coco = []
        crop_region = None

        if len(low_conf) > 0:
            global_stats["images_with_cloud"] += 1

            # 计算裁剪区域 (实际图片空间)
            low_conf_actual = []
            for d in low_conf:
                gt_bbox = d["bbox"]
                sx, sy = actual_w / gt_w, actual_h / gt_h
                actual_bbox = [gt_bbox[0] * sx, gt_bbox[1] * sy, gt_bbox[2] * sx, gt_bbox[3] * sy]
                low_conf_actual.append({
                    "bbox": [actual_bbox[0], actual_bbox[1],
                             actual_bbox[0] + actual_bbox[2], actual_bbox[1] + actual_bbox[3]]
                })

            crop_region = compute_crop_region(actual_w, actual_h, low_conf_actual, SCALE_FACTOR)
            cx1, cy1, cx2, cy2 = crop_region
            crop = img_array[cy1:cy2, cx1:cx2].copy()

            # 云端推理
            cloud_dets = send_to_cloud(crop, conf=CLOUD_CONF)

            # 坐标映射: crop局部 → 实际图片 → GT空间
            for det in cloud_dets:
                bx1, by1, bx2, by2 = parse_bbox(det["bbox"])
                score = det.get("score", det.get("confidence", 0.0))
                cls_id = det.get("class_id", det.get("category_id", 0))
                if score < CLOUD_CONF:
                    continue
                actual_x1 = cx1 + bx1
                actual_y1 = cy1 + by1
                actual_x2 = cx1 + bx2
                actual_y2 = cy1 + by2
                xywh = xyxy_to_coco_xywh(actual_x1, actual_y1, actual_x2, actual_y2)
                gt_bbox = scale_bbox_to_gt(xywh, actual_w, actual_h, gt_w, gt_h)
                if gt_bbox[2] <= 0 or gt_bbox[3] <= 0:
                    continue
                cloud_dets_coco.append({
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": gt_bbox,
                    "score": float(score),
                })

        # 如果没有低置信度，跳过云端分析但仍计入统计
        if crop_region is None:
            global_stats["total_images"] += 1
            continue

        # Error Analysis
        result = analyze_image(
            img_id, img_info, img_array, gt_anns,
            yolo_dets, cloud_dets_coco, crop_region,
            high_conf, low_conf,
        )

        # 更新全局统计
        global_stats["total_images"] += 1
        global_stats["cloud_total"] += result["cloud_total"]
        global_stats["cloud_tp"] += result["cloud_tp"]
        global_stats["cloud_fp"] += result["cloud_fp"]
        global_stats["cloud_class_err"] += result["cloud_class_err"]
        global_stats["cloud_miss"] += result["cloud_miss"]
        global_stats["gt_in_crop"] += result["gt_in_crop"]
        global_stats["cloud_gain"] += result["cloud_gain"]
        global_stats["cloud_confirm"] += result["cloud_confirm"]
        global_stats["cloud_rescue"] += result["cloud_rescue"]
        global_stats["cloud_redundant"] += result["cloud_redundant"]
        global_stats["cloud_harm"] += result["cloud_harm"]
        global_stats["yolo_low_total"] += result["yolo_low_total"]
        global_stats["yolo_low_tp"] += result["yolo_low_tp"]
        global_stats["yolo_low_fp"] += result["yolo_low_fp"]
        global_stats["yolo_high_total"] += result["yolo_high_total"]
        global_stats["yolo_high_tp"] += result["yolo_high_tp"]

        # 按类别统计
        for d in result["cloud_matched_detail"]:
            cat = d["class_id"]
            if d["match_type"] == "correct":
                cloud_tp_by_cat[cat] += 1
            elif d["match_type"] == "fp":
                cloud_fp_by_cat[cat] += 1

        for g in result["gt_in_crop_unmatched_detail"]:
            cloud_miss_by_cat[g["category_id"]] += 1
            cloud_miss_by_size[g["size"]] += 1

        # GT in crop 按类别和尺寸
        for g in [gg for gg in ann_map[img_id] if gg["id"] in result["gt_in_crop_detail"]]:
            gt_in_crop_by_cat[g["category_id"]] += 1
            gt_in_crop_by_size[get_size_category(g["area"])] += 1

        # cloud TP/FP 按尺寸
        for d in result["cloud_matched_detail"]:
            if d["match_type"] == "correct" and d["matched_gt_id"]:
                for g in gt_anns:
                    if g["id"] == d["matched_gt_id"]:
                        cloud_tp_by_size[get_size_category(g["area"])] += 1
                        break
            elif d["match_type"] == "fp":
                det_area = (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
                cloud_fp_by_size[get_size_category(det_area)] += 1

        per_image_results.append({
            "img_id": img_id,
            "cloud_total": result["cloud_total"],
            "cloud_tp": result["cloud_tp"],
            "cloud_fp": result["cloud_fp"],
            "cloud_miss": result["cloud_miss"],
            "cloud_gain": result["cloud_gain"],
            "cloud_harm": result["cloud_harm"],
        })

        if len(per_image_results) % CHECKPOINT_INTERVAL == 0:
            with open(os.path.join(OUTPUT_DIR, "checkpoint.json"), "w") as f:
                json.dump({"processed": len(per_image_results), "stats": global_stats}, f, default=str)

    # ===== 输出 =====
    s = global_stats
    print("\n" + "=" * 70)
    print(f"Cloud Negative Gain Error Analysis | {s['total_images']} 张图片")
    print(f"分流阈值: {SPLIT_THRESHOLD} | 裁剪: {SCALE_FACTOR:.0%} | 匹配IoU: {MATCH_IOU}")
    print("=" * 70)

    # ---- 1. 云端检测质量总表 ----
    print(f"\n{'='*70}")
    print("1. 云端检测质量总表")
    print(f"{'='*70}")
    cloud_precision = s["cloud_tp"] / max(s["cloud_total"], 1)
    cloud_recall = s["cloud_tp"] / max(s["cloud_tp"] + s["cloud_miss"], 1)
    print(f"  {'指标':<25} {'数量':>8} {'占比':>8}")
    print(f"  {'-'*45}")
    print(f"  {'云端总检测':<25} {s['cloud_total']:>8} {'100%':>8}")
    print(f"  {'  Correct (TP)':<25} {s['cloud_tp']:>8} {s['cloud_tp']/max(s['cloud_total'],1)*100:>7.1f}%")
    print(f"  {'  False Positive (FP)':<25} {s['cloud_fp']:>8} {s['cloud_fp']/max(s['cloud_total'],1)*100:>7.1f}%")
    print(f"  {'  Class Error':<25} {s['cloud_class_err']:>8} {s['cloud_class_err']/max(s['cloud_total'],1)*100:>7.1f}%")
    print(f"  {'GT在裁剪区域内':<25} {s['gt_in_crop']:>8}")
    print(f"  {'  Cloud Miss (FN)':<25} {s['cloud_miss']:>8} {s['cloud_miss']/max(s['gt_in_crop'],1)*100:>7.1f}%")
    print(f"  {'  Cloud Detected':<25} {s['cloud_tp']:>8} {s['cloud_tp']/max(s['gt_in_crop'],1)*100:>7.1f}%")
    print(f"\n  云端 Precision: {cloud_precision:.4f}")
    print(f"  云端 Recall:    {cloud_recall:.4f}")

    # ---- 2. 按类别分解 ----
    print(f"\n{'='*70}")
    print("2. 按类别分解")
    print(f"{'='*70}")
    print(f"  {'类别':<15} {'GT在ROI':>8} {'TP':>6} {'FP':>6} {'Miss':>6} {'Prec':>8} {'Recall':>8}")
    print(f"  {'-'*60}")
    for cid in sorted(set(list(cloud_tp_by_cat.keys()) + list(cloud_fp_by_cat.keys()) + list(cloud_miss_by_cat.keys()))):
        name = CAT_NAMES.get(cid, f"cls{cid}")
        tp = cloud_tp_by_cat.get(cid, 0)
        fp = cloud_fp_by_cat.get(cid, 0)
        miss = cloud_miss_by_cat.get(cid, 0)
        gt_crop = gt_in_crop_by_cat.get(cid, 0)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + miss, 1)
        print(f"  {name:<15} {gt_crop:>8} {tp:>6} {fp:>6} {miss:>6} {prec:>8.4f} {rec:>8.4f}")

    # ---- 3. 按尺寸分解 ----
    print(f"\n{'='*70}")
    print("3. 按目标尺寸分解")
    print(f"{'='*70}")
    print(f"  {'尺寸':<10} {'GT在ROI':>8} {'TP':>6} {'FP':>6} {'Miss':>6} {'Prec':>8} {'Recall':>8}")
    print(f"  {'-'*55}")
    for size in ["small", "medium", "large"]:
        tp = cloud_tp_by_size.get(size, 0)
        fp = cloud_fp_by_size.get(size, 0)
        miss = cloud_miss_by_size.get(size, 0)
        gt_crop = gt_in_crop_by_size.get(size, 0)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + miss, 1)
        print(f"  {size:<10} {gt_crop:>8} {tp:>6} {fp:>6} {miss:>6} {prec:>8.4f} {rec:>8.4f}")

    # ---- 4. 云端增益 vs 负增益 ----
    print(f"\n{'='*70}")
    print("4. 云端增益 vs 负增益")
    print(f"{'='*70}")
    print(f"  {'类型':<30} {'数量':>8} {'说明'}")
    print(f"  {'-'*70}")
    print(f"  {'Cloud Gain (总增益)':<30} {s['cloud_gain']:>8}  云端正确检出 GT")
    print(f"  {'  Cloud Rescue (救援)':<30} {s['cloud_rescue']:>8}  YOLO完全漏检, 云端检出")
    print(f"  {'  Cloud Confirm (确认)':<30} {s['cloud_confirm']:>8}  YOLO低置信, 云端确认")
    print(f"  {'Cloud Redundant (冗余)':<30} {s['cloud_redundant']:>8}  与YOLO高置信重复, NMS去重")
    print(f"  {'Cloud Harm (负增益)':<30} {s['cloud_harm']:>8}  云端 FP 进入融合")
    print(f"  {'Cloud Miss (漏检)':<30} {s['cloud_miss']:>8}  GT在ROI内但云端未检出")
    print(f"  {'Cloud Class Error':<30} {s['cloud_class_err']:>8}  位置对但类别错")

    net_gain = s["cloud_gain"] - s["cloud_harm"]
    print(f"\n  净增益 = Gain({s['cloud_gain']}) - Harm({s['cloud_harm']}) = {net_gain}")
    print(f"  增益/负增益比 = {s['cloud_gain'] / max(s['cloud_harm'], 1):.2f}:1")

    # ---- 5. YOLO 低置信检测质量 ----
    print(f"\n{'='*70}")
    print("5. YOLO 低置信检测质量 (被上传云端的部分)")
    print(f"{'='*70}")
    yolo_low_prec = s["yolo_low_tp"] / max(s["yolo_low_total"], 1)
    print(f"  低置信检测总数: {s['yolo_low_total']}")
    print(f"  其中正确 (TP):  {s['yolo_low_tp']} ({yolo_low_prec*100:.1f}%)")
    print(f"  其中误检 (FP):  {s['yolo_low_fp']} ({(1-yolo_low_prec)*100:.1f}%)")
    print(f"  → {'YOLO低置信检测大部分是正确的' if yolo_low_prec > 0.5 else 'YOLO低置信检测大部分是误检'}")

    # ---- 6. 逐图统计 (前20张) ----
    print(f"\n{'='*70}")
    print("6. 逐图统计 (前20张有云端检测的图片)")
    print(f"{'='*70}")
    print(f"  {'img_id':<12} {'Cloud':>6} {'TP':>4} {'FP':>4} {'Miss':>5} {'Gain':>5} {'Harm':>5}")
    print(f"  {'-'*45}")
    for r in per_image_results[:20]:
        if r["cloud_total"] > 0:
            print(f"  {r['img_id']:<12} {r['cloud_total']:>6} {r['cloud_tp']:>4} {r['cloud_fp']:>4} {r['cloud_miss']:>5} {r['cloud_gain']:>5} {r['cloud_harm']:>5}")

    # ---- 保存 ----
    output = {
        "summary": {k: v for k, v in s.items()},
        "by_category": {
            CAT_NAMES.get(cid, f"cls{cid}"): {
                "gt_in_crop": gt_in_crop_by_cat.get(cid, 0),
                "tp": cloud_tp_by_cat.get(cid, 0),
                "fp": cloud_fp_by_cat.get(cid, 0),
                "miss": cloud_miss_by_cat.get(cid, 0),
            }
            for cid in sorted(set(list(cloud_tp_by_cat.keys()) + list(cloud_fp_by_cat.keys()) + list(cloud_miss_by_cat.keys())))
        },
        "by_size": {
            size: {
                "gt_in_crop": gt_in_crop_by_size.get(size, 0),
                "tp": cloud_tp_by_size.get(size, 0),
                "fp": cloud_fp_by_size.get(size, 0),
                "miss": cloud_miss_by_size.get(size, 0),
            }
            for size in ["small", "medium", "large"]
        },
        "per_image": per_image_results,
        "config": {
            "split_threshold": SPLIT_THRESHOLD,
            "scale_factor": SCALE_FACTOR,
            "match_iou": MATCH_IOU,
            "yolo_conf": YOLO_CONF,
            "cloud_conf": CLOUD_CONF,
        },
    }

    output_path = os.path.join(OUTPUT_DIR, "cloud_negative_gain_analysis.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[保存] 结果 -> {output_path}")

    ckpt = os.path.join(OUTPUT_DIR, "checkpoint.json")
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print("[清理] checkpoint 已删除")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud Negative Gain Error Analysis")
    parser.add_argument("--max-images", type=int, default=0, help="最多处理图片数 (0=全部)")
    parser.add_argument("--server", type=str, default=CLOUD_SERVER_URL, help="云端服务器地址")
    parser.add_argument("--split", type=float, default=SPLIT_THRESHOLD, help="置信度分流阈值")
    args = parser.parse_args()

    MAX_IMAGES = args.max_images
    CLOUD_SERVER_URL = args.server
    SPLIT_THRESHOLD = args.split

    run_analysis(args)
