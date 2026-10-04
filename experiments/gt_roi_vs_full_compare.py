import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLOUD_URL = "http://localhost:8000/detect"
ROI_RESIZE = 512

def expand_bbox_xyxy(bbox_xywh, ratio, img_w, img_h):
    x, y, w, h = bbox_xywh
    cx = x + w / 2.0
    cy = y + h / 2.0
    nw = w * ratio
    nh = h * ratio
    rx1 = max(0.0, cx - nw / 2.0)
    ry1 = max(0.0, cy - nh / 2.0)
    rx2 = min(float(img_w), cx + nw / 2.0)
    ry2 = min(float(img_h), cy + nh / 2.0)
    return [rx1, ry1, rx2, ry2]

def crop_roi_from_image(img, roi_xyxy, target_size=512):
    x1, y1, x2, y2 = [int(v) for v in roi_xyxy]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    resized = cv2.resize(crop, (target_size, target_size))
    return resized

def encode_image(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()

def detect_cloud(img_bytes, orig_w, orig_h, timeout=30, roi_offset=None, roi_scale=None):
    files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
    data = {"conf_threshold": "0.05"}
    try:
        resp = requests.post(CLOUD_URL, files=files, data=data, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        print(f"  Cloud error: {e}")
        return []

    dets = []
    if isinstance(result, dict):
        detections = result.get("detections", [])
        for d in detections:
            bbox = d.get("bbox", {})
            if isinstance(bbox, list) and len(bbox) == 4:
                x1, y1, x2, y2 = [float(v) for v in bbox]
            elif isinstance(bbox, dict):
                x1 = float(bbox.get("x1", 0))
                y1 = float(bbox.get("y1", 0))
                x2 = float(bbox.get("x2", 0))
                y2 = float(bbox.get("y2", 0))
            else:
                continue

            x1 = max(0, min(x1, orig_w))
            x2 = max(0, min(x2, orig_w))
            y1 = max(0, min(y1, orig_h))
            y2 = max(0, min(y2, orig_h))
            if x2 <= x1 or y2 <= y1:
                continue

            class_id = int(d.get("class_id", 0))
            category_id = class_id
            confidence = float(d.get("confidence", 0))
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)

            if roi_offset is not None and roi_scale is not None:
                x1 = roi_offset[0] + x1 * roi_scale[0]
                y1 = roi_offset[1] + y1 * roi_scale[1]
                w = w * roi_scale[0]
                h = h * roi_scale[1]

            dets.append({
                "image_id": d.get("image_id", ""),
                "category_id": category_id,
                "bbox": [x1, y1, w, h],
                "score": confidence,
            })

    return dets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="/app/data/seadronessee/compressed_version/annotations/instances_val2017.json")
    parser.add_argument("--img-root", default="/app/data/seadronessee/compressed_version/val2017")
    parser.add_argument("--cloud-url", default="http://localhost:8000/detect")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--context-ratio", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", default="experiments/gt_roi_vs_full")
    args = parser.parse_args()

    global CLOUD_URL
    CLOUD_URL = args.cloud_url

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading GT...")
    coco_gt = COCO(args.gt)

    all_anns = coco_gt.getAnnIds()
    anns = coco_gt.loadAnns(all_anns)

    print(f"Total annotations: {len(anns)}")

    valid_anns = [a for a in anns if a["area"] > 0]
    print(f"Valid annotations: {len(valid_anns)}")

    sampled = random.sample(valid_anns, min(args.num_samples, len(valid_anns)))
    print(f"Sampled {len(sampled)} annotations")

    # 按 image_id 分组
    by_image = {}
    for ann in sampled:
        img_id = ann["image_id"]
        if img_id not in by_image:
            by_image[img_id] = []
        by_image[img_id].append(ann)

    image_ids = list(by_image.keys())
    print(f"Unique images: {len(image_ids)}")

    # 构建每个样本的 GT（包含原始图像的所有 GT，不只是采样的）
    # 但评估时只看 ROI 区域内的 GT
    full_results = []
    roi_results = []

    roi_stats = {
        "total_samples": 0,
        "cloud_success_roi": 0,
        "cloud_success_full": 0,
        "roi_dets_total": 0,
        "full_dets_total": 0,
    }

    for idx, img_id in enumerate(image_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        img_path = Path(args.img_root) / file_name
        if not img_path.exists():
            img_path = Path(args.img_root) / Path(file_name).name
        if not img_path.exists():
            matches = list(Path(args.img_root).rglob(Path(file_name).name))
            if matches:
                img_path = matches[0]
            else:
                continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        anns_for_img = by_image[img_id]

        # === 全图推理 ===
        full_bytes = encode_image(img)
        full_dets = detect_cloud(full_bytes, img_w, img_h)

        for d in full_dets:
            d["image_id"] = img_id
        full_results.extend(full_dets)
        roi_stats["cloud_success_full"] += 1 if full_dets else 0
        roi_stats["full_dets_total"] += len(full_dets)

        # === ROI 推理（对每个采样的 GT bbox 做一次） ===
        for ann in anns_for_img:
            roi_stats["total_samples"] += 1
            bbox_xywh = ann["bbox"]
            cat_id = ann["category_id"]

            ratio = args.context_ratio
            if cat_id in (1, 5):
                ratio = 2.5

            roi_xyxy = expand_bbox_xyxy(bbox_xywh, ratio, img_w, img_h)
            roi_img = crop_roi_from_image(img, roi_xyxy, ROI_RESIZE)

            if roi_img is None:
                continue

            roi_bytes = encode_image(roi_img)
            roi_w_orig = roi_xyxy[2] - roi_xyxy[0]
            roi_h_orig = roi_xyxy[3] - roi_xyxy[1]
            roi_dets = detect_cloud(
                roi_bytes, img_w, img_h,
                roi_offset=(roi_xyxy[0], roi_xyxy[1]),
                roi_scale=(roi_w_orig / ROI_RESIZE, roi_h_orig / ROI_RESIZE),
            )

            for d in roi_dets:
                d["image_id"] = img_id
            roi_results.extend(roi_dets)
            roi_stats["cloud_success_roi"] += 1 if roi_dets else 0
            roi_stats["roi_dets_total"] += len(roi_dets)

        if (idx + 1) % 20 == 0:
            print(f"[{idx+1}/{len(image_ids)}] processed")

    # 保存结果
    full_out = args.out_prefix + "_full_results.json"
    roi_out = args.out_prefix + "_roi_results.json"
    stat_out = args.out_prefix + "_stats.json"

    with open(full_out, "w") as f:
        json.dump(full_results, f)
    with open(roi_out, "w") as f:
        json.dump(roi_results, f)
    with open(stat_out, "w") as f:
        json.dump(roi_stats, f, indent=2)

    print(f"\nSaved: {full_out} ({len(full_results)} dets)")
    print(f"Saved: {roi_out} ({len(roi_results)} dets)")
    print(f"Saved: {stat_out}")
    print(json.dumps(roi_stats, indent=2))

    # 评估
    print("\n=== Evaluating on sampled images ===")
    pred_ids_full = {d["image_id"] for d in full_results}
    pred_ids_roi = {d["image_id"] for d in roi_results}
    all_pred_ids = pred_ids_full | pred_ids_roi

    for name, dets in [("Full-image cloud", full_results), ("ROI cloud", roi_results)]:
        if not dets:
            print(f"\n{name}: No detections, skipping eval")
            continue

        pred_ids = {d["image_id"] for d in dets}
        gt_imgs = coco_gt.loadImgs(list(pred_ids))
        gt_image_ids = {img["id"] for img in gt_imgs}
        gt_anns_subset = [a for a in coco_gt.loadAnns(coco_gt.getAnnIds()) if a["image_id"] in gt_image_ids]

        subset_gt = {
            "images": gt_imgs,
            "annotations": gt_anns_subset,
            "categories": coco_gt.dataset["categories"],
        }

        import tempfile
        gt_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        json.dump(subset_gt, gt_file)
        gt_file.close()

        dt_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        json.dump(dets, dt_file)
        dt_file.close()

        coco_eval_gt = COCO(gt_file.name)
        coco_dt = coco_eval_gt.loadRes(dt_file.name)
        coco_eval = COCOeval(coco_eval_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        import os
        os.unlink(gt_file.name)
        os.unlink(dt_file.name)

        print(f"\n--- {name} ---")

if __name__ == "__main__":
    main()
