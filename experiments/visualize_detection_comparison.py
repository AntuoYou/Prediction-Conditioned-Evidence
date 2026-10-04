import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import requests
from ultralytics import YOLO

COCO_COLORS = {
    1: (0, 0, 255),    # swimmer - red
    2: (255, 0, 0),    # boat - blue
    3: (0, 255, 0),    # jetski - green
    4: (0, 255, 255),  # life_saving - yellow
    5: (255, 0, 255),  # buoy - magenta
}

CAT_NAMES = {
    1: "swimmer",
    2: "boat",
    3: "jetski",
    4: "life_saving",
    5: "buoy",
}

YOLO_TO_COCO = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}

def load_coco_gt(gt_path):
    with open(gt_path) as f:
        gt = json.load(f)
    by_img = {}
    for ann in gt["annotations"]:
        img_id = ann["image_id"]
        if img_id not in by_img:
            by_img[img_id] = []
        by_img[img_id].append(ann)
    return gt["images"], by_img

def load_cloud_full_results(cloud_path):
    with open(cloud_path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    by_img = {}
    for item in data:
        img_id = item.get("image_id")
        if img_id:
            by_img[img_id] = item.get("detections", [])
    return by_img

def draw_boxes(img, detections, color_map, label_prefix="", score_thresh=0.0, font_scale=0.6):
    out = img.copy()
    for det in detections:
        if isinstance(det, dict):
            if "bbox" in det:
                bbox = det["bbox"]
                if isinstance(bbox, dict):
                    x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
                elif isinstance(bbox, list) and len(bbox) == 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                else:
                    continue
                cat_id = int(det.get("category_id", det.get("class_id", 0) + 1))
                score = float(det.get("score", det.get("confidence", 0)))
            else:
                continue
        else:
            continue

        if score < score_thresh:
            continue

        color = color_map.get(cat_id, (128, 128, 128))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        name = CAT_NAMES.get(cat_id, f"cls{cat_id}")
        text = f"{label_prefix}{name}:{score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(out, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    return out

def run_yolo(model, img_path, conf=0.3):
    results = model(str(img_path), imgsz=640, conf=conf, iou=0.7, verbose=False)[0]
    dets = []
    if results.boxes is not None:
        for box, score, cls_id in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            x1, y1, x2, y2 = [float(v) for v in box]
            cat_id = YOLO_TO_COCO.get(int(cls_id), int(cls_id) + 1)
            dets.append({
                "category_id": cat_id,
                "bbox": [x1, y1, x2, y2],
                "score": float(score),
            })
    return dets

def run_cloud_roi(img, roi_xyxy, orig_w, orig_h):
    x1, y1, x2, y2 = roi_xyxy
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    files = {"file": ("roi.jpg", buf.tobytes(), "image/jpeg")}
    data = {"conf_threshold": "0.05"}
    try:
        resp = requests.post("http://localhost:8000/detect", files=files, data=data, timeout=30)
        resp.raise_for_status()
        result = resp.json()
    except Exception:
        return []

    dets = []
    for d in result.get("detections", []):
        bbox = d.get("bbox", {})
        if isinstance(bbox, list) and len(bbox) == 4:
            cx1, cy1, cx2, cy2 = [float(v) for v in bbox]
        elif isinstance(bbox, dict):
            cx1, cy1, cx2, cy2 = float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"])
        else:
            continue

        # 映射回原始图像坐标
        scale_x = (x2 - x1) / max(1, crop.shape[1])
        scale_y = (y2 - y1) / max(1, crop.shape[0])
        cx1 = x1 + cx1 * scale_x
        cy1 = y1 + cy1 * scale_y
        cx2 = x1 + cx2 * scale_x
        cy2 = y1 + cy2 * scale_y

        cx1 = max(0, min(cx1, orig_w))
        cx2 = max(0, min(cx2, orig_w))
        cy1 = max(0, min(cy1, orig_h))
        cy2 = max(0, min(cy2, orig_h))

        dets.append({
            "category_id": int(d.get("class_id", 0)) + 1,
            "bbox": [cx1, cy1, cx2, cy2],
            "score": float(d.get("confidence", 0)),
        })
    return dets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="/app/data/seadronessee/compressed_version/annotations/instances_val2017.json")
    parser.add_argument("--img-root", default="/app/data/seadronessee/compressed_version/val2017")
    parser.add_argument("--yolo", default="/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt")
    parser.add_argument("--cloud", default="experiments/cloud_full_1547_conf005.json")
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="/app/Salience-DETR/experiments/visualization")
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    images, gt_by_img = load_coco_gt(args.gt)
    cloud_full_by_img = load_cloud_full_results(args.cloud)
    model = YOLO(args.yolo)

    # 选有 GT 且云端有结果的图
    valid_ids = []
    for img in images:
        img_id = img["id"]
        img_name = img["file_name"]
        if img_id in gt_by_img and img_name in cloud_full_by_img:
            valid_ids.append(img_id)

    selected = random.sample(valid_ids, min(args.num_images, len(valid_ids)))
    print(f"Selected {len(selected)} images for visualization")

    for idx, img_id in enumerate(selected, 1):
        img_info = next(i for i in images if i["id"] == img_id)
        file_name = img_info["file_name"]
        img_w, img_h = img_info["width"], img_info["height"]

        img_path = Path(args.img_root) / file_name
        if not img_path.exists():
            img_path = Path(args.img_root) / Path(file_name).name

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        base_name = Path(file_name).stem

        # 1. 原图
        cv2.imwrite(str(out_dir / f"{base_name}_01_original.jpg"), img)

        # 2. GT 标注
        gt_dets = []
        for ann in gt_by_img.get(img_id, []):
            x, y, w, h = ann["bbox"]
            gt_dets.append({
                "category_id": ann["category_id"],
                "bbox": [x, y, x + w, y + h],
                "score": 1.0,
            })
        gt_vis = draw_boxes(img, gt_dets, COCO_COLORS, "GT_", score_thresh=0.0)
        cv2.imwrite(str(out_dir / f"{base_name}_02_gt.jpg"), gt_vis)

        # 3. YOLO 检测
        yolo_dets = run_yolo(model, img_path, conf=0.3)
        yolo_vis = draw_boxes(img, yolo_dets, COCO_COLORS, "YOLO_", score_thresh=0.3)
        cv2.imwrite(str(out_dir / f"{base_name}_03_yolo.jpg"), yolo_vis)

        # 4. 云端全图检测
        cloud_full_dets = []
        for d in cloud_full_by_img.get(file_name, []):
            bbox = d.get("bbox", {})
            if isinstance(bbox, dict):
                cloud_full_dets.append({
                    "category_id": int(d.get("class_id", 0)) + 1,
                    "bbox": [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]],
                    "score": float(d.get("confidence", 0)),
                })
        cloud_full_vis = draw_boxes(img, cloud_full_dets, COCO_COLORS, "CLOUD_", score_thresh=0.05)
        cv2.imwrite(str(out_dir / f"{base_name}_04_cloud_full.jpg"), cloud_full_vis)

        # 5. 选一个 YOLO 低置信目标做 ROI 云端检测
        low_conf_yolo = [d for d in yolo_dets if d["score"] < 0.7]
        if low_conf_yolo:
            target = low_conf_yolo[0]
            x1, y1, x2, y2 = [int(v) for v in target["bbox"]]
            # 扩展 1.5x
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            rx1 = max(0, int(cx - w * 0.75))
            ry1 = max(0, int(cy - h * 0.75))
            rx2 = min(img_w, int(cx + w * 0.75))
            ry2 = min(img_h, int(cy + h * 0.75))

            roi_dets = run_cloud_roi(img, (rx1, ry1, rx2, ry2), img_w, img_h)
            roi_vis = draw_boxes(img, roi_dets, COCO_COLORS, "ROI_", score_thresh=0.05)

            # 画 ROI 裁剪框（白色虚线）
            cv2.rectangle(roi_vis, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)
            cv2.putText(roi_vis, "ROI_crop", (rx1, ry1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imwrite(str(out_dir / f"{base_name}_05_cloud_roi.jpg"), roi_vis)

        print(f"[{idx}/{len(selected)}] Saved visualization for {base_name}")

    print(f"\nAll images saved to: {out_dir}")
    for f in sorted(out_dir.glob("*.jpg")):
        print(f"  {f.name}")

if __name__ == "__main__":
    main()
