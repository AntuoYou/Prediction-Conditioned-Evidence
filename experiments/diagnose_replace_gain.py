import os
import sys
import json
import cv2

sys.path.insert(0, "/app/Salience-DETR")
sys.path.insert(0, "/app/Salience-DETR/edge")

from edge.edge_pipeline import init_yolo, run_yolo, YOLO_IMG_DIR, CLOUD_URL, CLOUD_CONF
from split_by_confidence import split_by_confidence
from roi_crop import prepare_cloud_requests
from client import CloudClient

GT_PATH = "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"

def xyxy_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union

def det_bbox_to_xyxy(d):
    b = d["bbox"]
    if isinstance(b, dict):
        return [float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])]
    return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]

def cloud_bbox_to_global_xyxy(d, crop_bbox):
    b = d.get("bbox")
    if isinstance(b, dict):
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
    else:
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]

    ox, oy = float(crop_bbox[0]), float(crop_bbox[1])

    return [
        float(x1) + ox,
        float(y1) + oy,
        float(x2) + ox,
        float(y2) + oy,
    ]

def load_gt():
    with open(GT_PATH) as f:
        gt = json.load(f)

    file_to_id = {img["file_name"]: img["id"] for img in gt["images"]}
    anns_by_img = {}

    for ann in gt["annotations"]:
        img_id = ann["image_id"]
        x, y, w, h = ann["bbox"]
        anns_by_img.setdefault(img_id, []).append({
            "category_id": int(ann["category_id"]),
            "bbox_xyxy": [float(x), float(y), float(x + w), float(y + h)],
            "ann_id": ann.get("id", -1),
        })

    return file_to_id, anns_by_img

def best_iou_to_gt(box_xyxy, class_id_0based, gts):
    coco_cat = int(class_id_0based) + 1

    best_iou = 0.0
    best_gt = None

    for gt in gts:
        if int(gt["category_id"]) != coco_cat:
            continue

        iou = xyxy_iou(box_xyxy, gt["bbox_xyxy"])
        if iou > best_iou:
            best_iou = iou
            best_gt = gt

    return best_iou, best_gt

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=300)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--output", type=str, default="experiments/replace_gain_diagnosis_300.json")
    args = parser.parse_args()

    file_to_id, anns_by_img = load_gt()

    init_yolo()
    cloud = CloudClient(url=CLOUD_URL)

    img_files = sorted([
        f for f in os.listdir(YOLO_IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if args.num is not None:
        img_files = img_files[:args.num]

    records = []

    stats = {
        "total_low_conf": 0,
        "cloud_available": 0,
        "yolo_tp": 0,
        "cloud_tp": 0,
        "tp_to_fn": 0,
        "fp_to_tp": 0,
        "iou_improved": 0,
        "iou_degraded": 0,
        "same_or_close": 0,
    }

    for idx, fn in enumerate(img_files, 1):
        print(f"[{idx}/{len(img_files)}] {fn}")

        path = os.path.join(YOLO_IMG_DIR, fn)
        img = cv2.imread(path)
        if img is None:
            continue

        img_id = file_to_id.get(fn)
        if img_id is None:
            continue

        gts = anns_by_img.get(img_id, [])

        all_dets = run_yolo(path)
        split = split_by_confidence(all_dets, conf_threshold=args.threshold)
        low_dets = split.low_conf_detections

        cloud_requests = prepare_cloud_requests(img, low_dets)

        if cloud_requests and cloud.health():
            cloud_results = cloud.batch_detect(cloud_requests, conf_threshold=CLOUD_CONF)
        else:
            cloud_results = []

        cloud_map = {}
        for cr in cloud_results:
            if cr.get("low_conf_index") is not None:
                cloud_map[cr.get("low_conf_index")] = cr

        for i, yolo_det in enumerate(low_dets):
            stats["total_low_conf"] += 1

            yolo_box = list(map(float, yolo_det.bbox))
            yolo_cls = int(yolo_det.class_id)
            yolo_iou, yolo_gt = best_iou_to_gt(yolo_box, yolo_cls, gts)
            yolo_is_tp = yolo_iou >= 0.5

            if yolo_is_tp:
                stats["yolo_tp"] += 1

            cr = cloud_map.get(i)

            best_cloud = None
            cloud_box = None
            cloud_cls = None
            cloud_score = None
            cloud_iou = 0.0
            cloud_is_tp = False

            if cr and cr.get("success", False) and cr.get("detections"):
                crop_bbox = cr.get("crop_bbox")
                candidates = []

                for d in cr.get("detections", []):
                    raw_cls = int(d.get("class_id", d.get("category_id", -1)))
                    cls0 = raw_cls - 1
                    box = cloud_bbox_to_global_xyxy(d, crop_bbox)
                    score = float(d.get("confidence", d.get("score", 0.0)))
                    iou, gt_match = best_iou_to_gt(box, cls0, gts)

                    candidates.append((iou, score, d, box, cls0))

                if candidates:
                    # 选择对 GT IoU 最高的云端框，用于诊断云端潜在最好表现
                    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
                    cloud_iou, cloud_score, best_cloud, cloud_box, cloud_cls = candidates[0]
                    cloud_is_tp = cloud_iou >= 0.5
                    stats["cloud_available"] += 1

                    if cloud_is_tp:
                        stats["cloud_tp"] += 1

            if yolo_is_tp and not cloud_is_tp:
                stats["tp_to_fn"] += 1

            if (not yolo_is_tp) and cloud_is_tp:
                stats["fp_to_tp"] += 1

            if cloud_box is not None:
                if cloud_iou > yolo_iou + 0.05:
                    stats["iou_improved"] += 1
                elif cloud_iou + 0.05 < yolo_iou:
                    stats["iou_degraded"] += 1
                else:
                    stats["same_or_close"] += 1

            records.append({
                "image_id": fn,
                "low_conf_index": i,
                "yolo_bbox": yolo_box,
                "yolo_score": float(yolo_det.confidence),
                "yolo_class_id": yolo_cls,
                "yolo_class_name": yolo_det.class_name,
                "yolo_best_iou": round(yolo_iou, 4),
                "yolo_is_tp": yolo_is_tp,
                "cloud_available": cloud_box is not None,
                "cloud_bbox": cloud_box,
                "cloud_score": cloud_score,
                "cloud_class_id": cloud_cls,
                "cloud_best_iou": round(cloud_iou, 4),
                "cloud_is_tp": cloud_is_tp,
                "delta_iou": round(cloud_iou - yolo_iou, 4),
            })

    out = {
        "summary": stats,
        "records": records,
    }

    with open(args.output, "w") as f:
        json.dump(out, f)

    print("\n=== Replace Gain Diagnosis ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("Saved:", args.output)

if __name__ == "__main__":
    main()
