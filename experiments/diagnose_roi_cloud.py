import os
import sys
import json
import time
import cv2

sys.path.insert(0, "/app/Salience-DETR")
sys.path.insert(0, "/app/Salience-DETR/edge")

from edge.edge_pipeline import init_yolo, run_yolo, YOLO_IMG_DIR, CLOUD_URL, CLOUD_CONF
from split_by_confidence import split_by_confidence
from roi_crop import prepare_cloud_requests
from client import CloudClient

def det_to_dict(d, source):
    return {
        "bbox": {
            "x1": round(float(d.bbox[0]), 1),
            "y1": round(float(d.bbox[1]), 1),
            "x2": round(float(d.bbox[2]), 1),
            "y2": round(float(d.bbox[3]), 1),
        },
        "confidence": round(float(d.confidence), 4),
        "class_id": int(d.class_id),
        "class_name": d.class_name,
        "source": source,
    }

def cloud_det_to_global_dict(d, crop_bbox):
    bbox = d.get("bbox")
    if isinstance(bbox, dict):
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    else:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

    ox, oy = float(crop_bbox[0]), float(crop_bbox[1])

    raw_cls = int(d.get("class_id", d.get("category_id", -1)))

    # 云端 DETR 为 COCO 1-based，统一转成 0-based
    cls_id = raw_cls - 1

    return {
        "bbox": {
            "x1": round(float(x1) + ox, 1),
            "y1": round(float(y1) + oy, 1),
            "x2": round(float(x2) + ox, 1),
            "y2": round(float(y2) + oy, 1),
        },
        "confidence": round(float(d.get("confidence", d.get("score", 0.0))), 4),
        "class_id": int(cls_id),
        "class_name": d.get("class_name", str(raw_cls)),
        "source": "cloud_roi_all",
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=YOLO_IMG_DIR)
    parser.add_argument("--num", type=int, default=300)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--output", type=str, default="experiments/roi_cloud_diagnosis_300.json")
    parser.add_argument("--cloud_url", type=str, default=CLOUD_URL)
    args = parser.parse_args()

    init_yolo()
    cloud_client = CloudClient(url=args.cloud_url)

    img_files = sorted([
        f for f in os.listdir(args.dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if args.num is not None:
        img_files = img_files[:args.num]

    results = []
    total_low = 0
    total_cloud = 0
    total_roi = 0
    total_success = 0

    for idx, fn in enumerate(img_files, 1):
        image_path = os.path.join(args.dir, fn)
        img = cv2.imread(image_path)
        if img is None:
            print(f"[SKIP] 读取失败: {image_path}")
            continue

        img_h, img_w = img.shape[:2]

        print(f"[{idx}/{len(img_files)}] {fn}")

        all_dets = run_yolo(image_path)
        split = split_by_confidence(all_dets, conf_threshold=args.threshold)

        low_conf_yolo = [
            det_to_dict(d, "low_conf_yolo")
            for d in split.low_conf_detections
        ]

        cloud_roi_all = []
        cloud_requests = prepare_cloud_requests(img, split.low_conf_detections)
        total_roi += len(cloud_requests)

        if cloud_requests and cloud_client.health():
            cloud_results = cloud_client.batch_detect(cloud_requests, conf_threshold=CLOUD_CONF)
        else:
            cloud_results = []

        for cr in cloud_results:
            if not cr.get("success", False):
                continue

            total_success += 1
            crop_bbox = cr.get("crop_bbox")
            if crop_bbox is None:
                continue

            for d in cr.get("detections", []):
                cloud_roi_all.append(cloud_det_to_global_dict(d, crop_bbox))

        total_low += len(low_conf_yolo)
        total_cloud += len(cloud_roi_all)

        results.append({
            "image_id": fn,
            "image_size": [img_w, img_h],
            "low_conf_yolo_count": len(low_conf_yolo),
            "cloud_roi_all_count": len(cloud_roi_all),
            "low_conf_yolo": low_conf_yolo,
            "cloud_roi_all": cloud_roi_all,
        })

    out = {
        "summary": {
            "images": len(results),
            "total_low_conf_yolo": total_low,
            "total_roi_requests": total_roi,
            "total_cloud_success": total_success,
            "total_cloud_roi_detections": total_cloud,
        },
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(out, f)

    print("\n=== Diagnosis Summary ===")
    print(json.dumps(out["summary"], indent=2, ensure_ascii=False))
    print("Saved:", args.output)

if __name__ == "__main__":
    main()
