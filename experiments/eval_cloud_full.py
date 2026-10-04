import os
import sys
import json
import cv2

sys.path.insert(0, "/app/Salience-DETR")
sys.path.insert(0, "/app/Salience-DETR/edge")

from edge.edge_pipeline import YOLO_IMG_DIR, CLOUD_URL, CLOUD_CONF
from client import CloudClient

def convert_cloud_det(d):
    bbox = d.get("bbox")
    if isinstance(bbox, dict):
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    else:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

    raw_cls = int(d.get("class_id", d.get("category_id", -1)))

    # 云端 DETR 输出 COCO 1-based，最终内部统一为 0-based
    cls0 = raw_cls - 1

    return {
        "bbox": {
            "x1": round(float(x1), 1),
            "y1": round(float(y1), 1),
            "x2": round(float(x2), 1),
            "y2": round(float(y2), 1),
        },
        "confidence": round(float(d.get("confidence", d.get("score", 0.0))), 4),
        "class_id": cls0,
        "class_name": d.get("class_name", str(raw_cls)),
        "source": "cloud_full"
    }

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=YOLO_IMG_DIR)
    parser.add_argument("--num", type=int, default=300)
    parser.add_argument("--output", type=str, default="experiments/cloud_full_300.json")
    parser.add_argument("--cloud_url", type=str, default=CLOUD_URL)
    parser.add_argument("--conf", type=float, default=CLOUD_CONF)
    args = parser.parse_args()

    cloud = CloudClient(url=args.cloud_url)

    img_files = sorted([
        f for f in os.listdir(args.dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if args.num is not None:
        img_files = img_files[:args.num]

    results = []

    for idx, fn in enumerate(img_files, 1):
        path = os.path.join(args.dir, fn)
        img = cv2.imread(path)
        if img is None:
            print(f"[SKIP] {fn}")
            continue

        h, w = img.shape[:2]

        print(f"[{idx}/{len(img_files)}] {fn}")

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue

        cr = cloud.detect(buf.tobytes(), roi_index=0, conf_threshold=args.conf)

        dets = []
        if cr.get("success", False):
            for d in cr.get("detections", []):
                dets.append(convert_cloud_det(d))

        results.append({
            "image_id": fn,
            "image_size": [w, h],
            "detections": dets,
            "cloud_count": len(dets),
        })

    out = {
        "summary": {
            "images": len(results),
            "total_cloud_detections": sum(r["cloud_count"] for r in results),
        },
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(out, f)

    print("\n=== Cloud Full Summary ===")
    print(json.dumps(out["summary"], indent=2, ensure_ascii=False))
    print("Saved:", args.output)

if __name__ == "__main__":
    main()
