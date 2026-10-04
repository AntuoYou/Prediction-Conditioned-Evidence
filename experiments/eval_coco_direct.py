import json
import sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

GT_PATH = "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"

def normalize_image_id(x):
    if isinstance(x, int):
        return x
    s = str(x)
    if s.endswith(".jpg"):
        s = s[:-4]
    return int(s)

def main():
    if len(sys.argv) < 2:
        print("Usage: python experiments/eval_coco_direct.py pred.json")
        sys.exit(1)

    pred_path = sys.argv[1]

    with open(GT_PATH) as f:
        gt = json.load(f)

    with open(pred_path) as f:
        pred = json.load(f)

    if isinstance(pred, dict):
        if "results" in pred:
            pred = pred["results"]
        elif "annotations" in pred:
            pred = pred["annotations"]
        elif "detections" in pred:
            pred = pred["detections"]
        else:
            raise ValueError(f"Unknown pred dict keys: {list(pred.keys())[:20]}")

    coco_results = []
    for d in pred:
        if "image_id" not in d or "bbox" not in d:
            continue

        image_id = normalize_image_id(d["image_id"])
        bbox = d["bbox"]

        if isinstance(bbox, dict):
            x1 = float(bbox["x1"])
            y1 = float(bbox["y1"])
            x2 = float(bbox["x2"])
            y2 = float(bbox["y2"])
            bbox = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
        else:
            bbox = [float(x) for x in bbox]

        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue

        if "category_id" in d:
            category_id = int(d["category_id"])
        elif "class_id" in d:
            category_id = int(d["class_id"]) + 1
        else:
            continue

        score = float(d.get("score", d.get("confidence", 1.0)))

        coco_results.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "score": score,
            }
        )

    pred_ids = {d["image_id"] for d in coco_results}
    gt_images = [img for img in gt["images"] if int(img["id"]) in pred_ids]
    gt_image_ids = {int(img["id"]) for img in gt_images}
    gt_annotations = [ann for ann in gt["annotations"] if int(ann["image_id"]) in gt_image_ids]

    subset_gt = {
        "info": gt.get("info", {}),
        "licenses": gt.get("licenses", []),
        "images": gt_images,
        "annotations": gt_annotations,
        "categories": gt["categories"],
    }

    out_gt = pred_path.replace(".json", "_direct_subset_gt.json")
    out_pred = pred_path.replace(".json", "_direct_coco.json")

    with open(out_gt, "w") as f:
        json.dump(subset_gt, f)

    with open(out_pred, "w") as f:
        json.dump(coco_results, f)

    print("Pred:", pred_path)
    print("Subset images:", len(gt_images))
    print("Subset annotations:", len(gt_annotations))
    print("COCO results:", len(coco_results))
    print("Pred image id sample:", sorted(list(pred_ids))[:10])
    print("GT image id sample:", [img["id"] for img in gt_images[:10]])

    if not coco_results or not gt_images:
        print("No valid predictions or no matched GT images.")
        return

    coco_gt = COCO(out_gt)
    coco_dt = coco_gt.loadRes(out_pred)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    precisions = coco_eval.eval["precision"]
    cats = coco_gt.loadCats(coco_gt.getCatIds())

    print("\nPer-class AP50:")
    for idx, cat in enumerate(cats):
        p = precisions[0, :, idx, 0, 2]
        p = p[p > -1]
        ap50 = float(np.mean(p)) if len(p) else float("nan")
        pred_count = sum(1 for d in coco_results if d["category_id"] == cat["id"])
        gt_count = len(coco_gt.getAnnIds(catIds=[cat["id"]]))
        print(f'  {cat["name"]:25s}: AP50={ap50:.4f}, GT={gt_count}, Pred={pred_count}')

if __name__ == "__main__":
    main()
