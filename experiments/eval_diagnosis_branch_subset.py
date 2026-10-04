import sys
import json
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if len(sys.argv) < 3:
    print("Usage: python experiments/eval_diagnosis_branch_subset.py <diagnosis_json> <branch>")
    print("branch: low_conf_yolo or cloud_roi_all")
    sys.exit(1)

diag_path = sys.argv[1]
branch = sys.argv[2]

gt_path = "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"
out_pred_path = diag_path.replace(".json", f"_{branch}_coco.json")
out_gt_path = diag_path.replace(".json", "_subset_gt.json")

with open(diag_path) as f:
    diag = json.load(f)

items = diag["results"]
file_names = set(item["image_id"] for item in items)

with open(gt_path) as f:
    gt = json.load(f)

subset_images = [
    img for img in gt["images"]
    if img["file_name"] in file_names
]

subset_img_ids = set(img["id"] for img in subset_images)

subset_annotations = [
    ann for ann in gt["annotations"]
    if ann["image_id"] in subset_img_ids
]

subset_gt = {
    "info": gt.get("info", {}),
    "licenses": gt.get("licenses", []),
    "images": subset_images,
    "annotations": subset_annotations,
    "categories": gt["categories"],
}

with open(out_gt_path, "w") as f:
    json.dump(subset_gt, f)

img_map = {
    img["file_name"]: img["id"]
    for img in subset_images
}

coco_results = []

for item in items:
    fn = item["image_id"]
    if fn not in img_map:
        continue

    image_id = int(img_map[fn])

    for d in item.get(branch, []):
        b = d["bbox"]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]

        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))

        if w <= 0 or h <= 0:
            continue

        coco_results.append({
            "image_id": image_id,
            "category_id": int(d["class_id"]) + 1,
            "bbox": [float(x1), float(y1), w, h],
            "score": float(d["confidence"]),
        })

with open(out_pred_path, "w") as f:
    json.dump(coco_results, f)

print("Branch:", branch)
print("Subset images:", len(subset_images))
print("Subset annotations:", len(subset_annotations))
print("COCO results:", len(coco_results))
print("Saved GT:", out_gt_path)
print("Saved pred:", out_pred_path)

if len(coco_results) == 0:
    print("No predictions, skip COCOeval.")
    sys.exit(0)

coco_gt = COCO(out_gt_path)
coco_dt = coco_gt.loadRes(out_pred_path)

coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

print("\n=== 各类别 AP@0.5 ===")
cats = coco_gt.loadCats(coco_gt.getCatIds())
prec = coco_eval.eval["precision"]

for i, cat in enumerate(cats):
    if cat["id"] == 0:
        continue

    ap50 = prec[0, :, i, 0, 2].mean()
    ngt = len(coco_gt.getAnnIds(catIds=[cat["id"]]))
    npred = len([r for r in coco_results if r["category_id"] == cat["id"]])

    print(f'  {cat["name"]:25s}: AP50={ap50:.4f}, GT={ngt}, Pred={npred}')
