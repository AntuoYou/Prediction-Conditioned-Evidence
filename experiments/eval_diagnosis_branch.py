import sys
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if len(sys.argv) < 3:
    print("Usage: python experiments/eval_diagnosis_branch.py <diagnosis_json> <branch>")
    print("branch: low_conf_yolo or cloud_roi_all")
    sys.exit(1)

diag_path = sys.argv[1]
branch = sys.argv[2]

gt_path = "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"
out_path = diag_path.replace(".json", f"_{branch}_coco.json")

coco_gt = COCO(gt_path)
img_map = {
    img["file_name"]: img["id"]
    for img in coco_gt.loadImgs(coco_gt.getImgIds())
}

with open(diag_path) as f:
    data = json.load(f)

items = data["results"]
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

with open(out_path, "w") as f:
    json.dump(coco_results, f)

print("Branch:", branch)
print("COCO results:", len(coco_results))
print("Saved:", out_path)

if len(coco_results) == 0:
    print("No predictions, skip COCOeval.")
    sys.exit(0)

coco_dt = coco_gt.loadRes(out_path)
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
