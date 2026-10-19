#!/usr/bin/env python3
"""Re-evaluate saved COCO predictions, including per-category AP and subsets."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import pathlib
import statistics
from collections.abc import Iterable


def read_ids(path: pathlib.Path | None) -> set[int] | None:
    if path is None:
        return None
    raw = path.read_text(encoding="utf-8-sig").strip()
    if not raw:
        return set()
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
        if isinstance(data, dict):
            data = data.get("image_ids", data.get("ids", []))
        return {int(v) for v in data}
    lines = raw.splitlines()
    if "," in lines[0] or "image_id" in lines[0].lower():
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return {int(row["image_id"]) for row in csv.DictReader(f) if row.get("image_id")}
    return {int(line.strip()) for line in lines if line.strip()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", type=pathlib.Path, required=True)
    ap.add_argument("--predictions", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--image-ids", type=pathlib.Path,
                    help="Optional JSON, CSV with image_id column, or one ID per line.")
    ap.add_argument("--score-floor", type=float, default=0.05)
    args = ap.parse_args()
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ids = read_ids(args.image_ids)
    with args.predictions.open("r", encoding="utf-8") as f:
        predictions = json.load(f)
    if not isinstance(predictions, list):
        ap.error("predictions JSON must be a COCO detections list")
    selected = [p for p in predictions
                if float(p.get("score", -1)) >= args.score_floor
                and (ids is None or int(p["image_id"]) in ids)]
    if not selected:
        ap.error("no predictions remain after image/score filtering; pycocotools loadRes cannot evaluate an empty list")

    coco_gt = COCO(str(args.gt))
    coco_dt = coco_gt.loadRes(selected)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.useCats = 1
    evaluator.params.maxDets = [1, 10, 100]
    if ids is None:
        eval_ids = sorted(int(image["id"]) for image in coco_gt.dataset["images"])
    else:
        known = {int(image["id"]) for image in coco_gt.dataset["images"]}
        eval_ids = sorted(ids & known)
    evaluator.params.imgIds = eval_ids
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "coco_eval.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log):
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()

    stats_names = ["AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L",
                   "AR_1", "AR_10", "AR_100", "AR_S", "AR_M", "AR_L"]
    summary = {name: float(value) for name, value in zip(stats_names, evaluator.stats)}
    summary.update({"n_images": len(eval_ids), "n_predictions": len(selected),
                    "score_floor": args.score_floor,
                    "prediction_path": str(args.predictions),
                    "ground_truth_path": str(args.gt),
                    "image_ids_path": str(args.image_ids) if args.image_ids else None})

    precision = evaluator.eval["precision"]  # [IoU, recall, category, area, maxDet]
    area_index = evaluator.params.areaRngLbl.index("all")
    maxdet_index = len(evaluator.params.maxDets) - 1
    categories = {int(cat["id"]): cat["name"] for cat in coco_gt.dataset["categories"]}
    per_category = []
    for cat_index, cat_id in enumerate(evaluator.params.catIds):
        values = precision[:, :, cat_index, area_index, maxdet_index]
        valid = values[values > -1]
        per_category.append({"category_id": int(cat_id),
                             "category_name": categories.get(int(cat_id), ""),
                             "AP": float(valid.mean()) if valid.size else -1.0,
                             "valid_precision_cells": int(valid.size)})
    with (args.out_dir / "per_category_ap.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category_id", "category_name", "AP", "valid_precision_cells"])
        writer.writeheader()
        writer.writerows(per_category)
    summary["per_category_ap_csv"] = str(args.out_dir / "per_category_ap.csv")
    (args.out_dir / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
