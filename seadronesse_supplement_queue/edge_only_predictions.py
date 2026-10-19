#!/usr/bin/env python3
"""Convert frozen YOLO edge-cache rows to original-coordinate COCO predictions."""

from __future__ import annotations

import argparse
import json
import pathlib


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edge-cache", type=pathlib.Path, required=True)
    ap.add_argument("--class-map", type=pathlib.Path, required=True,
                    help='JSON object mapping YOLO class id strings to COCO category IDs, e.g. {"0": 1}.')
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--score-floor", type=float, default=0.05)
    args = ap.parse_args()
    mapping_raw = json.loads(args.class_map.read_text(encoding="utf-8"))
    mapping = {int(key): int(value) for key, value in mapping_raw.items()}
    predictions = []
    unknown_labels = set()
    n_images = 0
    with args.edge_cache.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            n_images += 1
            boxes = row.get("boxes") or []
            scores = row.get("scores") or []
            labels = row.get("labels") or []
            if not (len(boxes) == len(scores) == len(labels)):
                raise ValueError(f"edge cache length mismatch at image_id={row.get('image_id')}")
            for box, score, label in zip(boxes, scores, labels):
                score = float(score)
                label = int(label)
                if score < args.score_floor:
                    continue
                if label not in mapping:
                    unknown_labels.add(label)
                    continue
                x1, y1, x2, y2 = map(float, box)
                width, height = x2 - x1, y2 - y1
                if width <= 0 or height <= 0:
                    continue
                predictions.append({"image_id": int(row["image_id"]),
                                    "category_id": mapping[label],
                                    "bbox": [x1, y1, width, height],
                                    "score": score})
    if unknown_labels:
        raise ValueError("class map missing YOLO labels: " + ",".join(map(str, sorted(unknown_labels))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"n_images": n_images, "n_predictions": len(predictions),
                      "score_floor": args.score_floor, "class_map": mapping,
                      "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
