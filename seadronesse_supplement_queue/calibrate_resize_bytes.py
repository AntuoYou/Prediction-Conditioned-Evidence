#!/usr/bin/env python3
"""Measure real context-only ZIP packet sizes for resize+JPEG candidates.

This is a byte-only calibration: it reads images and calls protocol_v2.codec,
but does not read labels, run inference, or choose a candidate by AP.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from PIL import Image
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    parser.add_argument("--image-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--targets-kb", default="54,314,360")
    parser.add_argument("--sample-step", type=int, default=50,
                        help="Uniformly sample sorted image paths; 50 is about 31/1547 frames.")
    parser.add_argument("--scales", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--qualities", default="50,65,80,85,86,90,95")
    args = parser.parse_args()
    if args.sample_step < 1:
        parser.error("--sample-step must be >= 1")
    sys.path.insert(0, str(args.repo_root.resolve()))
    from protocol_v2 import codec

    images = sorted(p for p in args.image_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[::args.sample_step]
    if not images:
        parser.error(f"no image files found in {args.image_dir}")
    targets = [float(v) for v in args.targets_kb.split(",") if v.strip()]
    scales = [int(v) for v in args.scales.split(",") if v.strip()]
    qualities = [int(v) for v in args.qualities.split(",") if v.strip()]
    if any(s < 1 for s in scales) or any(q < 1 or q > 100 for q in qualities):
        parser.error("scales must be positive and JPEG qualities must be 1..100")

    rows = []
    for scale in scales:
        for quality in qualities:
            sizes = []
            for i, image_path in enumerate(images):
                with Image.open(image_path) as im:
                    rgb = np.asarray(im.convert("RGB"))
                packet = codec.encode_context_only(rgb, i, scale=scale, q=quality)
                sizes.append(len(packet))
            mean_kb = sum(sizes) / len(sizes) / 1024.0
            rows.append({"scale": scale, "jpeg_quality": quality,
                         "n_images": len(sizes), "mean_packet_bytes": sum(sizes) / len(sizes),
                         "mean_packet_kb": mean_kb, "min_packet_bytes": min(sizes),
                         "max_packet_bytes": max(sizes)})

    for target in targets:
        for row in rows:
            row[f"within_5pct_of_{target:g}kb"] = abs(row["mean_packet_kb"] - target) / target <= 0.05
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    for target in targets:
        eligible = [r for r in rows if abs(r["mean_packet_kb"] - target) / target <= 0.05]
        closest = min(rows, key=lambda r: abs(r["mean_packet_kb"] - target))
        print(f"target={target:g}KB  within_5pct={len(eligible)}  "
              f"closest=scale:{closest['scale']} q:{closest['jpeg_quality']} "
              f"mean:{closest['mean_packet_kb']:.2f}KB")
    print(f"candidate_table={args.output} sample_images={len(images)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
