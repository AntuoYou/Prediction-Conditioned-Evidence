#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Byte-only calibration for the simple-ROI B3/B6 operating points.

This script uses only COCO image metadata (not labels) and never calls the
cloud detector. It measures complete v2 packet lengths on a deterministic
image sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from statistics import mean


def csv_values(value: str, cast):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--gt", required=True, help="COCO JSON; only image metadata is read")
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--edge", required=True)
    ap.add_argument("--edge-conf", type=float, default=0.01)
    ap.add_argument("--output", required=True)
    ap.add_argument("--targets-kb", default="87,311")
    ap.add_argument("--sample-images", type=int, default=31)
    ap.add_argument("--expands", default="0.1,0.3,0.7")
    ap.add_argument("--roi-scales", default="0.5,0.75,1.0")
    ap.add_argument("--qualities", default="1,5,10,20,35,50,65,80,90,95")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo_root)
    sys.path.insert(0, repo)
    import cv2
    from protocol_v2 import run_a
    from protocol_v2 import selectors

    with open(args.gt, encoding="utf-8") as f:
        images = sorted(json.load(f)["images"], key=lambda x: int(x["id"]))
    if not images:
        raise RuntimeError("COCO image list is empty")
    n = min(max(1, args.sample_images), len(images))
    if n == 1:
        sample = [images[0]]
    else:
        # Evenly spaced deterministic sample over sorted image IDs.
        positions = [round(i * (len(images) - 1) / (n - 1)) for i in range(n)]
        sample = [images[i] for i in positions]

    edge = selectors.load_edge_cache(args.edge, args.edge_conf)
    expands = csv_values(args.expands, float)
    roi_scales = csv_values(args.roi_scales, float)
    qualities = csv_values(args.qualities, int)
    targets = csv_values(args.targets_kb, float)
    rows = []

    for expand in expands:
        for roi_scale in roi_scales:
            for quality in qualities:
                cfg = dict(run_a.CONFIGS["B3"])
                cfg.update({"expand": expand, "roi_scale": roi_scale,
                            "q_roi": quality})
                sizes = []
                for im in sample:
                    iid = int(im["id"])
                    if iid not in edge:
                        raise RuntimeError("edge cache missing image_id=%d" % iid)
                    path = os.path.join(args.img_dir, im["file_name"])
                    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise RuntimeError("failed to read image: " + path)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    packet, _n_unit, _trace = run_a.build_packet(
                        cfg, rgb, iid, edge, seed=20261018)
                    sizes.append(len(packet))
                avg = mean(sizes)
                row = {
                    "expand": expand,
                    "roi_scale": roi_scale,
                    "q_roi": quality,
                    "n_images": len(sizes),
                    "mean_packet_bytes": avg,
                    "mean_packet_kb": avg / 1024.0,
                    "min_packet_bytes": min(sizes),
                    "max_packet_bytes": max(sizes),
                }
                for target in targets:
                    key = "within_5pct_%gkb" % target
                    row[key] = abs(avg / 1024.0 - target) <= 0.05 * target
                rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fields = list(rows[0]) if rows else []
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    for target in targets:
        best = min(rows, key=lambda r: abs(r["mean_packet_kb"] - target))
        print("target=%gKB nearest=expand:%g roi_scale:%g q_roi:%d mean:%.3fKB within_5pct=%s" % (
            target, best["expand"], best["roi_scale"], best["q_roi"],
            best["mean_packet_kb"], best["within_5pct_%gkb" % target]))
    print("candidate_table=%s sample_images=%d candidates=%d" % (
        os.path.abspath(args.output), len(sample), len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
