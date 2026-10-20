#!/usr/bin/env python3
"""Byte-only calibration for corrected A7@54 KiB and B4 matched to actual B6.

Reads the frozen cfg from prior run manifests and invokes the server checkout's
protocol_v2.run_a.build_packet. It never reads annotations, uses AP, or calls a
cloud detector. Because the original C_dev list is unavailable, the fixed 31
image sample is for byte matching only and must be disclosed as such.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import statistics
import sys

import numpy as np


def load_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def even_sample(images, n):
    images = sorted(images, key=lambda item: int(item["id"]))
    n = min(max(1, n), len(images))
    if n == 1:
        return [images[0]]
    positions = [round(i * (len(images) - 1) / (n - 1)) for i in range(n)]
    return [images[i] for i in positions]


def measure(run_a, cfg, sample, img_dir, edge, seed):
    sizes = []
    for im in sample:
        iid = int(im["id"])
        path = img_dir / im["file_name"]
        bgr = run_a.cv2.imread(str(path), run_a.cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"could not read image: {path}")
        rgb = run_a.cv2.cvtColor(bgr, run_a.cv2.COLOR_BGR2RGB)
        packet, _n_unit, _trace = run_a.build_packet(cfg, rgb, iid, edge, seed=seed)
        sizes.append(len(packet))
    return sizes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=pathlib.Path, required=True)
    ap.add_argument("--gt", type=pathlib.Path, required=True,
                    help="COCO JSON; only the image list/filenames are used")
    ap.add_argument("--img-dir", type=pathlib.Path, required=True)
    ap.add_argument("--edge", type=pathlib.Path, required=True)
    ap.add_argument("--a7-run", default="A7_support_centers_v2")
    ap.add_argument("--b4-base-run", default="A5_v3",
                    help="Existing support+exploration regional run whose cfg defines B4")
    ap.add_argument("--b6-run", default="B6_roi_max_allowed_q95_unmatched_v2")
    ap.add_argument("--edge-conf", type=float, default=0.01)
    ap.add_argument("--sample-images", type=int, default=31)
    ap.add_argument("--a7-target-kib", type=float, default=53.544)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))
    import cv2
    from protocol_v2 import run_a, selectors
    run_a.cv2 = cv2

    gt = load_json(args.gt)
    sample = even_sample(gt["images"], args.sample_images)
    sample_ids = [int(im["id"]) for im in sample]
    edge = selectors.load_edge_cache(str(args.edge), args.edge_conf)
    if any(iid not in edge for iid in sample_ids):
        raise RuntimeError("edge cache is missing one or more calibration sample images")

    def run_manifest(run_id):
        p = root / "experiments/runs" / run_id / "run_manifest.json"
        if not p.is_file():
            raise FileNotFoundError(f"run manifest not found: {p}")
        return load_json(p)

    a7_manifest = run_manifest(args.a7_run)
    b4_manifest = run_manifest(args.b4_base_run)
    b6_manifest = run_manifest(args.b6_run)
    a7_cfg = dict(a7_manifest["cfg"])
    b4_cfg = dict(b4_manifest["cfg"])
    if a7_cfg.get("mode") != "packed" or a7_cfg.get("select") != "evidence":
        raise ValueError(f"A7 base run is not evidence-packed: {a7_cfg}")
    if b4_cfg.get("mode") != "regional" or not b4_cfg.get("explore"):
        raise ValueError(f"B4 base run is not support+exploration regional: {b4_cfg}")
    b6_summary = load_json(root / "experiments/runs" / args.b6_run / "summary.json")
    b6_target = float(b6_summary["payload_kb_mean"])

    # Byte-only candidates; no AP-based parameter selection.
    a7_ratios = [0.025, 0.0375, 0.05, 0.0625, 0.075, 0.10, 0.125, 0.15]
    b4_qualities = [35, 50, 65, 80, 90, 95]
    rows = []
    a7_rows = []
    for ratio in a7_ratios:
        cfg = dict(a7_cfg)
        cfg["ratio"] = ratio
        sizes = measure(run_a, cfg, sample, args.img_dir, edge, 20261018)
        row = {"method": "A7_support_centers", "ratio": ratio, "q_roi": "",
               "n_images": len(sizes), "mean_packet_bytes": statistics.mean(sizes),
               "mean_packet_kib": statistics.mean(sizes) / 1024,
               "min_packet_bytes": min(sizes), "max_packet_bytes": max(sizes),
               "target_kib": args.a7_target_kib,
               "gap_pct": abs(statistics.mean(sizes) / 1024 - args.a7_target_kib)
                          / args.a7_target_kib * 100}
        rows.append(row)
        a7_rows.append(row)

    b4_rows = []
    for quality in b4_qualities:
        cfg = dict(b4_cfg)
        cfg["q_roi"] = quality
        sizes = measure(run_a, cfg, sample, args.img_dir, edge, 20261018)
        row = {"method": "B4_support_exploration_regional", "ratio": "",
               "q_roi": quality, "n_images": len(sizes),
               "mean_packet_bytes": statistics.mean(sizes),
               "mean_packet_kib": statistics.mean(sizes) / 1024,
               "min_packet_bytes": min(sizes), "max_packet_bytes": max(sizes),
               "target_kib": b6_target,
               "gap_pct": abs(statistics.mean(sizes) / 1024 - b6_target)
                          / b6_target * 100}
        rows.append(row)
        b4_rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        ap.error(f"output file exists; choose a new path: {args.out}")
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best_a7 = min(a7_rows, key=lambda r: r["gap_pct"])
    # Respect the original calibration constraint: choose highest q not above
    # the actual B6 mean. If none fit, identify the closest candidate honestly.
    b4_under = [r for r in b4_rows if r["mean_packet_kib"] <= b6_target]
    best_b4 = max(b4_under, key=lambda r: r["q_roi"]) if b4_under else min(
        b4_rows, key=lambda r: r["gap_pct"])
    meta = {
        "sample_image_ids": sample_ids,
        "sample_ids_sha256": hashlib.sha256(",".join(map(str, sample_ids)).encode()).hexdigest(),
        "n_sample_images": len(sample),
        "sampling": "evenly spaced sorted validation image IDs; bytes only, no labels/AP used",
        "original_c_dev_manifest_recovered": False,
        "a7_base_run": args.a7_run,
        "a7_base_cfg": a7_cfg,
        "a7_target_kib": args.a7_target_kib,
        "a7_recommended_ratio_by_byte_gap": best_a7["ratio"],
        "a7_recommended_mean_kib": best_a7["mean_packet_kib"],
        "b4_base_run": args.b4_base_run,
        "b4_base_cfg": b4_cfg,
        "b6_target_run": args.b6_run,
        "b6_actual_mean_kib": b6_target,
        "b4_recommended_q_roi_highest_not_over_target": best_b4["q_roi"],
        "b4_recommended_mean_kib": best_b4["mean_packet_kib"],
        "note": "Exploratory byte matching only because C_dev IDs are unavailable. Freeze selected settings before looking at full validation AP; if the full-run budget gap exceeds the predeclared tolerance, report the unmatched gap honestly."
    }
    meta_path = args.out.with_suffix(".manifest.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"A7 target={args.a7_target_kib:.3f} KiB -> ratio={best_a7['ratio']} mean={best_a7['mean_packet_kib']:.3f} KiB gap={best_a7['gap_pct']:.2f}%")
    print(f"B6 actual target={b6_target:.3f} KiB -> B4 q_roi={best_b4['q_roi']} mean={best_b4['mean_packet_kib']:.3f} KiB gap={best_b4['gap_pct']:.2f}%")
    print(f"candidate_csv={args.out} manifest={meta_path} sample_images={len(sample)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
