#!/usr/bin/env python3
"""Audit patch candidate/retention counts for the checked-out protocol_v2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

import cv2
import numpy as np


def raw_picks(sel, rgb, rois, ratio: float, mode: str) -> tuple[list[tuple[int, int]], int, list[int]]:
    """Mirror select_patches' budget split, but retain per-ROI picks pre-dedup."""
    grids = [sel.roi_patch_grid(roi, sel.PATCH) for roi in rois]
    totals = [rows * cols for rows, cols in grids]
    grand = sum(totals)
    if grand <= 0:
        return [], 0, totals
    budget = max(1, int(round(ratio * grand)))
    alloc = [max(1, int(round(budget * n / float(grand)))) for n in totals]
    while sum(alloc) > budget and max(alloc) > 1:
        alloc[alloc.index(max(alloc))] -= 1
    while sum(alloc) < budget:
        alloc[alloc.index(min(alloc))] += 1

    gray = np.asarray(rgb).mean(axis=2) if mode == "evidence" else None
    centers = [((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0) for r in rois]
    out: list[tuple[int, int]] = []
    for roi, (rows, cols), keep in zip(rois, grids, alloc):
        if rows <= 0 or cols <= 0:
            continue
        if mode == "uniform":
            cells = sel._uniform_indices_legacy(rows, cols, keep)
        elif mode == "spatial":
            cells = sel._spatial_indices(rows, cols, keep)
        elif mode == "evidence":
            cells = sel._evidence_indices(gray, roi, rows, cols, keep, centers, sel.PATCH)
        else:
            raise ValueError(mode)
        out.extend((int(roi[0] + c * sel.PATCH), int(roi[1] + r * sel.PATCH)) for r, c in cells)
    return out, budget, totals


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=pathlib.Path, required=True)
    ap.add_argument("--gt", type=pathlib.Path, default=pathlib.Path(
        "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"))
    ap.add_argument("--img-dir", type=pathlib.Path, default=pathlib.Path(
        "/app/data/seadronessee/compressed_version/val2017"))
    ap.add_argument("--edge-cache", type=pathlib.Path, default=pathlib.Path(
        "/app/Salience-DETR/experiments/edge_cache/edge_best_conf001.jsonl"))
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--edge-conf", type=float, default=0.01)
    ap.add_argument("--ratio", type=float, default=0.25)
    ap.add_argument("--expand", type=float, default=0.1)
    ap.add_argument("--explore-k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="Audit first N GT images; use 0 for all frames.")
    args = ap.parse_args()
    sys.path.insert(0, str(args.repo_root.resolve()))
    from protocol_v2 import selectors as sel

    with args.gt.open("r", encoding="utf-8") as f:
        gt = json.load(f)
    images = gt["images"][:args.limit] if args.limit else gt["images"]
    edge = sel.load_edge_cache(str(args.edge_cache), args.edge_conf)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "selector_audit.csv"
    modes = ("uniform", "spatial", "evidence")
    scenarios = ("support_only", "support_plus_exploration")
    totals_report = {f"{scenario}:{mode}": {"frames": 0, "raw_selected": 0,
                      "unique_selected": 0, "requested_budget": 0,
                      "duplicate_coordinates": 0, "out_of_bounds": 0,
                      "candidate_patches": 0, "roi_count": 0}
                     for scenario in scenarios for mode in modes}
    fields = ["image_id", "scenario", "mode", "support_rois", "candidate_rois",
              "candidate_patches", "requested_budget", "raw_selected",
              "unique_selected", "duplicate_coordinates", "out_of_bounds",
              "per_roi_candidates", "per_roi_allocations"]
    missing_edge = []
    bad_images = []
    with csv_path.open("w", encoding="utf-8", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fields)
        writer.writeheader()
        for im in images:
            iid = int(im["id"])
            rec = edge.get(iid)
            if rec is None:
                missing_edge.append(iid)
                continue
            path = args.img_dir / im["file_name"]
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                bad_images.append(iid)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            support = sel.support_rois(rec["dets"], w, h, expand=args.expand,
                                       min_ctx=sel.MIN_CTX)
            support = sel.merge_rois(support)
            if not support:
                support = [(0, 0, w, h)]
            explore = sel.exploration_rois(support, w, h, n=args.explore_k, grid=(6, 10))
            candidates = {
                "support_only": list(support),
                "support_plus_exploration": list(support) + list(explore),
            }
            for scenario, rois in candidates.items():
                for mode in modes:
                    raw, budget, per_roi = raw_picks(sel, rgb, rois, args.ratio, mode)
                    selected = sel.select_patches(rgb, rois, args.ratio, mode, ps=sel.PATCH)
                    unique = set(selected)
                    out_of_bounds = sum(x < 0 or y < 0 or x + sel.PATCH > w or y + sel.PATCH > h
                                        for x, y in raw)
                    dups = len(raw) - len(set(raw))
                    key = f"{scenario}:{mode}"
                    agg = totals_report[key]
                    agg["frames"] += 1
                    agg["raw_selected"] += len(raw)
                    agg["unique_selected"] += len(unique)
                    agg["requested_budget"] += budget
                    agg["duplicate_coordinates"] += dups
                    agg["out_of_bounds"] += out_of_bounds
                    agg["candidate_patches"] += sum(per_roi)
                    agg["roi_count"] += len(rois)
                    writer.writerow({
                        "image_id": iid, "scenario": scenario, "mode": mode,
                        "support_rois": len(support), "candidate_rois": len(rois),
                        "candidate_patches": sum(per_roi), "requested_budget": budget,
                        "raw_selected": len(raw), "unique_selected": len(unique),
                        "duplicate_coordinates": dups, "out_of_bounds": out_of_bounds,
                        "per_roi_candidates": json.dumps(per_roi, separators=(",", ":")),
                        "per_roi_allocations": json.dumps({
                            "ratio": args.ratio, "requested_total": budget,
                            "raw_total": len(raw), "unique_total": len(unique)}, separators=(",", ":")),
                    })
    report = {
        "repo_root": str(args.repo_root.resolve()), "gt": str(args.gt),
        "edge_cache": str(args.edge_cache), "n_requested_images": len(images),
        "n_missing_edge_cache": len(missing_edge), "missing_edge_image_ids": missing_edge,
        "n_unreadable_images": len(bad_images), "unreadable_image_ids": bad_images,
        "ratio": args.ratio, "expand": args.expand, "explore_k": args.explore_k,
        "selector_file": str(args.repo_root / "protocol_v2" / "selectors.py"),
        "results": totals_report,
        "note": "Raw/unique counts are implementation diagnostics, not AP results. Compare candidate scenarios before running inference.",
    }
    (args.out_dir / "selector_audit_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if missing_edge or bad_images else 0


if __name__ == "__main__":
    raise SystemExit(main())
