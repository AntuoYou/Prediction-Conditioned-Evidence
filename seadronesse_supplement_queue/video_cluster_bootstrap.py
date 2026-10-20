#!/usr/bin/env python3
"""Paired video-cluster bootstrap for saved SeaDronesSee COCO predictions.

Uses images[*].source.video as the cluster key when available. Images without a
video identifier are kept as singleton clusters; their broader temporal
dependence cannot be recovered from this annotation file. Clones image and
annotation IDs when a cluster is sampled more than once, then computes paired
COCO AP deltas. CPU-only; no detector or NPU service is used.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import csv
import io
import json
import math
import multiprocessing as mp
import pathlib
import statistics
import time

import numpy as np

RUNS = {
    "A2_s5": "experiments/runs/A2_s5",
    "A3_v2": "experiments/runs/A3_v2",
    "A4_s11": "experiments/runs/A4_grid_seed11_v2",
    "A4_s23": "experiments/runs/A4_grid_seed23_v2",
    "A4_s37": "experiments/runs/A4_grid_seed37_v2",
    "A5": "experiments/runs/A5_v3",
    "A5_expand07": "experiments/runs/A5_expand07",
    "A7_legacy": "experiments/runs/A7_v2",
    "A7": "experiments/runs/A7_support_centers_v2",
    "A8": "experiments/runs/A8_support_centers_v2",
    "B3": "experiments/runs/B3_roi87_v2",
    "resize54": "experiments/runs/resize_match_54kb_v2",
    "resize314": "experiments/runs/resize_match_314kb_v2",
    "resize360": "experiments/runs/resize_match_360kb_v2",
}

# name, method members, baseline members, metrics
PAIRS = [
    ("A5-A4_grid_mean", ["A5"], ["A4_s11", "A4_s23", "A4_s37"], ["AP", "AP50", "AP_S"]),
    ("A7-A8_corrected", ["A7"], ["A8"], ["AP", "AP50", "AP_S"]),
    ("A8-B3", ["A8"], ["B3"], ["AP", "AP50", "AP_S"]),
    ("A3-A2_s5", ["A3_v2"], ["A2_s5"], ["AP_S"]),
    ("A5-resize314", ["A5"], ["resize314"], ["AP", "AP50", "AP_S"]),
    ("A5_expand07-resize360", ["A5_expand07"], ["resize360"], ["AP", "AP50", "AP_S"]),
    ("A7_legacy-resize54", ["A7_legacy"], ["resize54"], ["AP", "AP50", "AP_S"]),
]
METRIC_INDEX = {"AP": 0, "AP50": 1, "AP_S": 3}
_WORK = {}


def _coco_eval(gt, categories, detections, new_ids):
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if not detections:
            return {"AP": None, "AP50": None, "AP_S": None}
        dt = gt.loadRes(detections)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = new_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    vals = {"AP": float(ev.stats[0]), "AP50": float(ev.stats[1]), "AP_S": float(ev.stats[3])}
    return {k: (v if math.isfinite(v) and v >= 0 else None) for k, v in vals.items()}


def _replicate(sampled_video_indices):
    images0 = _WORK["images"]
    anns0 = _WORK["annotations_by_image"]
    dets0 = _WORK["detections_by_run_image"]
    categories = _WORK["categories"]
    videos = _WORK["videos"]
    groups = _WORK["groups"]
    next_id = _WORK["next_id"]

    clone_pairs = []
    for video_idx in sampled_video_indices:
        for old_id in groups[videos[video_idx]]:
            clone_pairs.append((old_id, next_id + len(clone_pairs)))
    new_ids = [new_id for _, new_id in clone_pairs]
    new_images, new_anns = [], []
    cloned_dets = {name: [] for name in dets0}
    ann_id = 1
    for old_id, new_id in clone_pairs:
        im = dict(images0[old_id])
        im["id"] = new_id
        new_images.append(im)
        for ann in anns0.get(old_id, []):
            item = dict(ann)
            item["id"] = ann_id
            item["image_id"] = new_id
            ann_id += 1
            new_anns.append(item)
        for run_name, by_image in dets0.items():
            for det in by_image.get(old_id, []):
                item = dict(det)
                item["image_id"] = new_id
                cloned_dets[run_name].append(item)

    from pycocotools.coco import COCO
    gt = COCO()
    gt.dataset = {"images": new_images, "annotations": new_anns,
                  "categories": categories}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        gt.createIndex()

    metrics = {}
    for run_name, detections in cloned_dets.items():
        metrics[run_name] = _coco_eval(gt, categories, detections, new_ids)
    deltas = {}
    for pair_name, left, right, pair_metrics in PAIRS:
        for metric in pair_metrics:
            lv = [metrics[n][metric] for n in left]
            rv = [metrics[n][metric] for n in right]
            deltas[f"{pair_name}|{metric}"] = (statistics.mean(lv) - statistics.mean(rv)
                                                if all(x is not None for x in lv + rv) else None)
    return deltas


def _load_inputs(root, gt_path, score_floor):
    gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
    images = {int(im["id"]): im for im in gt_data["images"]}
    if len(images) != len(gt_data["images"]):
        raise ValueError("duplicate GT image IDs")
    unclustered_singletons = 0
    groups = {}
    for iid, im in images.items():
        src = im.get("source") or {}
        video = src.get("video")
        if video:
            # Namespace the filename by dataset source to avoid accidental
            # merging if two folders happen to reuse a video basename.
            key = "video::{}::{}::{}".format(
                src.get("drone", ""), src.get("folder_name", ""), video)
        else:
            # folder_name is only "rgb"/"multispectral" for the unlinked
            # still images, so it is not a valid cluster identifier.
            key = "singleton_image::{}".format(iid)
            unclustered_singletons += 1
        groups.setdefault(key, []).append(iid)
    for ids in groups.values():
        ids.sort(key=lambda i: (int(images[i].get("source", {}).get("frame_no", i)), i))

    annotations_by_image = {iid: [] for iid in images}
    for ann in gt_data["annotations"]:
        annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)
    detections_by_run_image = {}
    summaries = {}
    for run_name, rel in RUNS.items():
        run_dir = root / rel
        pred_path = run_dir / "predictions_coco.json"
        summary_path = run_dir / "summary.json"
        if not pred_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"required predictions/summary missing for {run_name}: {run_dir}")
        raw = json.loads(pred_path.read_text(encoding="utf-8"))
        by_image = {}
        for det in raw:
            if float(det.get("score", 0)) < score_floor:
                continue
            by_image.setdefault(int(det["image_id"]), []).append(det)
        detections_by_run_image[run_name] = by_image
        summaries[run_name] = json.loads(summary_path.read_text(encoding="utf-8"))
    return (gt_data, images, groups, annotations_by_image,
            detections_by_run_image, summaries, unclustered_singletons)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path("/app/Salience-DETR"))
    ap.add_argument("--gt", type=pathlib.Path, default=pathlib.Path("/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"))
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("experiments/supplement_audit/final_bootstrap"))
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--score-floor", type=float, default=0.05)
    args = ap.parse_args()
    if args.reps < 1 or args.workers < 1:
        ap.error("--reps and --workers must be positive")
    root, gt_path = args.repo_root.resolve(), args.gt.resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        ap.error(f"output directory is not empty; choose a new directory: {out_dir}")
    (gt_data, images, groups, annotations_by_image, dets, summaries,
     unclustered_singletons) = _load_inputs(root, gt_path, args.score_floor)
    videos = sorted(groups)
    if len(images) != 1547:
        raise ValueError(f"expected 1547 GT images, got {len(images)}")
    video_clusters = len(videos) - unclustered_singletons
    print(f"clusters={len(videos)} video_clusters={video_clusters} singleton_images={unclustered_singletons} frames={len(images)} score_floor={args.score_floor} reps={args.reps} workers={args.workers}", flush=True)
    sizes = [len(groups[v]) for v in videos]
    print(f"cluster_frames min/median/max={min(sizes)}/{statistics.median(sizes)}/{max(sizes)}", flush=True)

    # Original point estimates come from full-split run summaries; A4 is the
    # arithmetic mean of its three preregistered seeds, never the best seed.
    point = {}
    for pair_name, left, right, metrics in PAIRS:
        for metric in metrics:
            def val(run_name):
                key = {"AP": "AP", "AP50": "AP50", "AP_S": "AP_S"}[metric]
                return float(summaries[run_name][key])
            point[f"{pair_name}|{metric}"] = statistics.mean(val(n) for n in left) - statistics.mean(val(n) for n in right)

    global _WORK
    _WORK = {"images": images,
             "annotations_by_image": annotations_by_image,
             "categories": gt_data["categories"], "groups": groups, "videos": videos,
             "detections_by_run_image": dets,
             "next_id": max(images) + 1}

    rng = np.random.RandomState(args.seed)
    draws = [rng.randint(0, len(videos), size=len(videos)).tolist() for _ in range(args.reps)]
    t0 = time.time()
    deltas = {f"{name}|{metric}": [] for name, _, _, metrics in PAIRS for metric in metrics}
    ctx = mp.get_context("fork")
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = [pool.submit(_replicate, draw) for draw in draws]
        for i, future in enumerate(cf.as_completed(futures), 1):
            values = future.result()
            for key, value in values.items():
                if value is not None and math.isfinite(value):
                    deltas[key].append(float(value))
            if i % 10 == 0 or i == args.reps:
                elapsed = time.time() - t0
                eta = elapsed / i * (args.reps - i)
                print(f"[{i}/{args.reps}] elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)

    with (out_dir / "video_cluster_bootstrap_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["comparison", "metric", "point_delta", "bootstrap_mean", "ci_low", "ci_high", "resampling_unit", "n_clusters", "requested_replicates", "valid_replicates", "seed", "score_floor"])
        writer.writeheader()
        for key, vals in deltas.items():
            name, metric = key.split("|", 1)
            if vals:
                lo, hi = np.percentile(vals, [2.5, 97.5])
                mean = statistics.mean(vals)
            else:
                lo = hi = mean = float("nan")
            writer.writerow({"comparison": name, "metric": metric, "point_delta": point[key],
                             "bootstrap_mean": mean, "ci_low": lo, "ci_high": hi,
                             "resampling_unit": "video", "n_clusters": len(videos),
                             "requested_replicates": args.reps, "valid_replicates": len(vals),
                             "seed": args.seed, "score_floor": args.score_floor})

    meta = {"gt": str(gt_path), "n_images": len(images), "n_videos": len(videos),
            "cluster_policy": "source.video grouped by drone/folder/video; images without source.video are singleton clusters because folder_name is generic",
            "n_video_clusters": video_clusters,
            "n_singleton_images_without_video_id": unclustered_singletons,
            "cluster_sizes": {"min": min(sizes), "median": statistics.median(sizes), "max": max(sizes)},
            "reps": args.reps, "seed": args.seed, "score_floor": args.score_floor,
            "workers": args.workers, "run_paths": RUNS,
            "note": "Paired cluster bootstrap. source.video groups are resampled together; images lacking video metadata are singleton units, so their within-video dependence is not modeled. Repeatedly sampled frames receive cloned image and annotation IDs. A4 is the arithmetic mean of seeds 11/23/37; intervals condition on these three seeds and do not estimate seed-to-seed variance."}
    (out_dir / "bootstrap_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"DONE summary={out_dir / 'video_cluster_bootstrap_summary.csv'} manifest={out_dir / 'bootstrap_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
