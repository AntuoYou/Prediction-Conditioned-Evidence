#!/usr/bin/env python3
"""Queue CPU-only COCO reevaluations of saved supplement predictions.

Runs each available prediction set at score floors 0.05 and 0.001. No NPU
service is called. Existing evaluation outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import pathlib
import subprocess
import sys


RUNS = [
    ("A0_edge_only", "experiments/supplement_audit/parallel/edge_only/edge_only_predictions.json"),
    ("A2_s5", "experiments/runs/A2_s5/predictions_coco.json"),
    ("A3_v2", "experiments/runs/A3_v2/predictions_coco.json"),
    ("A4_grid_seed11_v2", "experiments/runs/A4_grid_seed11_v2/predictions_coco.json"),
    ("A4_grid_seed23_v2", "experiments/runs/A4_grid_seed23_v2/predictions_coco.json"),
    ("A4_grid_seed37_v2", "experiments/runs/A4_grid_seed37_v2/predictions_coco.json"),
    ("A5_v3", "experiments/runs/A5_v3/predictions_coco.json"),
    ("A5_expand07", "experiments/runs/A5_expand07/predictions_coco.json"),
    ("A7_legacy", "experiments/runs/A7_v2/predictions_coco.json"),
    ("A7_support_centers_r25", "experiments/runs/A7_support_centers_v2/predictions_coco.json"),
    ("A7_support_centers_r40", "experiments/runs/A7_support_centers_r40_v2/predictions_coco.json"),
    ("A7_support_centers_r50", "experiments/runs/A7_support_centers_r50_v2/predictions_coco.json"),
    ("A7_support_centers_r100", "experiments/runs/A7_support_centers_r100_v2/predictions_coco.json"),
    ("A8_support_centers", "experiments/runs/A8_support_centers_v2/predictions_coco.json"),
    ("B3_roi87", "experiments/runs/B3_roi87_v2/predictions_coco.json"),
    ("B6_roi_unmatched", "experiments/runs/B6_roi_max_allowed_q95_unmatched_v2/predictions_coco.json"),
    ("resize54", "experiments/runs/resize_match_54kb_v2/predictions_coco.json"),
    ("resize314", "experiments/runs/resize_match_314kb_v2/predictions_coco.json"),
    ("resize360", "experiments/runs/resize_match_360kb_v2/predictions_coco.json"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path("/app/Salience-DETR"))
    ap.add_argument("--gt", type=pathlib.Path, default=pathlib.Path("/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"))
    ap.add_argument("--out-root", type=pathlib.Path, default=pathlib.Path("experiments/supplement_audit/final_eval"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    root = args.repo_root.resolve()
    if args.workers < 1:
        ap.error("--workers must be >= 1")
    if not args.gt.is_file():
        ap.error(f"GT not found: {args.gt}")
    evaluator = root / "seadronesse_supplement_queue/evaluate_predictions.py"
    if not evaluator.is_file():
        ap.error(f"evaluator not found: {evaluator}")
    tasks = []
    skipped = []
    for run_id, relpred in RUNS:
        pred = root / relpred
        if not pred.is_file():
            skipped.append({"run_id": run_id, "status": "missing_prediction", "path": str(pred)})
            continue
        for floor in (0.05, 0.001):
            tag = f"{run_id}_score_{str(floor).replace('.', '')}"
            out = root / args.out_root / tag
            summary = out / "eval_summary.json"
            if summary.is_file():
                skipped.append({"run_id": run_id, "score_floor": floor, "status": "already_evaluated", "output": str(out)})
                continue
            if out.exists() and any(out.iterdir()):
                skipped.append({"run_id": run_id, "score_floor": floor, "status": "output_exists_needs_review", "output": str(out)})
                continue
            out.mkdir(parents=True, exist_ok=True)
            tasks.append((run_id, floor, pred, out))

    def evaluate(task):
        run_id, floor, pred, out = task
        log = out / "eval.log"
        cmd = [sys.executable, "-u", str(evaluator), "--gt", str(args.gt.resolve()),
               "--predictions", str(pred), "--out-dir", str(out),
               "--score-floor", str(floor)]
        with log.open("w", encoding="utf-8") as stream:
            proc = subprocess.run(cmd, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=False)
        return {"run_id": run_id, "score_floor": floor,
                "status": "completed" if proc.returncode == 0 and (out / "eval_summary.json").is_file() else "failed",
                "return_code": proc.returncode, "output": str(out), "log": str(log)}

    results = []
    print(f"Starting {len(tasks)} CPU evaluation tasks with workers={args.workers}; no NPU inference.", flush=True)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate, task): task for task in tasks}
        for future in cf.as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{result['status'].upper()} {result['run_id']} score_floor={result['score_floor']}", flush=True)

    report = {"created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
              "repo_root": str(root), "queued": len(tasks), "results": results, "skipped": skipped}
    report_path = root / args.out_root / "evaluation_queue_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failed = [r for r in results if r["status"] != "completed"]
    print(f"QUEUE_DONE completed={len(results)-len(failed)} failed={len(failed)} skipped={len(skipped)} report={report_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
