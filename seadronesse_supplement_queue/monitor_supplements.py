#!/usr/bin/env python3
"""One-shot or periodic status and acceptance monitor for current SeaDronesSee jobs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def process_matches(needles: tuple[str, ...]) -> list[int]:
    """Find matching Linux processes by /proc cmdline; never signals processes."""
    found: list[int] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return found
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except OSError:
            continue
        if all(needle in command for needle in needles):
            found.append(int(entry.name))
    return found


def last_progress(log_path: Path) -> tuple[int, int, str] | None:
    if not log_path.is_file():
        return None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        match = re.search(r"\[(\d+)\s*/\s*(\d+)\].*(?:ETA\s+([^\s]+))?", line)
        if match:
            return int(match.group(1)), int(match.group(2)), match.group(3) or ""
    return None


def monitor_b3(root: Path) -> str:
    run_id = "B3_roi87_v2"
    run = root / "experiments/runs" / run_id
    log = root / "experiments/supplement_audit/next/roi/B3_full.log"
    summary = read_json(run / "summary.json")
    per_image = run / "per_image.csv"
    pids = process_matches(("protocol_v2.run_a", "--run-id", run_id))

    if summary is not None and per_image.is_file():
        with per_image.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        ids = [row.get("image_id", "") for row in rows]
        ok = sum(row.get("status", "").lower() == "ok" for row in rows)
        failed = len(rows) - ok
        byte_mismatches = "n/a"
        index_path = run / "packet_index.csv"
        if index_path.is_file():
            with index_path.open("r", encoding="utf-8-sig", newline="") as stream:
                index = {row.get("image_id", ""): row for row in csv.DictReader(stream)}
            byte_mismatches = sum(
                row.get("image_id") not in index
                or not row.get("packet_bytes", "").isdigit()
                or not index[row["image_id"]].get("total", "").isdigit()
                or int(row["packet_bytes"]) != int(index[row["image_id"]]["total"])
                for row in rows
            )
        passed = len(rows) == 1547 and len(set(ids)) == 1547 and ok == 1547 and failed == 0
        if byte_mismatches != "n/a":
            passed = passed and byte_mismatches == 0
        return (
            f"B3 87KB: {'PASS' if passed else 'CHECK'} | rows={len(rows)} "
            f"unique={len(set(ids))} ok={ok} failed={failed} "
            f"byte_mismatch={byte_mismatches} AP={summary.get('AP')} "
            f"meanKB={summary.get('payload_kb_mean')}"
        )

    progress = last_progress(log)
    if pids or progress:
        if progress:
            done, total, eta = progress
            pct = 100.0 * done / total if total else 0.0
            eta_text = f", ETA {eta}" if eta else ""
            return f"B3 87KB: RUNNING {done}/{total} ({pct:.1f}%){eta_text}; pid={pids or 'unknown'}"
        return f"B3 87KB: RUNNING; pid={pids}"
    return "B3 87KB: NOT STARTED or stopped without complete run outputs"


def monitor_b6(root: Path) -> str:
    run_id = "B6_roi_max_allowed_q95_unmatched_v2"
    run = root / "experiments/runs" / run_id
    log = root / "experiments/supplement_audit/next/roi/B6_full.log"
    summary = read_json(run / "summary.json")
    per_image = run / "per_image.csv"
    pids = process_matches(("protocol_v2.run_a", "--run-id", run_id))

    if summary is not None and per_image.is_file():
        with per_image.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        ids = [row.get("image_id", "") for row in rows]
        ok = sum(row.get("status", "").lower() == "ok" for row in rows)
        failed = len(rows) - ok
        byte_mismatches = "n/a"
        index_path = run / "packet_index.csv"
        if index_path.is_file():
            with index_path.open("r", encoding="utf-8-sig", newline="") as stream:
                index = {row.get("image_id", ""): row for row in csv.DictReader(stream)}
            byte_mismatches = sum(
                row.get("image_id") not in index
                or not row.get("packet_bytes", "").isdigit()
                or not index[row["image_id"]].get("total", "").isdigit()
                or int(row["packet_bytes"]) != int(index[row["image_id"]]["total"])
                for row in rows
            )
        passed = len(rows) == 1547 and len(set(ids)) == 1547 and ok == 1547 and failed == 0
        if byte_mismatches != "n/a":
            passed = passed and byte_mismatches == 0
        return (
            f"B6 unmatched: {'PASS' if passed else 'CHECK'} | rows={len(rows)} "
            f"unique={len(set(ids))} ok={ok} failed={failed} "
            f"byte_mismatch={byte_mismatches} AP={summary.get('AP')} "
            f"meanKB={summary.get('payload_kb_mean')} (not a 311KB match)"
        )

    progress = last_progress(log)
    if pids or progress:
        if progress:
            done, total, eta = progress
            pct = 100.0 * done / total if total else 0.0
            eta_text = f", ETA {eta}" if eta else ""
            return f"B6 unmatched: RUNNING {done}/{total} ({pct:.1f}%){eta_text}; pid={pids or 'unknown'}"
        return f"B6 unmatched: RUNNING; pid={pids}"
    return "B6 unmatched: NOT STARTED or stopped without complete run outputs"


def monitor_selector(root: Path) -> str:
    out = root / "experiments/supplement_audit/next/selector_reaudit"
    csv_path = out / "selector_audit.csv"
    summary_path = out / "selector_audit_summary.json"
    pids = process_matches(("audit_selectors.py", "selector_reaudit"))
    rows = 0
    if csv_path.is_file():
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = sum(1 for _ in csv.reader(stream))
        except OSError:
            pass
    expected = 1547 * 2 * 3 + 1  # 2 candidate scopes × 3 selectors × 1,547 images + header
    summary = read_json(summary_path)

    if summary is not None and not pids:
        results = summary.get("results", {})
        return (
            f"Selector re-audit: COMPLETE | rows={rows}/{expected} "
            f"requested_images={summary.get('n_requested_images')} "
            f"scenarios={len(results)} summary={summary_path}"
        )
    if pids:
        pct = 100.0 * rows / expected if expected else 0.0
        return f"Selector re-audit: RUNNING | rows={rows}/{expected} ({pct:.1f}%) pid={pids}"
    if summary is not None:
        return f"Selector re-audit: SUMMARY EXISTS, process not found | rows={rows}/{expected}"
    return f"Selector re-audit: INCOMPLETE | rows={rows}/{expected}; no matching process or summary"


def monitor_offline_checks(root: Path) -> tuple[str, str, bool]:
    eval_dir = root / "experiments/supplement_audit/next/eval_B3_independent"
    eval_summary_path = eval_dir / "eval_summary.json"
    eval_summary = read_json(eval_summary_path)
    eval_pids = process_matches(("evaluate_predictions.py", "B3_roi87_v2/predictions_coco.json"))
    if eval_summary is not None:
        eval_line = (
            f"B3 independent COCO eval: COMPLETE | AP={eval_summary.get('AP')} "
            f"AP_S={eval_summary.get('AP_S')} n_images={eval_summary.get('n_images')}"
        )
    elif eval_pids:
        eval_line = f"B3 independent COCO eval: RUNNING | pid={eval_pids}"
    else:
        eval_line = "B3 independent COCO eval: NOT STARTED"

    audit_path = root / "experiments/supplement_audit/next/acceptance/run_audit_latest.json"
    audit = read_json(audit_path)
    audit_pids = process_matches(("audit_run_outputs.py", "run_audit_latest.json"))
    if audit is not None:
        review = [
            item.get("run_id", "unknown")
            for item in audit.get("runs", [])
            if item.get("status") != "structurally_consistent"
        ]
        label = "PASS" if not audit.get("n_needs_review") else "REPORT READY / REVIEW REQUIRED"
        audit_line = (
            f"Historical run audit: {label} | runs={audit.get('n_runs')} "
            f"consistent={audit.get('n_consistent')} needs_review={audit.get('n_needs_review')} "
            f"ids={review}"
        )
    elif audit_pids:
        audit_line = f"Historical run audit: RUNNING | pid={audit_pids}"
    else:
        audit_line = "Historical run audit: NOT STARTED"
    return eval_line, audit_line, bool(eval_pids or audit_pids)


def report(root: Path) -> bool:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"===== SeaDronesSee supplement monitor {now} =====")
    print(monitor_b3(root))
    print(monitor_b6(root))
    print(monitor_selector(root))
    eval_line, audit_line, active_offline = monitor_offline_checks(root)
    print(eval_line)
    print(audit_line)
    active_b3 = bool(process_matches(("protocol_v2.run_a", "--run-id", "B3_roi87_v2")))
    active_b6 = bool(process_matches(("protocol_v2.run_a", "--run-id", "B6_roi_max_allowed_q95_unmatched_v2")))
    active_selector = bool(process_matches(("audit_selectors.py", "selector_reaudit")))
    return active_b3 or active_b6 or active_selector or active_offline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default="/app/Salience-DETR")
    parser.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                        help="Repeat every N seconds; 0 prints one status snapshot.")
    args = parser.parse_args()
    root = Path(args.repo_root).expanduser().resolve()
    if args.watch <= 0:
        report(root)
        return 0
    try:
        while True:
            pending = report(root)
            if not pending:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
