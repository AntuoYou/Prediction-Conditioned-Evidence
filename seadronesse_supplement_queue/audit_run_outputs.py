#!/usr/bin/env python3
"""Audit protocol-v2 run folders without rerunning inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics
import sys


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def number(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def percentile(values: list[int], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def audit_run(run_dir: pathlib.Path, check_packet_files: bool) -> dict:
    issues: list[str] = []
    per_image = run_dir / "per_image.csv"
    summary_file = run_dir / "summary.json"
    manifest_file = run_dir / "run_manifest.json"
    if not per_image.is_file():
        return {"run_id": run_dir.name, "status": "incomplete",
                "issues": ["missing per_image.csv"]}

    with per_image.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        issues.append("per_image.csv has no data rows")
    fields = set(rows[0]) if rows else set()
    for field in ("image_id", "status", "packet_bytes"):
        if field not in fields:
            issues.append(f"per_image.csv missing required column: {field}")

    ids = [row.get("image_id", "") for row in rows]
    nonempty_ids = [image_id for image_id in ids if image_id]
    duplicates = len(nonempty_ids) - len(set(nonempty_ids))
    if duplicates:
        issues.append(f"duplicate image_id rows: {duplicates}")
    ok_rows = [row for row in rows if row.get("status", "").lower() in {"ok", "success", "completed"}]
    failed_rows = [row for row in rows if row not in ok_rows]
    sizes = [n for row in ok_rows if (n := number(row.get("packet_bytes"))) is not None]
    if len(sizes) != len(ok_rows):
        issues.append(f"packet_bytes missing/invalid for {len(ok_rows)-len(sizes)} successful rows")

    packet_file_checks = 0
    packet_hash_checks = 0
    if check_packet_files and "packet_path" in fields:
        for row in ok_rows:
            raw_path = row.get("packet_path", "")
            if not raw_path:
                continue
            candidate = pathlib.Path(raw_path)
            packet_path = candidate if candidate.is_absolute() else run_dir / candidate
            if not packet_path.is_file():
                issues.append(f"packet file missing: {raw_path}")
                continue
            packet_file_checks += 1
            actual_bytes = packet_path.stat().st_size
            expected_bytes = number(row.get("packet_bytes"))
            if expected_bytes is not None and actual_bytes != expected_bytes:
                issues.append(f"packet byte mismatch for {raw_path}: csv={expected_bytes} file={actual_bytes}")
            expected_sha = row.get("packet_sha256", "").strip().lower()
            if expected_sha:
                packet_hash_checks += 1
                if sha256(packet_path) != expected_sha:
                    issues.append(f"packet SHA256 mismatch: {raw_path}")

    summary: dict = {}
    if summary_file.is_file():
        try:
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            issues.append(f"summary.json unreadable: {exc}")
    else:
        issues.append("missing summary.json")

    checks = (("n_ok", len(ok_rows)), ("n_failed", len(failed_rows)))
    for key, observed in checks:
        if key in summary and summary[key] != observed:
            issues.append(f"summary {key}={summary[key]} but per_image rows={observed}")
    if "payload_kb_mean" in summary and sizes:
        observed_kb = statistics.mean(sizes) / 1024
        if abs(float(summary["payload_kb_mean"]) - observed_kb) > 0.01:
            issues.append(f"summary payload_kb_mean={summary['payload_kb_mean']} but per_image={observed_kb:.4f}")

    manifest_present = manifest_file.is_file()
    if not manifest_present:
        issues.append("missing run_manifest.json")
    else:
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if manifest.get("run_id") not in (None, run_dir.name):
                issues.append(f"manifest run_id does not match folder ({manifest.get('run_id')})")
        except (json.JSONDecodeError, OSError) as exc:
            issues.append(f"run_manifest.json unreadable: {exc}")

    predictions_present = any((run_dir / name).is_file() for name in
                              ("predictions_coco.json", "predictions.json", "predictions_raw.json"))
    if not predictions_present:
        issues.append("no prediction JSON found")

    index_file = run_dir / "packet_index.csv"
    index_checks = 0
    if index_file.is_file():
        with index_file.open("r", encoding="utf-8-sig", newline="") as f:
            index_rows = list(csv.DictReader(f))
        index_ids = [row.get("image_id", "") for row in index_rows]
        if len(index_ids) != len(set(index_ids)):
            issues.append("packet_index.csv has duplicate image_id rows")
        successful_bytes = {row.get("image_id"): number(row.get("packet_bytes"))
                            for row in ok_rows}
        for row in index_rows:
            total = number(row.get("total"))
            payload = number(row.get("payload"))
            overhead = number(row.get("overhead"))
            if total is None or payload is None or overhead is None:
                issues.append(f"packet_index row has invalid byte fields for image_id={row.get('image_id')}")
                continue
            index_checks += 1
            if payload + overhead != total:
                issues.append(f"packet_index bytes do not close for image_id={row.get('image_id')}")
            frame_total = successful_bytes.get(row.get("image_id"))
            if frame_total is not None and total != frame_total:
                issues.append(f"packet_index/per_image byte mismatch for image_id={row.get('image_id')}")
        if len(index_rows) != len(ok_rows):
            issues.append(f"packet_index rows={len(index_rows)} but successful per_image rows={len(ok_rows)}")
    else:
        issues.append("missing packet_index.csv (cannot reconcile payload/metadata/container bytes)")

    if failed_rows:
        issues.append(f"run contains {len(failed_rows)} failed/non-success image rows")
    if "AP" not in summary:
        issues.append("summary has no COCO AP; run is incomplete or evaluation failed")

    return {
        "run_id": run_dir.name,
        "status": "needs_review" if issues else "structurally_consistent",
        "n_rows": len(rows), "n_ok_rows": len(ok_rows), "n_failed_rows": len(failed_rows),
        "n_packet_sizes": len(sizes),
        "packet_bytes_mean": statistics.mean(sizes) if sizes else None,
        "packet_bytes_median": percentile(sizes, 0.5),
        "packet_bytes_p90": percentile(sizes, 0.9),
        "packet_bytes_p95": percentile(sizes, 0.95),
        "packet_files_checked": packet_file_checks,
        "packet_hashes_checked": packet_hash_checks,
        "packet_index_rows_checked": index_checks,
        "manifest_present": manifest_present,
        "predictions_present": predictions_present,
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True,
                        help="Runs root, e.g. experiments/runs")
    parser.add_argument("--output", type=pathlib.Path,
                        help="Optional JSON audit report path")
    parser.add_argument("--check-packet-files", action="store_true",
                        help="Hash and compare saved packet files; may take time")
    args = parser.parse_args()
    run_dirs = sorted(p.parent for p in args.root.glob("*/per_image.csv"))
    if not run_dirs:
        print(f"No per_image.csv files under {args.root}", file=sys.stderr)
        return 2
    results = [audit_run(run_dir, args.check_packet_files) for run_dir in run_dirs]
    report = {"runs_root": str(args.root.resolve()), "runs": results,
              "n_runs": len(results),
              "n_consistent": sum(r["status"] == "structurally_consistent" for r in results),
              "n_needs_review": sum(r["status"] != "structurally_consistent" for r in results)}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 1 if report["n_needs_review"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
