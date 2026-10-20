#!/usr/bin/env python3
"""One-command structural and pairwise acceptance report for the current runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys

RUN_IDS = [
    "A2_s5", "A3_v2", "A4_grid_seed11_v2", "A4_grid_seed23_v2", "A4_grid_seed37_v2",
    "A5_v3", "A5_expand07", "A7_v2", "A7_support_centers_v2",
    "A7_support_centers_r40_v2", "A7_support_centers_r50_v2", "A7_support_centers_r100_v2",
    "A8_support_centers_v2", "B3_roi87_v2", "B6_roi_max_allowed_q95_unmatched_v2",
    "resize_match_54kb_v2", "resize_match_314kb_v2", "resize_match_360kb_v2",
]


def load_run(root: pathlib.Path, run_id: str) -> dict:
    d = root / run_id
    result = {"run_id": run_id, "status": "MISSING", "issues": []}
    required = [d / "run_manifest.json", d / "summary.json", d / "per_image.csv",
                d / "packet_index.csv", d / "predictions_coco.json"]
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        result["issues"] = ["missing " + ", ".join(missing)]
        return result
    try:
        manifest = json.loads(required[0].read_text(encoding="utf-8"))
        summary = json.loads(required[1].read_text(encoding="utf-8"))
        with required[2].open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        with required[3].open(encoding="utf-8-sig", newline="") as f:
            index = list(csv.DictReader(f))
    except Exception as exc:
        result["issues"] = [f"read error: {exc}"]
        return result
    issues = result["issues"]
    ids = [r.get("image_id", "") for r in rows]
    idx = {r.get("image_id", ""): r for r in index}
    if manifest.get("run_id") not in (None, run_id):
        issues.append(f"manifest run_id={manifest.get('run_id')}")
    if len(rows) != 1547 or len(set(ids)) != 1547:
        issues.append(f"per_image rows/unique={len(rows)}/{len(set(ids))}, expected 1547/1547")
    failed = [r for r in rows if r.get("status", "").lower() not in ("ok", "success", "completed")]
    if failed:
        issues.append(f"failed rows={len(failed)}")
    if len(index) != 1547 or len(idx) != 1547:
        issues.append(f"packet_index rows/unique={len(index)}/{len(idx)}, expected 1547/1547")
    mismatch, badsum = 0, 0
    for r in rows:
        pr = idx.get(r.get("image_id", ""))
        try:
            if pr is None or int(pr["total"]) != int(r["packet_bytes"]):
                mismatch += 1
            if pr is not None and int(pr["total"]) != int(pr["payload"]) + int(pr["overhead"]):
                badsum += 1
        except (KeyError, TypeError, ValueError):
            mismatch += 1
    if mismatch:
        issues.append(f"packet/per_image mismatch={mismatch}")
    if badsum:
        issues.append(f"total != payload+overhead rows={badsum}")
    if summary.get("n_ok") != 1547 or summary.get("n_failed") != 0:
        issues.append(f"summary n_ok/n_failed={summary.get('n_ok')}/{summary.get('n_failed')}")
    if "AP" not in summary:
        issues.append("summary AP missing")
    result.update({"status": "PASS" if not issues else "CHECK", "manifest": manifest,
                   "summary": summary, "n_rows": len(rows), "n_failed": len(failed),
                   "packet_rows": len(index), "byte_mismatch": mismatch, "bad_sums": badsum})
    return result


def pair(name: str, left: dict, right: dict) -> dict:
    if not left or not right or not left.get("summary") or not right.get("summary"):
        return {"comparison": name, "status": "MISSING_RUN"}
    a, b = left["summary"], right["summary"]
    ak, bk = float(a.get("payload_kb_mean", float("nan"))), float(b.get("payload_kb_mean", float("nan")))
    return {"comparison": name, "status": "READY",
            "left_kib": ak, "right_kib": bk,
            "budget_gap_pct_vs_right": abs(ak-bk)/bk*100 if bk else None,
            "delta_AP_pp": (float(a.get("AP", 0))-float(b.get("AP", 0)))*100,
            "delta_AP50_pp": (float(a.get("AP50", 0))-float(b.get("AP50", 0)))*100,
            "delta_AP_S_pp": (float(a.get("AP_S", 0))-float(b.get("AP_S", 0)))*100}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path("/app/Salience-DETR"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("experiments/supplement_audit/final_acceptance"))
    args = ap.parse_args()
    root = args.repo_root.resolve() / "experiments/runs"
    runs = {rid: load_run(root, rid) for rid in RUN_IDS}

    # A4 is summarized across the three predeclared seeds; never select a best seed.
    seed_ids = ["A4_grid_seed11_v2", "A4_grid_seed23_v2", "A4_grid_seed37_v2"]
    a4 = {}
    seed_runs = [runs[x] for x in seed_ids]
    if all(r.get("summary") for r in seed_runs):
        keys = ("AP", "AP50", "AP_S", "AP_M", "AP_L", "payload_kb_mean")
        a4 = {"run_id": "A4_three_seed_mean", "status": "PASS" if all(r["status"] == "PASS" for r in seed_runs) else "CHECK",
              "summary": {k: statistics.mean(float(r["summary"][k]) for r in seed_runs) for k in keys}}

    pairs = [
        pair("A8 corrected vs B3 simple ROI", runs["A8_support_centers_v2"], runs["B3_roi87_v2"]),
        pair("A7 corrected r25 vs A2_s5 resize", runs["A7_support_centers_v2"], runs["A2_s5"]),
        pair("A3 support-only vs A2_s5 resize", runs["A3_v2"], runs["A2_s5"]),
        pair("A5 vs resize314", runs["A5_v3"], runs["resize_match_314kb_v2"]),
        pair("A5 expand07 vs resize360", runs["A5_expand07"], runs["resize_match_360kb_v2"]),
        pair("legacy A7 vs resize54", runs["A7_v2"], runs["resize_match_54kb_v2"]),
        pair("A5 vs A4 three-seed mean", runs["A5_v3"], a4),
    ]
    conditionals = [
        {"item": "corrected A7 at ~54 KiB", "status": "CONDITIONAL_NOT_RUN", "reason": "Only needed if the paper claims the corrected A7 point at the 54 KiB operating point."},
        {"item": "B4 paired with B6 at ~169 KiB", "status": "PROTOCOL_CONFLICT", "reason": "The written plan defines B4/B5/B6 at 311 KiB; 169 KiB would be a revised, post-hoc operating point and must be frozen before scheduling."},
        {"item": "A3 at ~299 KiB", "status": "DROPPED_BY_SCOPE", "reason": "Do not run unless retaining an equal-budget A3-vs-A4 claim; the protocol does not require it."},
    ]
    report = {"repo_root": str(args.repo_root.resolve()), "expected_runs": len(RUN_IDS),
              "pass_runs": sum(r["status"] == "PASS" for r in runs.values()),
              "check_or_missing_runs": sum(r["status"] != "PASS" for r in runs.values()),
              "runs": list(runs.values()), "A4_three_seed_mean": a4 or None,
              "pairwise": pairs, "conditional_inference": conditionals,
              "note": "This is a structural/pairwise acceptance report. It does not compute bootstrap intervals; run those as a separate CPU statistical step."}
    outdir = args.repo_root.resolve() / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "acceptance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# SeaDronesSee 补充实验一键验收", "", f"通过运行：{report['pass_runs']}/{report['expected_runs']}", "", "## 运行文件验收", ""]
    for r in report["runs"]:
        s = r.get("summary", {})
        lines.append(f"- `{r['run_id']}`：{r['status']}；AP={s.get('AP')}；均值 KiB={s.get('payload_kb_mean')}；问题={'; '.join(r.get('issues', [])) or '无'}")
    lines += ["", "## 配对对照（左侧减右侧；差值单位为百分点）", ""]
    for p in pairs:
        if p["status"] != "READY":
            lines.append(f"- {p['comparison']}：{p['status']}")
        else:
            lines.append(f"- {p['comparison']}：预算偏差={p['budget_gap_pct_vs_right']:.2f}%；ΔAP={p['delta_AP_pp']:.2f}；ΔAP50={p['delta_AP50_pp']:.2f}；ΔAP_S={p['delta_AP_S_pp']:.2f}")
    lines += ["", "## 条件性推理", ""]
    for x in conditionals:
        lines.append(f"- {x['item']}：{x['status']}；{x['reason']}")
    lines += ["", report["note"], ""]
    (outdir / "acceptance.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"acceptance: {outdir / 'acceptance.md'}")
    print(f"structurally PASS: {report['pass_runs']}/{report['expected_runs']}; CHECK/MISSING={report['check_or_missing_runs']}")
    for p in pairs:
        if p["status"] == "READY":
            print(f"{p['comparison']}: budget_gap={p['budget_gap_pct_vs_right']:.2f}% ΔAP={p['delta_AP_pp']:.2f}pp ΔAP50={p['delta_AP50_pp']:.2f}pp ΔAP_S={p['delta_AP_S_pp']:.2f}pp")
        else:
            print(f"{p['comparison']}: {p['status']}")
    return 0 if report["check_or_missing_runs"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
