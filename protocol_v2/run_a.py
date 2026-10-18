#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
protocol_v2/run_a.py -- A 组实验 runner（方案 §05）

支持 A1–A8：
    A1  full          整图 JPEG Q95
    A2  ctx           全图 1/8 低清背景
    A3  regional      support only（ε=0.1）
    A4  regional      support ε=0.1 + random 8 patch
    A5  regional      support ε=0.7 + exploration 8 patch
    A6  packed        legacy uniform 25%（保留旧缺陷）
    A7  packed        prediction-guided 25%（证据感知）
    A8  packed        spatial uniform 25%（方案 §05 二维均匀）

A3–A8 需要边缘缓存（cache_edge 产出），所有配置共用同一份，保证 ROI 起点一致。

流程（每帧）
    原图 -> 选择内容 -> packet.bin -> 记录 len + sha256 -> POST /detect_packet
         -> 服务端只看到 packet（不碰原图/GT）-> 同一模型 -> 原图坐标预测

用法
    python -m protocol_v2.run_a --config A1 --limit 20
    python -m protocol_v2.run_a --config A8 --run-id A8_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime

import cv2
import numpy as np
import requests

sys.path.insert(0, "/app/Salience-DETR")
from protocol_v2 import codec      # noqa: E402
from protocol_v2 import selectors as sel  # noqa: E402

GT_PATH = "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"
IMG_DIR = "/app/data/seadronessee/compressed_version/val2017"
RUNS = "/app/Salience-DETR/experiments/runs"
EDGE_CACHE = "/app/Salience-DETR/experiments/edge_cache/edge_best_conf001.jsonl"

_BG = {"scale": 8, "q_bg": 95}
_TOK = {"scale": 8, "q_bg": 95, "q_atlas": 85}

CONFIGS = {
    "A1": {"desc": "Full frame, JPEG Q95, whole packet",
           "mode": "full", "q": 95},
    "A2": {"desc": "Context only, bg 1/8, JPEG Q95",
           "mode": "ctx", "scale": 8, "q": 95},
    "A3": {"desc": "Support only, eps=0.1 (regional)",
           "mode": "regional", "expand": 0.1, "q_roi": 80, **_BG},
    "A4": {"desc": "Support eps=0.1 + random 8 (regional)",
           "mode": "regional", "expand": 0.1, "random": 8, "q_roi": 95, **_BG},
    "A5": {"desc": "Support eps=0.1 + exploration 8 grid cells (regional)",
           "mode": "regional", "expand": 0.1, "explore": 4, "grid": (6, 10),
           "q_roi": 95, **_BG},
    "A6": {"desc": "Legacy uniform 25% (packed)",
           "mode": "packed", "expand": 0.1, "ratio": 1.00, "select": "uniform", **_TOK},
    "A7": {"desc": "Prediction-guided 25% (packed)",
           "mode": "packed", "expand": 0.1, "ratio": 1.00, "select": "evidence", **_TOK},
    "A8": {"desc": "Spatial uniform 25% (packed)",
           "mode": "packed", "expand": 0.1, "ratio": 1.00, "select": "spatial", **_TOK},
    "B3": {"desc": "Support only, no merge (regional)",
           "mode": "regional", "expand": 0.1, "merge": False, "q_roi": 95, **_BG},
}


def need_edge(cfg):
    return cfg["mode"] in ("regional", "packed")


def build_packet(cfg, rgb, iid, edge):
    """返回 (packet_bytes, n_unit)   n_unit = ROI 数或 patch 数"""
    h, w = rgb.shape[:2]
    mode = cfg["mode"]

    if mode == "full":
        return codec.encode_full_frame(rgb, iid, q=cfg["q"]), 0
    if mode == "ctx":
        return codec.encode_context_only(rgb, iid, scale=cfg["scale"], q=cfg["q"]), 0

    rec = edge.get(iid)
    if rec is None:
        raise RuntimeError("边缘缓存缺帧 image_id=%s" % iid)

    sup = sel.support_rois(rec["dets"], w, h,
                           expand=cfg.get("expand", 0.1),
                           min_ctx=cfg.get("min_ctx", sel.MIN_CTX))
    if cfg.get("merge", True):
        sup = sel.merge_rois(sup)
    if not sup:
        # 无边缘检测时用整图兜底，避免生成"空包"造成假性省带宽
        sup = [(0, 0, w, h)]

    if mode == "regional":
        rois = list(sup)
        if cfg.get("explore"):
            # 报告 §2.2：Exploration 是 6x10 网格中的 Top-K 个网格单元（整块矩形）
            rois += sel.exploration_rois(sup, w, h, n=cfg["explore"],
                                         grid=cfg.get("grid", (6, 10)))
        if cfg.get("random"):
            rois += [(x, y, x + sel.PATCH, y + sel.PATCH)
                     for x, y in sel.random_patches(sup, w, h, n=cfg["random"])]
        pkt = codec.encode_regional(rgb, iid, rois, scale=cfg["scale"],
                                    q_bg=cfg["q_bg"], q_roi=cfg["q_roi"])
        return pkt, len(rois)

    if mode == "packed":
        patches = sel.select_patches(rgb, sup, cfg["ratio"], cfg["select"], ps=sel.PATCH)
        pkt = codec.encode_packed(rgb, iid, patches, scale=cfg["scale"],
                                  q_bg=cfg["q_bg"], q_atlas=cfg["q_atlas"],
                                  patch_size=sel.PATCH)
        return pkt, len(patches)

    raise NotImplementedError("mode %s" % mode)


def post_packet(url, packet, expect_sha, conf=0.05, timeout=300, retries=3):
    last = None
    for k in range(retries):
        try:
            files = {"file": ("packet.bin", packet, "application/octet-stream")}
            data = {"conf_threshold": str(conf), "expect_sha256": expect_sha}
            r = requests.post(url, files=files, data=data, timeout=timeout)
            if r.status_code != 200:
                raise RuntimeError("HTTP %d: %s" % (r.status_code, r.text[:300]))
            return r.json()
        except Exception as exc:
            last = exc
            time.sleep(1.0 * (k + 1))
    raise RuntimeError("重试 %d 次仍失败: %r" % (retries, last))


def coco_eval(gt_path, pred_path, img_ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_path)
    e = COCOeval(coco_gt, coco_dt, "bbox")
    e.params.imgIds = sorted(img_ids)
    e.params.useCats = 1
    e.evaluate()
    e.accumulate()
    e.summarize()
    return {
        "AP": float(e.stats[0]), "AP50": float(e.stats[1]), "AP75": float(e.stats[2]),
        "AP_S": float(e.stats[3]), "AP_M": float(e.stats[4]), "AP_L": float(e.stats[5]),
        "AR_1": float(e.stats[6]), "AR_10": float(e.stats[7]), "AR_100": float(e.stats[8]),
        "AR_S": float(e.stats[9]), "AR_M": float(e.stats[10]), "AR_L": float(e.stats[11]),
        "n_img": len(e.params.imgIds),
    }


COLS = ["image_id", "file", "packet_bytes", "overhead", "n_det", "n_unit", "ms", "status"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="A1", choices=sorted(CONFIGS))
    ap.add_argument("--url", default="http://localhost:8002/detect_packet")
    ap.add_argument("--gt", default=GT_PATH)
    ap.add_argument("--img-dir", default=IMG_DIR)
    ap.add_argument("--edge", default=EDGE_CACHE)
    ap.add_argument("--edge-conf", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--run-id", default="")
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = args.run_id or ("%s_%s" % (args.config, ts))
    out = os.path.join(RUNS, run_id)
    os.makedirs(out, exist_ok=True)

    print("[run] %s  %s" % (run_id, cfg["desc"]))
    print("[run] out = %s" % out)

    with open(args.gt) as f:
        gt = json.load(f)
    images = gt["images"]
    if args.limit:
        images = images[:args.limit]
    print("[run] 帧数 = %d" % len(images))

    # ---------- 边缘缓存 ----------
    edge = {}
    if need_edge(cfg):
        if not os.path.exists(args.edge):
            sys.exit("[FATAL] 该配置需要边缘缓存，但 %s 不存在" % args.edge)
        edge = sel.load_edge_cache(args.edge, args.edge_conf)
        print("[edge] 载入 %d 帧  conf>=%.2f  %s"
              % (len(edge), args.edge_conf, args.edge))
        missing = [im["id"] for im in images if im["id"] not in edge]
        if missing:
            sys.exit("[FATAL] 缓存缺 %d 帧（示例 %s）——缓存未跑完，先等 cache_edge 结束"
                     % (len(missing), missing[:5]))
        n_det = sum(len(edge[im["id"]]["dets"]) for im in images)
        print("[edge] 平均边缘检测 %.2f 框/帧" % (n_det / float(len(images))))

    per_path = os.path.join(out, "per_image.csv")
    pred_path = os.path.join(out, "predictions_coco.json")
    idx_path = os.path.join(out, "packet_index.csv")
    fail_path = os.path.join(out, "failures.jsonl")

    done = {}
    if args.resume and os.path.exists(per_path):
        with open(per_path) as f:
            for row in csv.DictReader(f):
                done[int(row["image_id"])] = row
        print("[resume] 已完成 %d 帧" % len(done))

    todo = [im for im in images if im["id"] not in done]
    print("[run] 待跑 %d 帧 -> %s" % (len(todo), args.url))

    preds, rows, idx_rows, fails = [], [], [], []
    t0 = time.time()
    for k, im in enumerate(todo, 1):
        iid = im["id"]
        path = os.path.join(args.img_dir, im["file_name"])
        try:
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("读图失败: %s" % path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            packet, n_unit = build_packet(cfg, rgb, iid, edge)
            sha = codec.sha256(packet)
            ps = codec.part_sum(packet)
            if ps["payload"] + ps["overhead"] != ps["total"]:
                raise RuntimeError("字节不闭合: %r" % ps)

            resp = post_packet(args.url, packet, sha, args.conf)
            dets = resp.get("detections", []) or []
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                preds.append({"image_id": iid, "category_id": int(d["class_id"]),
                              "bbox": [x1, y1, w, h], "score": float(d["confidence"])})

            rows.append({"image_id": iid, "file": im["file_name"],
                         "packet_bytes": ps["total"], "overhead": ps["overhead"],
                         "n_det": len(dets), "n_unit": n_unit,
                         "ms": resp.get("inference_time_ms", 0), "status": "ok"})
            idx_rows.append({"image_id": iid, "total": ps["total"],
                             "payload": ps["payload"], "overhead": ps["overhead"],
                             "detail": json.dumps(ps["parts"], separators=(",", ":"))})
        except Exception as exc:
            fails.append({"image_id": iid, "file": im["file_name"],
                          "error": repr(exc)})
            rows.append({"image_id": iid, "file": im["file_name"],
                         "packet_bytes": 0, "overhead": 0, "n_det": -1, "n_unit": 0,
                         "ms": 0, "status": "failed"})
            print("  [%d/%d] %s FAILED %r" % (k, len(todo), im["file_name"], exc))
            traceback.print_exc()

        if k % 20 == 0 or k == len(todo):
            el = time.time() - t0
            print("  [%d/%d] %.1fs elapsed, ETA %.0fs"
                  % (k, len(todo), el, el / k * (len(todo) - k)))

    # 合并 resume 的历史
    all_rows = list(done.values()) + rows
    with open(per_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        wtr.writeheader()
        for r in sorted(all_rows, key=lambda x: int(x["image_id"])):
            r.setdefault("n_unit", 0)
            wtr.writerow(r)

    with open(idx_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["image_id", "total", "payload",
                                            "overhead", "detail"])
        wtr.writeheader()
        for r in idx_rows:
            wtr.writerow(r)

    with open(fail_path, "w") as f:
        for r in fails:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # resume 时合并旧预测（BUGFIX: 旧版 seen 误用新预测构造，导致新预测被全部丢弃）
    if args.resume and os.path.exists(pred_path):
        with open(pred_path) as f:
            old = json.load(f)
        if old:
            seen = {(p["image_id"], p["category_id"], tuple(p["bbox"]), p["score"])
                    for p in old}
            preds = old + [p for p in preds
                           if (p["image_id"], p["category_id"],
                               tuple(p["bbox"]), p["score"]) not in seen]
            print("[resume] 合并旧预测 %d 条 -> 合计 %d 条" % (len(old), len(preds)))
    with open(pred_path, "w") as f:
        json.dump(preds, f)

    ok_rows = [r for r in all_rows if r["status"] == "ok"]
    sizes = [int(r["packet_bytes"]) for r in ok_rows]
    if sizes:
        print("\n[payload] n=%d  mean=%.1f KB  median=%.1f  P90=%.1f  P95=%.1f  min=%.1f  max=%.1f"
              % (len(sizes),
                 np.mean(sizes) / 1024, np.median(sizes) / 1024,
                 np.percentile(sizes, 90) / 1024, np.percentile(sizes, 95) / 1024,
                 min(sizes) / 1024, max(sizes) / 1024))

    summary = {"run_id": run_id, "config": args.config, "cfg": cfg,
               "n_ok": len(ok_rows), "n_failed": len(fails),
               "n_pred": len(preds)}
    if sizes:
        summary.update({
            "payload_kb_mean": float(np.mean(sizes) / 1024),
            "payload_kb_median": float(np.median(sizes) / 1024),
            "payload_kb_p90": float(np.percentile(sizes, 90) / 1024),
            "payload_kb_p95": float(np.percentile(sizes, 95) / 1024),
        })
    units = [int(r["n_unit"]) for r in ok_rows if int(r.get("n_unit", 0) or 0) > 0]
    if units:
        summary["n_unit_mean"] = float(np.mean(units))

    if ok_rows and not fails:
        try:
            st = coco_eval(args.gt, pred_path, [int(r["image_id"]) for r in ok_rows])
            summary.update(st)
        except Exception as exc:
            print("[COCOeval] 失败: %r" % (exc,))
            traceback.print_exc()
    else:
        print("[COCOeval] 有失败帧 (%d) 或无结果, 本轮标为不完整, 不出主表数字"
              % len(fails))

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out, "run_manifest.json"), "w") as f:
        json.dump({"run_id": run_id, "config": args.config, "cfg": cfg,
                   "gt": args.gt, "img_dir": args.img_dir,
                   "edge_cache": args.edge if need_edge(cfg) else None,
                   "edge_conf": args.edge_conf,
                   "codec_version": codec.VERSION,
                   "url": args.url, "conf": args.conf,
                   "created": ts}, f, indent=2, ensure_ascii=False)

    print("\n[summary] %s" % json.dumps(
        {k: v for k, v in summary.items() if k in
         ("AP", "AP50", "AP_S", "AP_M", "AP_L", "payload_kb_mean",
          "n_unit_mean", "n_ok", "n_failed")},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
