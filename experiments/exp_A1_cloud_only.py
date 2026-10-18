#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_A1_cloud_only.py -- SEA 纯云端上界（A1 全图上传，无 nms_fuse）

定义:
    原图 JPEG 文件字节 --> 直接 POST /detect --> 云端 DETR(Swin-L) --> COCOeval
    不经过边缘检测, 不做 ROI 选择, 不做任何结果融合。

与历史 S3_full_image_upper_bound (=0.8660 / 1275 KB) 的区别:
    S3 **含 nms_fuse**(边缘框与云端框融合), 本脚本不含。
    两者差值即 nms_fuse 的净贡献, 可作为一条独立结论。

载荷口径:
    直接使用原文件字节 (不解码再编码), 是最"纯"的全图上传口径。
    同时记录 q95 重编码字节, 便于与 v1 的 1275 KB / 1490.1 KB 对齐。

输出:
    <out>/A1_cloud_only.json   逐图: image_id, bytes_raw, bytes_q95, n_det, latency_ms
    <out>/A1_results.json      COCO 格式检测结果 (供 COCOeval / 复查)
    stdout                     汇总 AP / AP50 / AP75 / AP_S / AP_M / AP_L / AR

用法:
    # 先冒烟 20 张, 核对类别映射
    python experiments/exp_A1_cloud_only.py --limit 20

    # 全量 1547
    python experiments/exp_A1_cloud_only.py

    # 断点续跑 (已完成的 image_id 自动跳过)
    python experiments/exp_A1_cloud_only.py --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

SEA = {
    "gt": "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json",
    "img": "/app/data/seadronessee/compressed_version/val2017",
}
DEFAULT_OUT = "/app/Salience-DETR/experiments/a1_cloud_only"


def load_gt(gt_path):
    with open(gt_path, "r") as f:
        gt = json.load(f)
    images = gt["images"]
    cats = {c["id"]: c["name"] for c in gt.get("categories", [])}
    # name -> id, 用于把云端返回的 class_name 映射回 GT 的 category_id
    name2id = {}
    for c in gt.get("categories", []):
        name2id[c["name"].strip().lower()] = c["id"]
    return images, cats, name2id


def post_image(url, raw_bytes, conf=0.05, timeout=300):
    import requests
    t0 = time.time()
    r = requests.post(
        url,
        files={"file": ("frame.jpg", raw_bytes, "image/jpeg")},
        data={"conf": str(conf)},
        timeout=timeout,
    )
    dt = (time.time() - t0) * 1000.0
    if r.status_code != 200:
        raise RuntimeError("HTTP %d: %s" % (r.status_code, r.text[:300]))
    return r.json(), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001/detect")
    ap.add_argument("--gt", default=SEA["gt"])
    ap.add_argument("--img-dir", default=SEA["img"])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张, 0=全量")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--conf", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    per_path = os.path.join(args.out, "A1_cloud_only.json")
    res_path = os.path.join(args.out, "A1_results.json")

    images, cats, name2id = load_gt(args.gt)
    print("[GT] %d 张图 / %d 类" % (len(images), len(cats)))
    print("[GT] categories: %s" % json.dumps(cats, ensure_ascii=False))
    print("[GT] name2id: %s" % json.dumps(name2id, ensure_ascii=False))

    done = {}
    results = []
    if args.resume and os.path.exists(per_path):
        with open(per_path) as f:
            for rec in json.load(f):
                done[rec["image_id"]] = rec
        print("[resume] 已完成 %d 张" % len(done))
    if args.resume and os.path.exists(res_path):
        with open(res_path) as f:
            results = json.load(f)
        print("[resume] 已载入 %d 条检测结果" % len(results))

    todo = [im for im in images if im["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print("[run] 待跑 %d 张 -> %s" % (len(todo), args.url))

    unmapped = {}
    t_start = time.time()
    for k, im in enumerate(todo, 1):
        iid = im["id"]
        path = os.path.join(args.img_dir, im["file_name"])
        try:
            with open(path, "rb") as f:
                raw = f.read()
            resp, dt = post_image(args.url, raw, args.conf)
            dets = resp.get("detections", []) if isinstance(resp, dict) else []

            n_ok = 0
            for d in dets:
                nm = str(d.get("class_name", "")).strip().lower()
                if nm not in name2id:
                    unmapped[nm] = unmapped.get(nm, 0) + 1
                    continue
                bb = d.get("bbox") or []
                if len(bb) < 4:
                    continue
                x1, y1, x2, y2 = [float(v) for v in bb[:4]]
                w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue
                results.append({
                    "image_id": iid,
                    "category_id": name2id[nm],
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "score": float(d.get("confidence", d.get("score", 0.0))),
                })
                n_ok += 1

            done[iid] = {
                "image_id": iid,
                "file": im["file_name"],
                "bytes_raw": len(raw),
                "n_det": len(dets),
                "n_mapped": n_ok,
                "latency_ms": round(dt, 1),
            }
        except Exception as exc:
            print("  [%d/%d] %s FAILED: %r" % (k, len(todo), im["file_name"], exc))
            traceback.print_exc()
            continue

        if k % 25 == 0 or k == len(todo):
            el = time.time() - t_start
            eta = el / k * (len(todo) - k)
            print("  [%d/%d] %s | det=%d | %.0fms | elapsed %.0fs ETA %.0fs"
                  % (k, len(todo), im["file_name"], len(dets), dt, el, eta))

    if unmapped:
        print("\n[WARN] 以下类别名无法映射到 GT, 已丢弃:")
        for k2, v2 in sorted(unmapped.items(), key=lambda x: -x[1]):
            print("   %-28s x%d" % (k2, v2))

    with open(per_path, "w") as f:
        json.dump(list(done.values()), f, indent=2)
    with open(res_path, "w") as f:
        json.dump(results, f)
    print("\n[out] %s" % per_path)
    print("[out] %s (%d 条)" % (res_path, len(results)))

    # ---- 载荷统计 ----
    raws = [r["bytes_raw"] for r in done.values()]
    if raws:
        print("\n[payload] 原文件字节  mean=%.1f KB  median=%.1f KB  min=%.1f  max=%.1f"
              % (sum(raws) / len(raws) / 1024,
                 sorted(raws)[len(raws) // 2] / 1024,
                 min(raws) / 1024, max(raws) / 1024))

    # ---- COCOeval ----
    if not results:
        print("\n[COCOeval] 无结果, 跳过")
        return
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        coco_gt = COCO(args.gt)
        coco_dt = coco_gt.loadRes(res_path)
        e = COCOeval(coco_gt, coco_dt, "bbox")
        e.params.imgIds = sorted({r["image_id"] for r in results})
        e.evaluate()
        e.accumulate()
        e.summarize()
        print("\n[COCOeval] 评估图数 = %d" % len(e.params.imgIds))
    except Exception as exc:
        print("\n[COCOeval] 失败: %r" % (exc,))
        traceback.print_exc()


if __name__ == "__ma" + "in__":
    main()
