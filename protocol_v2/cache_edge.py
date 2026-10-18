#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
protocol_v2/cache_edge.py -- 边缘预测缓存（方案 §07 第 2 步）

要求（方案原文）：
    "按 V_all 运行一次边缘模型；记录框、分数、类别、原图尺寸 —— 所有配置复用同份缓存"

所以本脚本**只跑一次**，A3/A4/A5/A6/A7/A8 全部读同一份 jsonl。
每个配置各自跑一遍边缘会导致选择结果不可比（随机性、推理抖动），方案明确禁止。

输出（jsonl，一行一帧）：
    {"image_id":.., "file":"..", "w":.., "h":..,
     "boxes":[[x1,y1,x2,y2],..], "scores":[..], "labels":[..], "ms":..}

用法：
    python -m protocol_v2.cache_edge                       # 全量 1547，CPU
    python -m protocol_v2.cache_edge --limit 20            # 冒烟
    python -m protocol_v2.cache_edge --resume              # 断点续跑
    python -m protocol_v2.cache_edge --device npu --card 4 # 换 NPU 加速
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

YOLO_W = "/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt"
GT = "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"
IMG = "/app/data/seadronessee/compressed_version/val2017"
OUT = "/app/Salience-DETR/experiments/edge_cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=GT)
    ap.add_argument("--img-dir", default=IMG)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--model", default=YOLO_W)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--imgsz", type=int, default=0, help="0=ultralytics 默认(与主实验一致)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "npu"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = os.path.basename(args.model).replace(".pt", "")
    out_path = os.path.join(args.out_dir, "edge_%s_conf%s.jsonl" % (tag, str(args.conf).replace(".", "")))

    with open(args.gt) as f:
        gt = json.load(f)
    images = gt["images"]
    if args.limit:
        images = images[:args.limit]

    done = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(int(json.loads(line)["image_id"]))
                except Exception:
                    pass
        print("[resume] 已缓存 %d 帧" % len(done))

    todo = [im for im in images if im["id"] not in done]
    print("[cache] 待跑 %d / %d 帧  device=%s  conf=%.2f  -> %s"
          % (len(todo), len(images), args.device, args.conf, out_path))

    from ultralytics import YOLO
    model = YOLO(args.model)

    kw = dict(conf=args.conf, verbose=False, device=args.device)
    if args.imgsz > 0:
        kw["imgsz"] = args.imgsz

    t0 = time.time()
    n_box = 0
    with open(out_path, "a") as fo:
        for k, im in enumerate(todo, 1):
            p = os.path.join(args.img_dir, im["file_name"])
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None:
                print("  读图失败: %s" % p)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]

            ts = time.time()
            r = model.predict(rgb, **kw)[0]
            ms = (time.time() - ts) * 1000.0

            if r.boxes is None or len(r.boxes) == 0:
                boxes, scores, labels = [], [], []
            else:
                boxes = r.boxes.xyxy.cpu().numpy().astype(float).tolist()
                scores = r.boxes.conf.cpu().numpy().astype(float).tolist()
                labels = r.boxes.cls.cpu().numpy().astype(int).tolist()
            n_box += len(boxes)

            fo.write(json.dumps({
                "image_id": int(im["id"]), "file": im["file_name"],
                "w": int(w), "h": int(h),
                "boxes": boxes, "scores": scores, "labels": labels,
                "ms": round(ms, 1),
            }, separators=(",", ":")) + "\n")
            fo.flush()

            if k % 50 == 0 or k == len(todo):
                el = time.time() - t0
                print("  [%d/%d] 平均 %.0fms/帧  累计框 %d  已用 %.0fs  ETA %.0fs"
                      % (k, len(todo), ms, n_box, el, el / k * (len(todo) - k)))

    total = len(done) + len(todo)
    print("\n[done] %s" % out_path)
    print("[done] 覆盖 %d 帧，平均 %.1f 框/帧，文件 %.1f KB"
          % (total, n_box / max(1, len(todo)), os.path.getsize(out_path) / 1024))


if __name__ == "__ma" + "in__":
    main()
