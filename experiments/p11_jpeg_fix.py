#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, json
from collections import Counter
import numpy as np
from PIL import Image
sys.path.insert(0, "/app/Salience-DETR/experiments")
import exp_resize_jpeg_baseline as rj

OUT = "/app/Salience-DETR/experiments/p11_dump"
os.makedirs(OUT, exist_ok=True)
IMG = rj.IMG_DIR
imgs = json.load(open(rj.GT_PATH))["images"]
print("num images:", len(imgs), flush=True)


def det_to_coco(d, iid):
    if isinstance(d, dict):
        bb = d.get("bbox")
        if not bb or len(bb) < 4:
            return None
        sc = float(d.get("confidence", d.get("score", 0.0)))
        cid = int(d.get("class_id", d.get("category_id", 0)))
        bb = [float(v) for v in bb[:4]]
    elif isinstance(d, (list, tuple)) and len(d) >= 4:
        bb = [float(d[0]), float(d[1]), float(d[2]), float(d[3])]
        sc = float(d[4]) if len(d) > 4 else 1.0
        cid = int(d[5]) if len(d) > 5 else 0
    else:
        return None
    return {"image_id": int(iid), "category_id": cid,
            "bbox": [bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]], "score": sc}


for tag, s, q in [("A3", 0.40, 50), ("A4", 0.50, 95)]:
    print("== %s s=%.2f q=%d ==" % (tag, s, q), flush=True)
    dets, tb, n = [], 0.0, 0
    for i, im in enumerate(imgs):
        p = os.path.join(IMG, im["file_name"])
        if not os.path.exists(p):
            continue
        img = rj.cv2.imread(p)
        canvas, pl = rj.compute_resize_jpeg(img, s, q)
        tb += float(pl)
        n += 1
        raw = rj.cloud_detect(rj.cv2.cvtColor(canvas, rj.cv2.COLOR_BGR2RGB), 0.05)
        if isinstance(raw, dict):
            raw = raw.get("detections", [])
        for d in (raw or []):
            r = det_to_coco(d, im["id"])
            if r:
                dets.append(r)
        if (i + 1) % 100 == 0:
            print("  %d/%d dets=%d" % (i + 1, len(imgs), len(dets)), flush=True)
    kb = tb / 1024.0 / max(1, n)
    if kb < 1:
        kb = tb / max(1, n)
    json.dump(dets, open(OUT + "/%s_dets.json" % tag, "w"))
    print("  %s: %d dets | %.1f KB/img | %d imgs | %s"
          % (tag, len(dets), kb, n,
             dict(sorted(Counter(x["category_id"] for x in dets).items()))), flush=True)
print("done", flush=True)
