#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P11 分析：① 恢复率 ② 排除调参帧 ③ 按视频分组的配对 bootstrap 95% 区间"""
import sys, os, io, json, argparse, contextlib, re
import numpy as np
from collections import defaultdict

EXP = "/app/Salience-DETR/experiments"
sys.path.insert(0, EXP)
DUMP = os.path.join(EXP, "p11_dump")
GT_PATH = "/app/datasets/afo_clean/annotations/instances_test.json"

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=200)
ap.add_argument("--exclude-first-n", type=int, default=200)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

import expE_evidence_aware as eE

GT = json.load(open(GT_PATH))
FILENAME = {int(im["id"]): im["file_name"] for im in GT["images"]}
ORDER = [int(im["id"]) for im in GT["images"]]
ANN = defaultdict(list)
for an in GT["annotations"]:
    b = an.get("bbox")
    if b and b[2] > 0 and b[3] > 0:
        ANN[int(an["image_id"])].append({"category_id": int(an["category_id"]),
                                         "bbox": list(map(float, b))})


def vid(img_id):
    m = re.match(r"^([^_]+)_", FILENAME.get(img_id, ""))
    return m.group(1) if m else "other"


def load(name):
    p = os.path.join(DUMP, name)
    return json.load(open(p)) if os.path.exists(p) else None


def evaluate(dets, img_ids, tag):
    if not dets: return None
    with contextlib.redirect_stdout(io.StringIO()):
        return eE.evaluate(dets, tag, set(int(i) for i in img_ids))


def iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    if xb <= xa or yb <= ya: return 0.0
    i = (xb-xa)*(yb-ya)
    return i/(a[2]*a[3] + b[2]*b[3] - i + 1e-9)


def match(dets, gts, thr=0.5):
    used, hit = set(), set()
    for d in sorted(dets, key=lambda x: -x["score"]):
        best, bi = thr, -1
        for gi, g in enumerate(gts):
            if gi in used or g["category_id"] != d["category_id"]: continue
            v = iou(d["bbox"], g["bbox"])
            if v > best: best, bi = v, gi
        if bi >= 0: used.add(bi); hit.add(bi)
    return hit


edge = load("edge.json")
A3 = load("A3_dets.json")
A4 = load("A4_dets.json")


def pick(ev, want):
    if not ev: return None, None
    for e in ev:
        if want.lower() in str(e["tag"]).lower(): return e["dets"], e["img_ids"]
    return ev[-1]["dets"], ev[-1]["img_ids"]


A5d, A5i = pick(load("A5_evaluates.json"), "e3")
A6d, A6i = pick(load("A6_evaluates.json"), "e2")
print("载入：edge=%s A3=%s A4=%s A5=%s A6=%s"
      % (len(edge or {}), len(A3 or []), len(A4 or []),
         "0" if A5d is None else len(A5d), "0" if A6d is None else len(A6d)))

CFG = [("A3", A3, ORDER), ("A4", A4, ORDER)]
if A5d: CFG.append(("A5", A5d, A5i))
if A6d: CFG.append(("A6", A6d, A6i))

print("\n" + "="*78)
print("① 全量指标复核（应与主表一致）")
print("="*78)
print("%-5s %10s %10s %10s %8s" % ("ID", "AP50", "mAP", "AP_S", "图数"))
base = {}
for tag, dets, ids in CFG:
    if not dets:
        print("%-5s %10s" % (tag, "[缺数据]")); continue
    m = evaluate(dets, ids, tag); base[tag] = m
    print("%-5s %10.4f %10.4f %10.4f %8d" % (tag, m["AP50"], m["AP"], m["APsmall"], len(ids)))

print("\n" + "="*78)
print("② 排除调参帧（test 前 %d 帧）" % a.exclude_first_n)
print("="*78)
keep = ORDER[a.exclude_first_n:]
ks = set(keep)
print("剩余 %d 帧 / 共 %d 帧" % (len(keep), len(ORDER)))
print("%-5s %10s %10s %10s" % ("ID", "AP50", "mAP", "AP_S"))
for tag, dets, ids in CFG:
    if not dets: continue
    d2 = [d for d in dets if int(d["image_id"]) in ks]
    m = evaluate(d2, [i for i in ids if i in ks], tag + "_ex")
    if m: print("%-5s %10.4f %10.4f %10.4f" % (tag, m["AP50"], m["AP"], m["APsmall"]))

if edge:
    print("\n" + "="*78)
    print("③ 恢复率（边缘漏检的 GT 中被方法重新检回的比例）")
    print("="*78)
    print("%-5s %10s %10s %10s" % ("ID", "边缘漏检", "已恢复", "恢复率"))
    for tag, dets, ids in CFG:
        if not dets or tag in ("A3", "A4"): continue
        by_img = defaultdict(list)
        for d in dets: by_img[int(d["image_id"])].append(d)
        miss = rec = 0
        for iid in ORDER:
            gts = ANN.get(iid, [])
            if not gts: continue
            ed = [{"category_id": d["category_id"], "bbox": d["bbox"], "score": d["score"]}
                  for d in edge.get(str(iid), [])]
            hit_e = match(ed, gts); hit_m = match(by_img.get(iid, []), gts)
            for gi in range(len(gts)):
                if gi not in hit_e:
                    miss += 1
                    if gi in hit_m: rec += 1
        print("%-5s %10d %10d %9.1f%%" % (tag, miss, rec, 100.0*rec/max(1, miss)))
    print("（口径：IoU≥0.5 且类别一致；对每个 det 按 score 降序贪心配对）")

print("\n" + "="*78)
print("④ 差值 95%% 区间（按视频前缀分组重采样，reps=%d）" % a.reps)
print("="*78)
groups = defaultdict(list)
for iid in ORDER: groups[vid(iid)].append(iid)
keys = sorted(groups.keys())
print("组数（视频前缀）：%d -> %s" % (len(keys), ", ".join(keys)))
rng = np.random.RandomState(a.seed)


def boot_pair(dA, dB):
    av, bv = defaultdict(list), defaultdict(list)
    for d in dA: av[int(d["image_id"])].append(d)
    for d in dB: bv[int(d["image_id"])].append(d)
    out = []
    for _ in range(a.reps):
        pg = [keys[i] for i in rng.randint(0, len(keys), len(keys))]
        sub = sorted({i for g in pg for i in groups[g]})
        if len(sub) < 20: continue
        ma = evaluate([d for i in sub for d in av.get(i, [])], sub, "bsA")
        mb = evaluate([d for i in sub for d in bv.get(i, [])], sub, "bsB")
        if ma and mb: out.append(ma["APsmall"] - mb["APsmall"])
    return np.array(out)


print("\n%-14s %10s %10s %22s" % ("对比", "点估计Δ", "bootstrap均值", "95% 区间"))
PAIRS = []
if A6d and A3: PAIRS.append(("A6 - A3", A6d, A3, "A6", "A3"))
if A5d and A4: PAIRS.append(("A5 - A4", A5d, A4, "A5", "A4"))
for name, dA, dB, ta, tb in PAIRS:
    d = boot_pair(dA, dB)
    if len(d) == 0:
        print("%-14s  [bootstrap 失败]" % name); continue
    lo, hi = np.percentile(d, [2.5, 97.5])
    pt = (base[ta]["APsmall"] - base[tb]["APsmall"]) if (ta in base and tb in base) else float("nan")
    print("%-14s %10.4f %10.4f  [%+.4f, %+.4f]  n=%d" % (name, pt, d.mean(), lo, hi, len(d)))

print("\n读法：区间不含 0 即该差值在 95% 水平上显著；跨 0 则证据不足。")
