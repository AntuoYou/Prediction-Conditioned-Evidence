#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_pipeline.py v2 -- 用主实验真实链路生成论文插图

v2 相对 v1 的改动:
  1. 选帧过滤掉小图 (--min-side 2000), 避免 1231x933 与 4K 混排
  2. GT 框改为白色粗实线, edge 框改为半透明细线, 不再互相盖住
  3. 所有载荷同时标注两种口径:
       stream = 协议口径 (small + atlas + 4N + 4)
       canvas = 当前脚手架把整张全分辨率画布编码后的实际字节
  4. pipeline 的 (d) 栏加 ROI 放大 inset, 让 patch 化真正可见
  5. 新增 fig_token_zoom.png: 原图ROI / A5 / A6-25% / A6-100% 四栏放大对比
  6. pipeline 帧自动选 A6 表现最好的一帧 (--fig-ids 可强制指定)

输出:
    <out>/fig_pipeline_<imgid>.png
    <out>/fig_qualitative.png
    <out>/fig_token_zoom.png
    <out>/viz_metrics.json
    <out>/captions.md

用法:
    python experiments/viz_pipeline.py --probe
    python experiments/viz_pipeline.py --url http://localhost:8001/detect --n 4
    python experiments/viz_pipeline.py --url ... --n 4 --fig-ids 4441
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback

import numpy as np
import cv2

ROOT = "/app/Salience-DETR"
EXP = os.path.join(ROOT, "experiments")
for _p in (ROOT, EXP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEA = {
    "yolo": "/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt",
    "gt": "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json",
    "img": "/app/data/seadronessee/compressed_version/val2017",
}

SEA_CLASSES = {"swimmer", "boat", "float", "buoy", "jetski", "life_saving_appliances"}


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def load_mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法为 %s 创建 spec" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def jpeg_bytes(rgb: np.ndarray, q: int = 95) -> int:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return int(buf.nbytes)


def iou_xyxy(a, b) -> float:
    xa1, ya1, xa2, ya2 = a[0], a[1], a[2], a[3]
    xb1, yb1, xb2, yb2 = b[0], b[1], b[2], b[3]
    iw = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    ih = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = iw * ih
    ua = (max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
          + max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1) - inter)
    return inter / ua if ua > 0 else 0.0


def match_tp_fp_fn(preds, gts, thr=0.5):
    order = sorted(range(len(preds)), key=lambda i: -preds[i][4])
    used = [False] * len(gts)
    tp, fp = [], []
    for i in order:
        p = preds[i]
        best_i, best_v = -1, 0.0
        for j, g in enumerate(gts):
            if used[j] or int(p[5]) != int(g[4]):
                continue
            v = iou_xyxy(p, g)
            if v > best_v:
                best_v, best_i = v, j
        if best_i >= 0 and best_v >= thr:
            used[best_i] = True
            tp.append((p, gts[best_i], best_v))
        else:
            fp.append(p)
    fn = [g for j, g in enumerate(gts) if not used[j]]
    return tp, fp, fn


def kb(n):
    return n / 1024.0


# --------------------------------------------------------------------------
# 云端
# --------------------------------------------------------------------------
class CloudClient:
    def __init__(self, url: str, conf: float = 0.05, timeout: int = 300):
        self.url = url
        self.conf = conf
        self.timeout = timeout
        self.field = None
        self.extra = {}

    def probe(self, verbose=True):
        import requests
        dummy = np.full((64, 64, 3), 128, dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(dummy, cv2.COLOR_RGB2BGR))
        raw = buf.tobytes()
        tried = []
        for fname in ("file", "image", "img", "upload"):
            for extra in ({}, {"roi_index": "0"}):
                try:
                    r = requests.post(
                        self.url,
                        files={fname: ("probe.jpg", raw, "image/jpeg")},
                        data=dict({"conf": str(self.conf)}, **extra),
                        timeout=60,
                    )
                except Exception as exc:
                    raise RuntimeError("无法连接 %s : %s" % (self.url, exc))
                tried.append((fname, extra, r.status_code))
                if r.status_code == 200:
                    self.field = fname
                    self.extra = extra
                    if verbose:
                        print("[cloud] 字段名=%s 额外参数=%s  -> %s" % (fname, extra, self.url))
                    return True
        raise RuntimeError("探测 /detect 失败, 尝试记录: %s" % tried)

    def detect(self, canvas_rgb: np.ndarray):
        import requests
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError("上传编码失败")
        r = requests.post(
            self.url,
            files={self.field: ("canvas.jpg", buf.tobytes(), "image/jpeg")},
            data=dict({"conf": str(self.conf)}, **self.extra),
            timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError("云端返回 %d: %s" % (r.status_code, r.text[:400]))
        j = r.json()
        dets = j.get("detections", []) if isinstance(j, dict) else []
        out = []
        for d in dets:
            bb = d.get("bbox") or d.get("box")
            if not bb or len(bb) < 4:
                continue
            out.append({
                "bbox": [float(v) for v in bb[:4]],
                "score": float(d.get("confidence", d.get("score", 0.0))),
                "cls": int(d.get("class_id", 0)),
                "name": str(d.get("class_name", "")),
            })
        return out, j


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
def load_gt(gt_path: str):
    with open(gt_path, "r") as f:
        gt = json.load(f)
    images = {im["id"]: im for im in gt["images"]}
    cats = {c["id"]: c["name"] for c in gt.get("categories", [])}
    anns = {}
    for a in gt["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    return images, cats, anns


def img_path(im):
    return os.path.join(SEA["img"], im["file_name"])


def probe_size(im):
    """读图幅, 用于过滤小图"""
    p = img_path(im)
    try:
        from PIL import Image as _I
        with _I.open(p) as f:
            w, h = f.size
        return int(w), int(h)
    except Exception:
        return 0, 0


def pick_frames(images, anns, n=4, ids=None, min_side=2000):
    if ids:
        want = set(str(x).strip() for x in ids)
        picked = []
        for iid, im in images.items():
            if str(iid) in want or str(im["file_name"]).split(".")[0] in want:
                picked.append(iid)
        if not picked:
            raise RuntimeError("--ids/--fig-ids 没有匹配到任何帧: %s" % ids)
        return picked[:n]

    scored = []
    skipped = 0
    for iid, im in images.items():
        al = anns.get(iid, [])
        if not al:
            continue
        areas = [float(a.get("area", 0)) for a in al]
        small = sum(1 for a in areas if a < 32 * 32)
        ratio = small / max(1, len(al))
        scored.append((len(al) * 1.0 + ratio * 12.0, iid))
    scored.sort(reverse=True)

    out = []
    for _, iid in scored:
        w, h = probe_size(images[iid])
        if min_side and max(w, h) < min_side:
            skipped += 1
            continue
        out.append(iid)
        if len(out) >= n:
            break
    if skipped:
        print("      过滤掉 %d 张小于 %d 的帧" % (skipped, min_side))
    return out


# --------------------------------------------------------------------------
# 边缘 / A5 / A6
# --------------------------------------------------------------------------
class EdgeModelCPU:
    def __init__(self, yolo, imgsz=0):
        self.yolo = yolo
        self.imgsz = int(imgsz)

    def __call__(self, im, conf=0.01, verbose=False):
        kw = dict(conf=conf, verbose=verbose, device="cpu")
        if self.imgsz > 0:
            kw["imgsz"] = self.imgsz
        return self.yolo.predict(im, **kw)


def make_support_rois_auto(expE, img, edge_dets, expand):
    W = img.shape[1]
    H = img.shape[0]
    boxes = [tuple(float(v) for v in d.bbox) for d in edge_dets]
    cands = [
        ("tuple", boxes),
        ("obj", edge_dets),
        ("dict", [{"bbox": tuple(float(v) for v in d.bbox)} for d in edge_dets]),
        ("5tuple", [(b[0], b[1], b[2], b[3], float(getattr(d, "score", 1.0)))
                    for b, d in zip(boxes, edge_dets)]),
    ]
    last_err = None
    for tag, data in cands:
        try:
            rois = expE.make_support_rois(img, data, expand=expand)
        except Exception as exc:
            last_err = "%s -> %s" % (tag, exc)
            continue
        rois = [r for r in rois if r and len(r) >= 4 and r[2] > r[0] and r[3] > r[1]]
        if rois:
            rois = [(int(max(0, r[0])), int(max(0, r[1])),
                     int(min(W, r[2])), int(min(H, r[3]))) for r in rois]
            print("      [expE.make_support_rois] 格式=%s -> %d 个 ROI" % (tag, len(rois)))
            return rois, tag
    if not boxes:
        return [], "empty"
    raise RuntimeError("expE.make_support_rois 所有格式均失败: %s" % last_err)


def run_frame(iid, im_info, gt_anns, yolo, expE, expB, client, token_ratio,
              budget, imgsz, dry_run):
    bgr = cv2.imread(img_path(im_info), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("读图失败: %s" % img_path(im_info))
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    gts = []
    for a in gt_anns.get(iid, []):
        x, y, w, h = a["bbox"]
        gts.append((float(x), float(y), float(x + w), float(y + h), int(a["category_id"])))
    print("   GT: %d 个目标, 图幅 %dx%d" % (len(gts), W, H))

    edge_out = expB.build_edge_output(img, iid, EdgeModelCPU(yolo, imgsz))
    edge_dets = list(edge_out.detections)
    print("   edge YOLOv8s: %d 个检测" % len(edge_dets))

    # A5
    a5 = {}
    try:
        canvas5, sup5, patches5 = expB.build_combined_canvas(img, edge_out, budget)
        rep5 = int(expB.calc_payload(img, sup5, patches5))
        act5 = jpeg_bytes(canvas5, 95)
        a5 = {"canvas": canvas5, "sup": sup5, "n_patch": len(patches5),
              "stream": rep5, "canvas_kb": act5, "ok": True}
        print("   A5: ROI=%d patch=%d stream=%.1fKB canvas=%.1fKB"
              % (len(sup5), len(patches5), kb(rep5), kb(act5)))
    except Exception as exc:
        a5 = {"ok": False, "err": repr(exc)}
        print("   A5 失败: %r" % (exc,))
        traceback.print_exc()

    # A6
    a6, a6_full = {}, {}
    rois6, sup6, exp6, sup_c, exp_c = [], [], [], None, None
    try:
        sup6, fmt = make_support_rois_auto(expE, img, edge_dets,
                                           float(getattr(expE, "EXPAND_RATIO", 0.1)))
        exp6 = expE.generate_exploration_rois(sup6, W, H,
                                              topk=int(getattr(expE, "TOPK_EXPLORE", 8)))
        exp6 = [(int(r[0]), int(r[1]), int(min(W, r[2])), int(min(H, r[3]))) for r in exp6]
        rois6 = list(sup6) + list(exp6)
        sup_c = expE.get_roi_centers(sup6)
        exp_c = expE.get_roi_centers(exp6)

        canvas6, rep6 = expE.pack_and_transmit(img, rois6, token_ratio, "evidence", sup_c, exp_c)
        act6 = jpeg_bytes(canvas6, 95)
        a6 = {"canvas": canvas6, "sup": sup6, "exp": exp6, "n_roi": len(rois6),
              "stream": int(rep6), "canvas_kb": act6, "ok": True}
        print("   A6: ROI=%d(sup %d + exp %d) stream=%.1fKB canvas=%.1fKB"
              % (len(rois6), len(sup6), len(exp6), kb(rep6), kb(act6)))

        # A6 token_ratio=1.0 (对照)
        cfull, repf = expE.pack_and_transmit(img, rois6, 1.0, "uniform", sup_c, exp_c)
        actf = jpeg_bytes(cfull, 95)
        a6_full = {"canvas": cfull, "stream": int(repf), "canvas_kb": actf, "ok": True}
        print("   A6@100%%: stream=%.1fKB canvas=%.1fKB" % (kb(repf), kb(actf)))
    except Exception as exc:
        a6 = {"ok": False, "err": repr(exc)}
        print("   A6 失败: %r" % (exc,))
        traceback.print_exc()

    # 云端
    res = {}
    targets = [("A5", a5.get("canvas")), ("A6", a6.get("canvas")),
               ("A6_full", a6_full.get("canvas"))]
    for tag, canvas in targets:
        if canvas is None:
            res[tag] = {"ok": False, "err": "no canvas"}
            continue
        if dry_run:
            res[tag] = {"ok": False, "err": "dry-run"}
            continue
        try:
            dets, raw = client.detect(canvas)
            preds = [(d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3],
                      d["score"], d["cls"]) for d in dets]
            tp, fp, fn = match_tp_fp_fn(preds, gts, 0.5)
            res[tag] = {"ok": True, "preds": preds, "tp": len(tp), "fp": len(fp), "fn": len(fn)}
            print("   cloud[%s]: %d det -> TP=%d FP=%d FN=%d"
                  % (tag, len(dets), len(tp), len(fp), len(fn)))
        except Exception as exc:
            res[tag] = {"ok": False, "err": repr(exc)}
            print("   cloud[%s] 失败: %r" % (tag, exc))

    return {"iid": iid, "file": im_info["file_name"], "img": img, "W": W, "H": H,
            "gts": gts,
            "edge": [(d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3],
                      float(d.score), int(d.label)) for d in edge_dets],
            "a5": a5, "a6": a6, "a6_full": a6_full, "res": res,
            "ratio": token_ratio}


# --------------------------------------------------------------------------
# 绘图样式
# --------------------------------------------------------------------------
C_GT = "#FFFFFF"
C_EDGE = "#FF3B30"
C_SUP = "#FF3B30"
C_EXP = "#FF9500"
C_TP = "#32D74B"
C_FP = "#FF375F"
C_FN = "#FFD60A"


def _disp(h, w, maxside):
    s = float(maxside) / max(h, w)
    return (s, int(round(w * s)), int(round(h * s))) if s < 1 else (1.0, w, h)


def _box(ax, b, color, ls="-", lw=1.6, alpha=1.0):
    from matplotlib.patches import Rectangle
    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
    if x2 <= x1 or y2 <= y1:
        return
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                           edgecolor=color, linestyle=ls, linewidth=lw, alpha=alpha))


def _clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#888888")
        sp.set_linewidth(0.6)


def _payload_2line(d):
    """双口径载荷文字"""
    if not d.get("ok"):
        return "FAILED"
    return "stream %.1f KB\ncanvas %.1f KB" % (kb(d["stream"]), kb(d["canvas_kb"]))


def pick_roi_crop(rec, pad=0.18):
    """挑一个含 GT、尺寸适中的 support ROI, 返回外扩后的 crop 框"""
    a6 = rec["a6"]
    if not a6.get("ok") or not a6["sup"]:
        return None
    gts = rec["gts"]
    W, H = rec["W"], rec["H"]
    best, best_score = None, -1
    for r in a6["sup"]:
        x1, y1, x2, y2 = r
        w, h = x2 - x1, y2 - y1
        if w < 60 or h < 60 or w * h > 0.25 * W * H:
            continue
        # 与 GT 的重叠
        hit = 0
        for g in gts:
            ix = max(0, min(x2, g[2]) - max(x1, g[0]))
            iy = max(0, min(y2, g[3]) - max(y1, g[1]))
            if ix > 0 and iy > 0:
                hit += 1
        if hit == 0:
            continue
        # 偏好: 含目标多的、边长 120-380 的
        side = max(w, h)
        size_pref = 1.0 - abs(np.log(max(side, 2) / 220.0)) * 0.35
        sc = hit * 2.0 + max(0.0, size_pref) * 3.0
        if sc > best_score:
            best_score, best = sc, r
    if best is None:
        return None
    x1, y1, x2, y2 = best
    px = int((x2 - x1) * pad) + 8
    py = int((y2 - y1) * pad) + 8
    return (max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py))


# --------------------------------------------------------------------------
# 图 1: pipeline (5 栏 + inset)
# --------------------------------------------------------------------------
def fig_pipeline(rec, out_path, maxside=1500):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    img, W, H = rec["img"], rec["W"], rec["H"]
    s, dw, dh = _disp(H, W, maxside)
    small = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA) if s < 1 else img

    a6 = rec["a6"]
    res6 = rec["res"].get("A6", {})

    def sc(b):
        return (b[0] * s, b[1] * s, b[2] * s, b[3] * s)

    fig, axes = plt.subplots(1, 5, figsize=(27, 5.6))

    # (a) 原图 + GT
    ax = axes[0]
    ax.imshow(small)
    for g in rec["gts"]:
        _box(ax, sc(g), C_GT, "-", 2.2)
    ax.set_title("(a) Raw frame + GT\n%d objects, %dx%d" % (len(rec["gts"]), W, H), fontsize=10)

    # (b) 边缘
    ax = axes[1]
    ax.imshow(small)
    for e in rec["edge"]:
        _box(ax, sc(e), C_EDGE, "-", 1.0, alpha=0.6)
    ax.set_title("(b) Edge YOLOv8s\n%d detections @ conf 0.01" % len(rec["edge"]), fontsize=10)

    # (c) ROI
    ax = axes[2]
    vis = (small.astype(np.float32) * 0.30).astype(np.uint8)
    if a6.get("ok"):
        for r in a6["sup"]:
            x1, y1, x2, y2 = [int(v * s) for v in r]
            if x2 > x1 and y2 > y1:
                vis[y1:y2, x1:x2] = small[y1:y2, x1:x2]
    ax.imshow(vis)
    if a6.get("ok"):
        for r in a6["sup"]:
            _box(ax, sc(r), C_SUP, "-", 1.6)
        for r in a6["exp"]:
            _box(ax, sc(r), C_EXP, "--", 1.4)
        # 只给前 3 个最大的 support ROI 画 patch 网格
        big = sorted(a6["sup"], key=lambda r: -(r[2] - r[0]) * (r[3] - r[1]))[:3]
        ps = float(getattr(expE_globals, "PATCH_SIZE", 16)) * s
        for r in big:
            x1, y1, x2, y2 = [int(v * s) for v in r]
            for xx in np.arange(x1, x2, ps):
                ax.plot([xx, xx], [y1, y2], color="white", lw=0.35, alpha=0.22)
            for yy in np.arange(y1, y2, ps):
                ax.plot([x1, x2], [yy, yy], color="white", lw=0.35, alpha=0.22)
    n_sup = len(a6["sup"]) if a6.get("ok") else 0
    n_exp = len(a6["exp"]) if a6.get("ok") else 0
    ax.set_title("(c) Support + Exploration ROI\nsupport %d (solid) / explore %d (dashed)"
                 % (n_sup, n_exp), fontsize=10)

    # (d) token 载荷 + inset
    ax = axes[3]
    if a6.get("ok"):
        c = a6["canvas"]
        cs = cv2.resize(c, (dw, dh), interpolation=cv2.INTER_AREA) if s < 1 else c
        ax.imshow(cs)
        ax.set_title("(d) Tokenized payload (ratio %.0f%%)\n%s"
                     % (rec["ratio"] * 100, _payload_2line(a6)), fontsize=10)
        crop = pick_roi_crop(rec)
        if crop:
            cx1, cy1, cx2, cy2 = crop
            _box(ax, sc(crop), "#FFD60A", "-", 1.8)
            # inset: 放大 crop
            ins = ax.inset_axes([0.585, 0.015, 0.395, 0.395])
            patch = c[cy1:cy2, cx1:cx2]
            ins.imshow(patch, interpolation="nearest")
            ins.set_xticks([])
            ins.set_yticks([])
            for sp in ins.spines.values():
                sp.set_color("#FFD60A")
                sp.set_linewidth(1.6)
            ins.set_title("zoom", fontsize=7, color="#FFD60A", pad=2)
    else:
        ax.imshow(small)
        ax.set_title("(d) Tokenized payload\nFAILED", fontsize=10)

    # (e) 云端结果
    ax = axes[4]
    if a6.get("ok"):
        c = a6["canvas"]
        cs = cv2.resize(c, (dw, dh), interpolation=cv2.INTER_AREA) if s < 1 else c
        ax.imshow(cs)
    else:
        ax.imshow(small)
    if res6.get("ok"):
        tp_l, fp_l, fn_l = match_tp_fp_fn(res6["preds"], rec["gts"], 0.5)
        for p, g, v in tp_l:
            _box(ax, sc(p), C_TP, "-", 1.9)
        for p in fp_l:
            _box(ax, sc(p), C_FP, "--", 1.3)
        for g in fn_l:
            _box(ax, sc(g), C_FN, "--", 1.6)
        ax.set_title("(e) Cloud DETR (Swin-L)\nTP %d / FP %d / FN %d"
                     % (res6["tp"], res6["fp"], res6["fn"]), fontsize=10)
    else:
        ax.set_title("(e) Cloud DETR\nFAILED: %s" % str(res6.get("err"))[:60], fontsize=9)

    for ax in axes:
        _clean(ax)

    handles = [
        Line2D([0], [0], color=C_GT, lw=2.4, label="Ground truth"),
        Line2D([0], [0], color=C_EDGE, lw=1.6, alpha=0.7, label="Edge detection / Support ROI"),
        Line2D([0], [0], color=C_EXP, lw=2, ls="--", label="Exploration ROI"),
        Line2D([0], [0], color=C_TP, lw=2, label="Cloud TP"),
        Line2D([0], [0], color=C_FP, lw=2, ls="--", label="Cloud FP"),
        Line2D([0], [0], color=C_FN, lw=2, ls="--", label="Missed (FN)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Cloud-edge collaborative detection: %s" % rec["file"], fontsize=11, y=1.02)
    plt.tight_layout(rect=(0, 0.04, 1, 0.99))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# 图 2: 定性 4 帧 x 3 栏
# --------------------------------------------------------------------------
def fig_qualitative(recs, out_path, maxside=900):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n = len(recs)
    fig, axes = plt.subplots(n, 3, figsize=(18.5, 4.4 * n), squeeze=False)

    for i, rec in enumerate(recs):
        img, W, H = rec["img"], rec["W"], rec["H"]
        s, dw, dh = _disp(H, W, maxside)
        small = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA) if s < 1 else img

        def sc(b):
            return (b[0] * s, b[1] * s, b[2] * s, b[3] * s)

        ax = axes[i][0]
        ax.imshow(small)
        for g in rec["gts"]:
            _box(ax, sc(g), C_GT, "-", 2.0)
        for e in rec["edge"]:
            _box(ax, sc(e), C_EDGE, "-", 0.9, alpha=0.55)
        ax.set_title("%s  |  GT %d, edge %d" % (rec["file"], len(rec["gts"]), len(rec["edge"])),
                     fontsize=9)
        _clean(ax)

        a5tp = rec["res"].get("A5", {}).get("tp", -1)
        a6tp = rec["res"].get("A6", {}).get("tp", -1)
        for j, tag in enumerate(("A5", "A6")):
            ax = axes[i][j + 1]
            d = rec["a5"] if tag == "A5" else rec["a6"]
            r = rec["res"].get(tag, {})
            if d.get("ok"):
                c = cv2.resize(d["canvas"], (dw, dh), interpolation=cv2.INTER_AREA) if s < 1 else d["canvas"]
                ax.imshow(c)
            else:
                ax.imshow(small)
            if r.get("ok"):
                tp_l, fp_l, fn_l = match_tp_fp_fn(r["preds"], rec["gts"], 0.5)
                for p, g, v in tp_l:
                    _box(ax, sc(p), C_TP, "-", 1.6)
                for p in fp_l:
                    _box(ax, sc(p), C_FP, "--", 1.1)
                for g in fn_l:
                    _box(ax, sc(g), C_FN, "--", 1.3)
                sub = "TP %d / FP %d / FN %d" % (r["tp"], r["fp"], r["fn"])
            else:
                sub = "cloud failed"
            nm = "A5 support+explore (region)" if tag == "A5" else "A6 support+explore+token"
            # 失败案例标记
            if tag == "A6" and a5tp >= 0 and a6tp >= 0 and a6tp < a5tp:
                nm += "  [underperforms A5]"
            ax.set_title("%s\n%s\n%s" % (nm, _payload_2line(d).replace("\n", "  |  "), sub),
                         fontsize=8.5)
            _clean(ax)

    handles = [
        Line2D([0], [0], color=C_GT, lw=2.4, label="Ground truth"),
        Line2D([0], [0], color=C_EDGE, lw=1.2, alpha=0.7, label="Edge detection"),
        Line2D([0], [0], color=C_TP, lw=2, label="TP"),
        Line2D([0], [0], color=C_FP, lw=2, ls="--", label="FP"),
        Line2D([0], [0], color=C_FN, lw=2, ls="--", label="FN"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.005))
    plt.tight_layout(rect=(0, 0.03, 1, 0.99))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# 图 3: token zoom 四栏放大
# --------------------------------------------------------------------------
def fig_token_zoom(rec, out_path, zoom=2.6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    crop = pick_roi_crop(rec)
    if crop is None:
        print("      [token_zoom] 找不到合适的 ROI, 跳过")
        return None
    cx1, cy1, cx2, cy2 = crop
    img = rec["img"]

    panels = [
        ("(a) Original", None, None),
        ("(b) A5 region mode\n%s" % _payload_2line(rec["a5"]).replace("\n", "  |  "),
         rec["a5"].get("canvas"), rec["res"].get("A5")),
        ("(c) A6 token %.0f%%\n%s" % (rec["ratio"] * 100,
                                      _payload_2line(rec["a6"]).replace("\n", "  |  ")),
         rec["a6"].get("canvas"), rec["res"].get("A6")),
        ("(d) A6 token 100%%\n%s" % _payload_2line(rec["a6_full"]).replace("\n", "  |  "),
         rec["a6_full"].get("canvas"), rec["res"].get("A6_full")),
    ]

    ch, cw = cy2 - cy1, cx2 - cx1
    fig, axes = plt.subplots(1, 4, figsize=(4.4 * 4, 4.4 * ch / max(cw, 1) + 0.9))

    for ax, (title, canvas, res) in zip(axes, panels):
        if canvas is None:
            ax.imshow(img[cy1:cy2, cx1:cx2], interpolation="nearest")
        else:
            ax.imshow(canvas[cy1:cy2, cx1:cx2], interpolation="nearest")
        # 画 GT 与检测框 (坐标平移到 crop 坐标系)
        for g in rec["gts"]:
            if g[2] < cx1 or g[0] > cx2 or g[3] < cy1 or g[1] > cy2:
                continue
            _box(ax, (g[0] - cx1, g[1] - cy1, g[2] - cx1, g[3] - cy1), C_GT, "-", 2.0)
        if res and res.get("ok"):
            tp_l, fp_l, fn_l = match_tp_fp_fn(res["preds"], rec["gts"], 0.5)
            for p, g, v in tp_l:
                _box(ax, (p[0] - cx1, p[1] - cy1, p[2] - cx1, p[3] - cy1), C_TP, "-", 1.8)
            for p in fp_l:
                _box(ax, (p[0] - cx1, p[1] - cy1, p[2] - cx1, p[3] - cy1), C_FP, "--", 1.2)
            for g in fn_l:
                _box(ax, (g[0] - cx1, g[1] - cy1, g[2] - cx1, g[3] - cy1), C_FN, "--", 1.5)
        ax.set_xlim(0, cw)
        ax.set_ylim(ch, 0)
        ax.set_title(title, fontsize=9)
        _clean(ax)

    handles = [
        Line2D([0], [0], color=C_GT, lw=2.4, label="Ground truth"),
        Line2D([0], [0], color=C_TP, lw=2, label="TP"),
        Line2D([0], [0], color=C_FP, lw=2, ls="--", label="FP"),
        Line2D([0], [0], color=C_FN, lw=2, ls="--", label="FN"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Token budget vs fidelity on one ROI (%s, crop %dx%d)" % (rec["file"], cw, ch),
                 fontsize=11, y=1.03)
    plt.tight_layout(rect=(0, 0.05, 1, 0.99))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
class _G:
    PATCH_SIZE = 16


expE_globals = _G()


def probe_env(expE_path, expB_path):
    print("=" * 70)
    print("[PROBE] 数据集路径")
    for k, v in SEA.items():
        print("  %-6s %-95s %s" % (k, v, "OK" if os.path.exists(v) else "MISSING"))
    print("[PROBE] 加载模块")
    expE = load_mod("viz_probe_E", expE_path)
    expB = load_mod("viz_probe_B", expB_path)
    print("  expE OK, expB OK")
    print("[PROBE] expE 关键函数")
    for fn in ("make_support_rois", "generate_exploration_rois", "patchify_roi",
               "pack_and_transmit", "get_roi_centers", "evidence_aware_select"):
        print("  %-26s %s" % (fn, "YES" if hasattr(expE, fn) else "NO"))
    for k in ("EXPAND_RATIO", "LOW_RES_SCALE", "TOPK_EXPLORE", "PATCH_SIZE"):
        print("  %-26s %s" % (k, getattr(expE, k, "N/A")))
    print("[PROBE] expB 关键函数")
    for fn in ("build_edge_output", "make_support_rois", "build_combined_canvas", "calc_payload"):
        print("  %-26s %s" % (fn, "YES" if hasattr(expB, fn) else "NO"))
    print("  EdgeOutput / Detection: %s / %s"
          % (hasattr(expB, "EdgeOutput"), hasattr(expB, "Detection")))
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001/detect")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--ids", default="")
    ap.add_argument("--fig-ids", default="", help="强制指定 pipeline / token_zoom 用哪一帧")
    ap.add_argument("--out", default=os.path.join(EXP, "viz_out"))
    ap.add_argument("--token-ratio", type=float, default=0.25)
    ap.add_argument("--budget", type=int, default=115)
    ap.add_argument("--imgsz", type=int, default=0)
    ap.add_argument("--maxside", type=int, default=1500)
    ap.add_argument("--min-side", type=int, default=2000, help="选帧最小边长, 过滤小图")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--skip-class-check", action="store_true")
    args = ap.parse_args()

    expE_path = os.path.join(EXP, "expE_evidence_aware.py")
    expB_path = os.path.join(EXP, "expB_support_plus_exploration.py")

    if args.probe:
        probe_env(expE_path, expB_path)
        return

    os.makedirs(args.out, exist_ok=True)
    print("[1/6] 加载模块 ...")
    expE = load_mod("viz_expE", expE_path)
    expB = load_mod("viz_expB", expB_path)
    expE_globals.PATCH_SIZE = int(getattr(expE, "PATCH_SIZE", 16))

    print("[2/6] 读取 GT ...")
    images, cats, anns = load_gt(SEA["gt"])
    print("      %d 帧 / %d 类" % (len(images), len(cats)))

    ids = [x for x in args.ids.split(",") if x.strip()] if args.ids else None
    picks = pick_frames(images, anns, args.n, ids, args.min_side)
    if args.fig_ids:
        fids = [x for x in args.fig_ids.split(",") if x.strip()]
        forced = pick_frames(images, anns, len(fids), fids, 0)
        for f in forced:
            if f in picks:
                picks.remove(f)
            picks.insert(0, f)
        picks = picks[:max(args.n, len(forced))]
        print("      强制包含: %s" % [images[f]["file_name"] for f in forced])
    print("      选中: %s" % [images[i]["file_name"] for i in picks])

    client = None
    if not args.dry_run:
        print("[3/6] 探测云端 %s ..." % args.url)
        client = CloudClient(args.url)
        client.probe()
        if not args.skip_class_check:
            im0 = images[picks[0]]
            bgr = cv2.imread(img_path(im0))
            rgbs = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            sc0 = cv2.resize(rgbs, (640, int(640 * rgbs.shape[0] / rgbs.shape[1])))
            dets, raw = client.detect(sc0)
            names = set(d["name"] for d in dets)
            print("      服务返回类别: %s" % (names or "(无检测)"))
            if names and not (names & SEA_CLASSES):
                raise SystemExit("\n[中止] 类别不在 SEA 6 类中: %s\n" % names)
    else:
        print("[3/6] dry-run, 跳过云端")

    print("[4/6] 加载 YOLOv8s (CPU) ...")
    from ultralytics import YOLO
    yolo = YOLO(SEA["yolo"])

    print("[5/6] 逐帧运行 ...")
    recs = []
    for k, iid in enumerate(picks, 1):
        print("  --- [%d/%d] %s ---" % (k, len(picks), images[iid]["file_name"]))
        try:
            rec = run_frame(iid, images[iid], anns, yolo, expE, expB, client,
                            args.token_ratio, args.budget, args.imgsz, args.dry_run)
            recs.append(rec)
        except Exception as exc:
            print("   帧失败: %r" % (exc,))
            traceback.print_exc()

    if not recs:
        raise SystemExit("没有任何帧成功")

    # 选 pipeline 帧: A6 TP 高、FN 低
    def fig_score(r):
        a6r = r["res"].get("A6", {})
        if not a6r.get("ok"):
            return -999
        return a6r.get("tp", 0) * 2 - a6r.get("fn", 0)

    main_rec = max(recs, key=fig_score)
    if args.fig_ids:
        fids = [x.split(".")[0].strip() for x in args.fig_ids.split(",") if x.strip()]
        for r in recs:
            if str(r["file"]).split(".")[0] in fids:
                main_rec = r
                break
    print("      pipeline 帧 = %s (A6 score %d)" % (main_rec["file"], fig_score(main_rec)))

    print("[6/6] 绘图 ...")
    p1 = fig_pipeline(main_rec, os.path.join(args.out, "fig_pipeline_%s.png"
                                             % str(main_rec["file"]).split(".")[0]),
                      args.maxside)
    print("      %s" % p1)
    p2 = fig_qualitative(recs, os.path.join(args.out, "fig_qualitative.png"),
                         max(600, args.maxside // 2))
    print("      %s" % p2)
    p3 = fig_token_zoom(main_rec, os.path.join(args.out, "fig_token_zoom.png"))
    if p3:
        print("      %s" % p3)

    metrics = []
    for r in recs:
        metrics.append({
            "image_id": r["iid"], "file": r["file"], "size": [r["W"], r["H"]],
            "n_gt": len(r["gts"]), "n_edge": len(r["edge"]),
            "A5": {"stream_kb": round(kb(r["a5"].get("stream", 0)), 1),
                   "canvas_kb": round(kb(r["a5"].get("canvas_kb", 0)), 1),
                   "roi": len(r["a5"].get("sup", [])) if r["a5"].get("ok") else 0,
                   "res": {k: v for k, v in r["res"].get("A5", {}).items()
                           if k in ("ok", "tp", "fp", "fn")}},
            "A6": {"stream_kb": round(kb(r["a6"].get("stream", 0)), 1),
                   "canvas_kb": round(kb(r["a6"].get("canvas_kb", 0)), 1),
                   "roi": r["a6"].get("n_roi", 0),
                   "res": {k: v for k, v in r["res"].get("A6", {}).items()
                           if k in ("ok", "tp", "fp", "fn")}},
            "A6_full": {"stream_kb": round(kb(r["a6_full"].get("stream", 0)), 1),
                        "canvas_kb": round(kb(r["a6_full"].get("canvas_kb", 0)), 1),
                        "res": {k: v for k, v in r["res"].get("A6_full", {}).items()
                                if k in ("ok", "tp", "fp", "fn")}},
        })
    with open(os.path.join(args.out, "viz_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("      %s" % os.path.join(args.out, "viz_metrics.json"))

    # captions
    L = ["# 图注草稿 (v2, 双口径)\n",
         "> stream = 协议口径 (low-res 背景 + atlas + 4N + 4); "
         "canvas = 当前脚手架整张全分辨率画布 JPEG 的实际字节。\n"]
    for r in recs:
        a5r, a6r = r["res"].get("A5", {}), r["res"].get("A6", {})
        L.append("\n## %s  (%dx%d)\n" % (r["file"], r["W"], r["H"]))
        L.append("- GT %d / edge %d / A5 ROI %d\n"
                 % (len(r["gts"]), len(r["edge"]), len(r["a5"].get("sup", []))))
        L.append("- A5  stream %.1f KB | canvas %.1f KB | TP %s FP %s FN %s\n"
                 % (kb(r["a5"].get("stream", 0)), kb(r["a5"].get("canvas_kb", 0)),
                    a5r.get("tp", "-"), a5r.get("fp", "-"), a5r.get("fn", "-")))
        L.append("- A6  stream %.1f KB | canvas %.1f KB | TP %s FP %s FN %s\n"
                 % (kb(r["a6"].get("stream", 0)), kb(r["a6"].get("canvas_kb", 0)),
                    a6r.get("tp", "-"), a6r.get("fp", "-"), a6r.get("fn", "-")))
        L.append("- A6@100%%  stream %.1f KB | canvas %.1f KB | TP %s FP %s FN %s\n"
                 % (kb(r["a6_full"].get("stream", 0)), kb(r["a6_full"].get("canvas_kb", 0)),
                    r["res"].get("A6_full", {}).get("tp", "-"),
                    r["res"].get("A6_full", {}).get("fp", "-"),
                    r["res"].get("A6_full", {}).get("fn", "-")))
        if a5r.get("ok") and a6r.get("ok") and a6r["tp"] < a5r["tp"]:
            L.append("- **注意**: 该帧 A6 TP 低于 A5 (%d < %d), 作为失败案例保留\n"
                     % (a6r["tp"], a5r["tp"]))
    with open(os.path.join(args.out, "captions.md"), "w") as f:
        f.write("".join(L))
    print("      %s" % os.path.join(args.out, "captions.md"))


if __name__ == "__ma" + "in__":
    main()
