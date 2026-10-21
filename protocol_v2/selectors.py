#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
protocol_v2/selectors.py -- A3–A8 的 ROI / patch 选择器（方案 §05）

设计原则
    * 所有配置共用同一份边缘缓存（experiments/edge_cache/edge_best_conf001.jsonl）
    * 边缘缓存只提供 boxes/scores/labels，不提供图像，因此 ROI 只依赖几何
    * A3/A4/A5 走 regional（整块裁剪），A6/A7/A8 走 packed（16x16 patch atlas）
    * 预算统一按"ROI 内 patch 总数 × ratio"，保证 A6/A7/A8 在同一预算下对比

A8 严格按方案 §05 的二维均匀定义实现：
    a = min(R, max(1, round(sqrt(k*R/C))))
    b = min(C, max(1, ceil(k/a)))
    rows = round(linspace(0, R-1, a)), cols = round(linspace(0, C-1, b))
    笛卡尔积 -> 若 |combos| > k，按 floor((i+0.5)*M/k) 取索引
    禁止"从第 1 行起连续补足"（这是旧 uniform_select 的缺陷）
"""

from __future__ import annotations

import json
import math

import numpy as np

PATCH = 16
MIN_CTX = 16
RANDOM_SEED = 20261018
N_EXPLORE = 8


# --------------------------------------------------------------------------
# 边缘缓存
# --------------------------------------------------------------------------
def load_edge_cache(path, score_thr=0.05):
    """读边缘缓存 jsonl -> {image_id: {"dets": [(x1,y1,x2,y2,score)], "w":, "h":}}"""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            dets = []
            for b, s in zip(o.get("boxes") or [], o.get("scores") or []):
                if float(s) < score_thr:
                    continue
                dets.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s)))
            out[int(o["image_id"])] = {"dets": dets, "w": int(o["w"]), "h": int(o["h"])}
    return out


# --------------------------------------------------------------------------
# ROI 生成
# --------------------------------------------------------------------------
def support_rois(dets, W, H, expand=0.1, min_ctx=MIN_CTX):
    """边缘框 -> support ROI（外扩 expand 倍边长，至少 min_ctx 像素）"""
    rois = []
    for x1, y1, x2, y2, _s in dets:
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        dx = max(bw * expand, min_ctx)
        dy = max(bh * expand, min_ctx)
        rx1 = int(max(0, math.floor(x1 - dx)))
        ry1 = int(max(0, math.floor(y1 - dy)))
        rx2 = int(min(W, math.ceil(x2 + dx)))
        ry2 = int(min(H, math.ceil(y2 + dy)))
        if rx2 > rx1 and ry2 > ry1:
            rois.append((rx1, ry1, rx2, ry2))
    return rois


def merge_rois(rois):
    """去掉被完全包含的 ROI，避免重复传输同一块像素"""
    out = []
    for r in rois:
        x1, y1, x2, y2 = r
        inside = False
        for a1, b1, a2, b2 in out:
            if a1 <= x1 and b1 <= y1 and x2 <= a2 and y2 <= b2:
                inside = True
                break
        if inside:
            continue
        out = [o for o in out
               if not (x1 <= o[0] and y1 <= o[1] and o[2] <= x2 and o[3] <= y2)]
        out.append(r)
    return out


def roi_patch_grid(roi, ps=PATCH):
    """单个 ROI 内的 patch 网格：返回 (n_rows, n_cols)"""
    x1, y1, x2, y2 = roi
    R = max(0, int(math.ceil((y2 - y1) / ps)))
    C = max(0, int(math.ceil((x2 - x1) / ps)))
    return R, C


def _roi_mask(rois, W, H, ps=PATCH):
    cols = int(math.ceil(W / ps))
    rows = int(math.ceil(H / ps))
    mask = np.zeros((rows, cols), bool)
    for x1, y1, x2, y2 in rois:
        r0, r1 = max(0, y1 // ps), min(rows, int(math.ceil(y2 / ps)))
        c0, c1 = max(0, x1 // ps), min(cols, int(math.ceil(x2 / ps)))
        if r1 > r0 and c1 > c0:
            mask[r0:r1, c0:c1] = True
    return mask


def exploration_patches(rois, W, H, n=N_EXPLORE, ps=PATCH, grid=6):
    """ROI 之外，按统一网格撒点，取离 support 中心最近的 n 个（A5 的 exploration）"""
    mask = _roi_mask(rois, W, H, ps)
    cand = []
    for gy in range(grid):
        for gx in range(grid):
            x = int((gx + 0.5) * W / grid) // ps * ps
            y = int((gy + 0.5) * H / grid) // ps * ps
            if x < 0 or y < 0 or x + ps > W or y + ps > H:
                continue
            if mask[y // ps, x // ps]:
                continue
            cand.append((x, y))
    if rois:
        cx = float(np.mean([(r[0] + r[2]) / 2.0 for r in rois]))
        cy = float(np.mean([(r[1] + r[3]) / 2.0 for r in rois]))
        cand.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
    return cand[:n]


def exploration_rois(rois, W, H, n=8, grid=(6, 10), alpha=0.3, beta=0.7):
    """
    Exploration ROI（报告 §2.2，V1-C 评分）

    将图像划分为 grid=(6,10) 的网格，每个单元格 g 计算
        score(g) = alpha * (1 - C) + beta * (1 - d / dmax)
        C    = 单元格被 support ROI 覆盖的比例
        d    = 单元格中心到最近 support ROI 中心的距离
        dmax = 图像对角线长度
    取 score 最高的 Top-K 个**网格单元**（整块矩形）作为 Exploration ROI。

    注意：返回的是网格单元的 bbox，不是 16x16 patch。
    """
    rows, cols = grid
    if not rois or rows <= 0 or cols <= 0:
        return []
    ch = H / float(rows)
    cw = W / float(cols)
    dmax = math.hypot(W, H)
    centers = [((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0) for r in rois]

    cands = []
    for gy in range(rows):
        for gx in range(cols):
            x1, y1 = gx * cw, gy * ch
            x2, y2 = min(W, (gx + 1) * cw), min(H, (gy + 1) * ch)
            area = max(1e-6, (x2 - x1) * (y2 - y1))
            covered = 0.0
            for a1, b1, a2, b2 in rois:
                iw = max(0.0, min(x2, a2) - max(x1, a1))
                ih = max(0.0, min(y2, b2) - max(y1, b1))
                covered += iw * ih
            C = min(1.0, covered / area)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            d = min(math.hypot(cx - a, cy - b) for a, b in centers) if centers else dmax
            score = alpha * (1.0 - C) + beta * (1.0 - min(1.0, d / dmax))
            cands.append((score, int(x1), int(y1), int(x2), int(y2), C, d))

    cands.sort(key=lambda t: -t[0])
    return [(x1, y1, x2, y2) for _s, x1, y1, x2, y2, _c, _d in cands[:n]]


def random_patches(rois, W, H, n=N_EXPLORE, ps=PATCH, seed=RANDOM_SEED):
    """ROI 之外随机取 n 个 patch（A4 的随机对照，固定 seed 保证可复现）"""
    mask = _roi_mask(rois, W, H, ps)
    rows, cols = mask.shape
    cand = [(c * ps, r * ps) for r in range(rows) for c in range(cols)
            if not mask[r, c] and c * ps + ps <= W and r * ps + ps <= H]
    if not cand:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cand), size=min(n, len(cand)), replace=False)
    return [cand[int(i)] for i in sorted(idx)]


def random_grid_rois(W, H, n=8, grid=(6, 10), seed=RANDOM_SEED):
    """Uniformly sample n whole cells from a fixed image grid.

    Cell IDs are row-major (row * n_cols + col). The caller records the IDs and
    seed so a random-grid comparison can be reconstructed exactly. Sampling is
    without replacement from all grid cells, as specified by the experiment
    plan; overlap with support ROIs is retained as part of the random baseline.
    """
    rows, cols = int(grid[0]), int(grid[1])
    if rows <= 0 or cols <= 0 or n <= 0:
        return [], []
    total = rows * cols
    k = min(int(n), total)
    rng = np.random.default_rng(int(seed))
    ids = sorted(int(i) for i in rng.choice(total, size=k, replace=False))
    rois = []
    for cell_id in ids:
        row, col = divmod(cell_id, cols)
        x1 = int(round(col * W / float(cols)))
        x2 = int(round((col + 1) * W / float(cols)))
        y1 = int(round(row * H / float(rows)))
        y2 = int(round((row + 1) * H / float(rows)))
        if x2 > x1 and y2 > y1:
            rois.append((x1, y1, x2, y2))
    return rois, ids


# --------------------------------------------------------------------------
# patch 选择（A6 / A7 / A8）
# --------------------------------------------------------------------------
def _uniform_indices_legacy(R, C, k):
    """A6：旧 uniform_select 行为——等间隔取，不足时从第 1 行起连续补足（保留缺陷以复现基线）"""
    if k <= 0 or R <= 0 or C <= 0:
        return []
    total = R * C
    k = min(k, total)
    step = max(1, int(round(total / float(k))))
    picked = list(range(0, total, step))[:k]
    i = 0
    while len(picked) < k and i < total:
        if i not in picked:
            picked.append(i)
        i += 1
    return [(int(x // C), int(x % C)) for x in sorted(picked)[:k]]


def _spatial_indices(R, C, k):
    """A8：方案 §05 的二维均匀"""
    if k <= 0 or R <= 0 or C <= 0:
        return []
    k = min(k, R * C)
    a = int(min(R, max(1, round(math.sqrt(k * R / float(C))))))
    # Ensure the spatial grid has at least k cells after dimension limits.
    a = min(R, max(a, int(math.ceil(k / float(C)))))
    b = int(min(C, max(1, int(math.ceil(k / float(a))))))
    ri = np.round(np.linspace(0, R - 1, a)).astype(int)
    ci = np.round(np.linspace(0, C - 1, b)).astype(int)
    combos = [(int(r), int(c)) for r in ri for c in ci]
    M = len(combos)
    if M > k:
        keep = [min(M - 1, int(math.floor((i + 0.5) * M / float(k)))) for i in range(k)]
        combos = [combos[i] for i in sorted(set(keep))]
    return combos


def _evidence_indices(gray, roi, R, C, k, sup_centers, ps=PATCH):
    """A7：证据感知——距离 support 中心越近、纹理越强，打分越高"""
    x1, y1 = roi[0], roi[1]
    if k <= 0:
        return []
    h, w = gray.shape[:2]
    scored = []
    for r in range(R):
        for c in range(C):
            px, py = x1 + c * ps, y1 + r * ps
            patch = gray[py:min(py + ps, h), px:min(px + ps, w)]
            tex = float(patch.std()) if patch.size else 0.0
            if sup_centers:
                d = min(math.hypot(px + ps / 2.0 - cx, py + ps / 2.0 - cy)
                        for cx, cy in sup_centers)
            else:
                d = 1e9
            score = 0.6 * math.exp(-d / 200.0) + 0.4 * min(tex / 64.0, 1.0)
            scored.append((score, r, c))
    scored.sort(reverse=True)
    keep = sorted((r, c) for _s, r, c in scored[:k])
    return keep


def select_patches(rgb, rois, ratio, mode, ps=PATCH, support_rois=None):
    """
    在 ROI 内按 ratio 比例挑选 patch 坐标

    mode: "uniform"(A6) | "evidence"(A7) | "spatial"(A8)
    返回 [(x, y), ...]  左上角坐标，已按 ROI 顺序排列
    """
    if not rois or ratio <= 0:
        return []
    gray = None
    if mode == "evidence":
        gray = np.asarray(rgb).mean(axis=2) if rgb.ndim == 3 else np.asarray(rgb)
        reference_rois = rois if support_rois is None else support_rois
        sup_centers = [((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0)
                       for r in reference_rois]

    grids = [roi_patch_grid(r, ps) for r in rois]
    totals = [R * C for R, C in grids]
    grand = sum(totals)
    if grand <= 0:
        return []
    budget = max(1, int(round(ratio * grand)))

    # 按各 ROI 的 patch 数比例分配预算，至少 1 个
    alloc = [max(1, int(round(budget * t / float(grand)))) for t in totals]
    while sum(alloc) > budget and max(alloc) > 1:
        alloc[alloc.index(max(alloc))] -= 1
    while sum(alloc) < budget:
        alloc[alloc.index(min(alloc))] += 1

    out = []
    for roi, (R, C), k in zip(rois, grids, alloc):
        if R <= 0 or C <= 0:
            continue
        if mode == "uniform":
            cells = _uniform_indices_legacy(R, C, k)
        elif mode == "spatial":
            cells = _spatial_indices(R, C, k)
        elif mode == "evidence":
            cells = _evidence_indices(gray, roi, R, C, k, sup_centers, ps)
        else:
            raise ValueError("未知 select mode: %s" % mode)
        x0, y0 = roi[0], roi[1]
        for r, c in cells:
            out.append((int(x0 + c * ps), int(y0 + r * ps)))

    # 去重（不同 ROI 可能产生同一 patch）
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq
