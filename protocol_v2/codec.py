#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
protocol_v2/codec.py -- 真实数据包编解码（方案 §03）

容器
    ZIP_STORED：不对 JPEG 二次压缩。固定文件顺序 + 固定时间戳 (1980,1,1,0,0,0)，
    保证同一输入产生同一字节串、同一 SHA256。

四种模式
    full_frame    manifest + image.jpg                              (A1)
    context_only  manifest + background.jpg                         (A2)
    regional      manifest + background.jpg + crop_<i>.jpg          (A3/A5)
    packed        manifest + background.jpg + atlas.jpg + coords.bin (A6/A7/A8)

硬约定（全部来自方案）
    1. packet.bin 的**文件长度**就是载荷，manifest/包头/容器开销全部计入。
    2. 不再沿用 "4N+4" 理论公式。
    3. coords.bin 为小端 uint32 坐标对 (x, y)。
    4. 解码器只读 packet，不访问原图与 GT（本模块顶部不 import 任何数据路径）。
    5. 重叠覆盖顺序 = rois 列表顺序，后写覆盖先写；顺序记入 manifest。
    6. 边界 patch 按 min(patch_size, H-y) / min(patch_size, W-x) 裁剪。
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile

import numpy as np
import cv2

VERSION = "v2"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)
MODE_FULL = "full_frame"
MODE_CTX = "context_only"
MODE_REGION = "regional"
MODE_PACKED = "packed"


# --------------------------------------------------------------------------
# 基础
# --------------------------------------------------------------------------
def _jpeg(rgb: np.ndarray, q: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


def _unjpeg(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("JPEG 解码失败")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _pack(files: list) -> bytes:
    """files: [(name, bytes), ...] 顺序即写入顺序，必须固定"""
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as z:
        for name, data in files:
            info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
    return bio.getvalue()


def _unpack(packet: bytes):
    z = zipfile.ZipFile(io.BytesIO(packet))
    names = z.namelist()
    if "manifest.json" not in names:
        raise RuntimeError("packet 缺少 manifest.json")
    man = json.loads(z.read("manifest.json").decode("utf-8"))
    if man.get("v") != VERSION:
        raise RuntimeError("protocol 版本不匹配: %r" % (man.get("v"),))
    return z, man


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _lowres(rgb: np.ndarray, scale: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (max(1, w // scale), max(1, h // scale)),
                      interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
# 编码
# --------------------------------------------------------------------------
def encode_full_frame(rgb, image_id, q=95):
    """A1: 整图 JPEG"""
    h, w = rgb.shape[:2]
    jpg = _jpeg(rgb, q)
    man = {
        "v": VERSION, "mode": MODE_FULL, "image_id": int(image_id),
        "w": int(w), "h": int(h), "q_image": int(q),
        "parts": {"image.jpg": len(jpg)},
    }
    mj = json.dumps(man, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _pack([("manifest.json", mj), ("image.jpg", jpg)])


def encode_context_only(rgb, image_id, scale=8, q=95):
    """A2: 低清背景，云端插值回原尺寸"""
    h, w = rgb.shape[:2]
    small = _lowres(rgb, scale)
    jpg = _jpeg(small, q)
    man = {
        "v": VERSION, "mode": MODE_CTX, "image_id": int(image_id),
        "w": int(w), "h": int(h), "scale": int(scale), "q_bg": int(q),
        "small_w": int(small.shape[1]), "small_h": int(small.shape[0]),
        "interp": "linear",
        "parts": {"background.jpg": len(jpg)},
    }
    mj = json.dumps(man, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _pack([("manifest.json", mj), ("background.jpg", jpg)])


def encode_regional(rgb, image_id, rois, scale=8, q_bg=95, q_roi=95,
                    roi_scale=1.0):
    """A3/A5/B3/B6: low-resolution background plus JPEG ROI crops.

    ``roi_scale`` downsamples each crop isotropically before JPEG encoding;
    the packet-only decoder restores it to the recorded ROI dimensions.
    """
    h, w = rgb.shape[:2]
    bg_jpg = _jpeg(_lowres(rgb, scale), q_bg)

    files = [("manifest.json", b"")]  # 占位，最后回填
    crops, parts = [], {"background.jpg": len(bg_jpg)}
    rois_clean = []
    for i, r in enumerate(rois):
        x1, y1, x2, y2 = [int(round(v)) for v in r[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = rgb[y1:y2, x1:x2]
        if float(roi_scale) <= 0 or float(roi_scale) > 1:
            raise ValueError("roi_scale must be in (0, 1]")
        if float(roi_scale) < 1.0:
            crop = cv2.resize(
                crop,
                (max(1, int(round(crop.shape[1] * float(roi_scale)))),
                 max(1, int(round(crop.shape[0] * float(roi_scale))))),
                interpolation=cv2.INTER_AREA,
            )
        jpg = _jpeg(crop, q_roi)
        name = "crop_%03d.jpg" % len(crops)
        crops.append((name, jpg))
        parts[name] = len(jpg)
        rois_clean.append([x1, y1, x2, y2])

    man = {
        "v": VERSION, "mode": MODE_REGION, "image_id": int(image_id),
        "w": int(w), "h": int(h), "scale": int(scale),
        "q_bg": int(q_bg), "q_roi": int(q_roi),
        "roi_scale": float(roi_scale),
        "rois": rois_clean,            # 顺序即覆盖顺序
        "n_roi": len(rois_clean),
        "parts": parts,
    }
    mj = json.dumps(man, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _pack([("manifest.json", mj), ("background.jpg", bg_jpg)] + crops)


def encode_packed(rgb, image_id, patches, scale=8, q_bg=95, q_atlas=85,
                  patch_size=16, atlas_cols=None):
    """
    A6/A7/A8: 低清背景 + patch atlas + 坐标

    patches: [(x, y), ...] 左上角坐标，顺序即 atlas 槽位顺序
    """
    h, w = rgb.shape[:2]
    ps = int(patch_size)
    bg_jpg = _jpeg(_lowres(rgb, scale), q_bg)

    n = len(patches)
    if n == 0:
        man = {"v": VERSION, "mode": MODE_PACKED, "image_id": int(image_id),
               "w": int(w), "h": int(h), "scale": int(scale),
               "q_bg": int(q_bg), "q_atlas": int(q_atlas),
               "patch_size": ps, "n_patch": 0,
               "atlas_cols": 0, "atlas_rows": 0,
               "parts": {"background.jpg": len(bg_jpg)}}
        mj = json.dumps(man, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return _pack([("manifest.json", mj), ("background.jpg", bg_jpg)])

    cols = int(atlas_cols) if atlas_cols else int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    atlas = np.full((rows * ps, cols * ps, 3), 114, dtype=np.uint8)

    coords = []
    for i, (x, y) in enumerate(patches):
        x, y = int(x), int(y)
        r, c = i // cols, i % cols
        patch = rgb[y:min(y + ps, h), x:min(x + ps, w)]
        ph, pw = patch.shape[:2]
        if ph <= 0 or pw <= 0:
            continue
        atlas[r * ps:r * ps + ph, c * ps:c * ps + pw] = patch
        coords.append((x, y))

    atlas_jpg = _jpeg(atlas, q_atlas)
    coords_bin = np.asarray(coords, dtype="<u4").tobytes()

    man = {
        "v": VERSION, "mode": MODE_PACKED, "image_id": int(image_id),
        "w": int(w), "h": int(h), "scale": int(scale),
        "q_bg": int(q_bg), "q_atlas": int(q_atlas),
        "patch_size": ps, "n_patch": len(coords),
        "atlas_cols": cols, "atlas_rows": rows,
        "parts": {"background.jpg": len(bg_jpg),
                  "atlas.jpg": len(atlas_jpg),
                  "coords.bin": len(coords_bin)},
    }
    mj = json.dumps(man, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _pack([("manifest.json", mj), ("background.jpg", bg_jpg),
                  ("atlas.jpg", atlas_jpg), ("coords.bin", coords_bin)])


# --------------------------------------------------------------------------
# 解码（只依赖 packet 自身）
# --------------------------------------------------------------------------
def decode(packet: bytes):
    """
    返回 (canvas_rgb: HxWx3 uint8, meta: dict)
    meta 含 mode / image_id / w / h / parts / 重建耗时相关字段
    """
    z, man = _unpack(packet)
    mode = man["mode"]
    W, H = int(man["w"]), int(man["h"])

    if mode == MODE_FULL:
        canvas = _unjpeg(z.read("image.jpg"))
        if canvas.shape[0] != H or canvas.shape[1] != W:
            canvas = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_LINEAR)

    elif mode == MODE_CTX:
        small = _unjpeg(z.read("background.jpg"))
        canvas = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)

    elif mode == MODE_REGION:
        small = _unjpeg(z.read("background.jpg"))
        canvas = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
        for i, (x1, y1, x2, y2) in enumerate(man["rois"]):
            crop = _unjpeg(z.read("crop_%03d.jpg" % i))
            target_w, target_h = x2 - x1, y2 - y1
            if target_w > 0 and target_h > 0:
                if crop.shape[1] != target_w or crop.shape[0] != target_h:
                    crop = cv2.resize(crop, (target_w, target_h),
                                      interpolation=cv2.INTER_LINEAR)
                canvas[y1:y2, x1:x2] = crop[:target_h, :target_w]

    elif mode == MODE_PACKED:
        small = _unjpeg(z.read("background.jpg"))
        canvas = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
        n = int(man["n_patch"])
        if n > 0:
            ps = int(man["patch_size"])
            cols = int(man["atlas_cols"])
            atlas = _unjpeg(z.read("atlas.jpg"))
            coords = np.frombuffer(z.read("coords.bin"), dtype="<u4").reshape(-1, 2)
            for i in range(min(n, coords.shape[0])):
                x, y = int(coords[i][0]), int(coords[i][1])
                r, c = i // cols, i % cols
                patch = atlas[r * ps:r * ps + ps, c * ps:c * ps + ps]
                ph, pw = min(ps, H - y), min(ps, W - x)
                if ph > 0 and pw > 0:
                    canvas[y:y + ph, x:x + pw] = patch[:ph, :pw]
    else:
        raise RuntimeError("未知 mode: %r" % (mode,))

    meta = {"mode": mode, "image_id": man.get("image_id"),
            "w": W, "h": H, "parts": man.get("parts", {}),
            "packet_bytes": len(packet), "sha256": sha256(packet)}
    return canvas, meta


def verify_packet(packet: bytes, expect_sha: str) -> None:
    got = sha256(packet)
    if got != expect_sha:
        raise RuntimeError("SHA256 不一致: got %s expect %s" % (got, expect_sha))


def part_sum(packet: bytes) -> dict:
    """校验用：各部分之和 + 容器开销是否等于总长"""
    z, man = _unpack(packet)
    parts = man.get("parts", {})
    payload = sum(int(v) for v in parts.values())
    return {"total": len(packet), "payload": payload,
            "overhead": len(packet) - payload, "parts": parts}
