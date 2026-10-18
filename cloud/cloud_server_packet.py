#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cloud/cloud_server_packet.py -- protocol_v2 云端服务（端口 8002 / card 2）

与 cloud_server.py 的区别（均对应方案 §02 的整改项）：
    1. 新增 /detect_packet：服务端自行解包，**不再接收已重建的画布**，
       因此不存在"重建画布二次 JPEG"这一步。
    2. 类别表预固化，不依赖验证集 GT 文件启动。
    3. 预测保留原始浮点，不做 round。
    4. /health 返回实际加载的权重与配置哈希。
    5. 记录真实前向张量尺寸（区分"输入尺寸"与"模型内部 resize 后尺寸"）。

/health 与 /detect 保留，供旧脚本兼容；/detect 与 /detect_packet 走同一个
detect_rgb()，保证两条入口的检测结果完全一致（方案 §03 验收项 4）。

启动：
    ASCEND_RT_VISIBLE_DEVICES=2 python cloud/cloud_server_packet.py
"""

import hashlib
import io
import os
import sys
import time
import uuid

sys.path.insert(0, "/app/Salience-DETR")

import cv2
import numpy as np
import torch
import torch_npu
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from typing import List

from util.lazy_load import Config
from util.utils import load_checkpoint, load_state_dict

from protocol_v2 import codec

CONFIG_PATH = "configs/salience_detr/salience_detr_swin_l_800_1333.py"
WEIGHT_PATH = ("/app/Salience-DETR/checkpoints/salience_detr_swin_l_800_1333/"
               "train/2026-07-23-19_59_20/best_ap.pth")

# 预固化类别表（与 SEA category_id 0..5 对应），不读 GT
SEA_CLASSES = ("ignored", "swimmer", "boat", "jetski",
               "life_saving_appliances", "buoy")

app = FastAPI(title="Salience DETR Cloud Service (protocol_v2)", version="2.0")
model = None
device = None
class_names = list(SEA_CLASSES)
meta_info = {}


class DetectionResult(BaseModel):
    bbox: List[float]
    confidence: float
    class_id: int
    class_name: str
    source: str = "cloud"


class CloudResponse(BaseModel):
    success: bool
    request_id: str
    roi_index: int
    detections: List[DetectionResult]
    inference_time_ms: float
    cloud_model: str = "SalienceDETR-SwinL-AscendNPU"


def _file_sha(path, algo="sha256"):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@app.on_event("startup")
async def load_model():
    global model, device, meta_info
    device = torch.device("npu:0")
    print("[Cloud] 推理设备: %s" % device)

    from util.misc import encode_labels

    model = Config(CONFIG_PATH).model
    ckpt = load_checkpoint(WEIGHT_PATH)
    ckpt = ckpt["model"] if "model" in ckpt else ckpt
    load_state_dict(model, ckpt)
    model.register_buffer("_classes_", torch.tensor(encode_labels(SEA_CLASSES)))
    model.to(device).eval()

    meta_info = {
        "weight_path": WEIGHT_PATH,
        "weight_sha256": _file_sha(WEIGHT_PATH),
        "config_path": CONFIG_PATH,
        "config_sha256": _file_sha(os.path.join("/app/Salience-DETR", CONFIG_PATH)),
        "classes": list(SEA_CLASSES),
        "protocol_version": codec.VERSION,
        "device": str(device),
    }
    print("[Cloud] 权重 sha256 = %s" % meta_info["weight_sha256"][:16])
    print("[Cloud] 类别: %d -> %s" % (len(SEA_CLASSES), list(SEA_CLASSES)))

    with torch.no_grad():
        _ = model([torch.randn(3, 800, 1333).to(device)])
    print("[Cloud] 模型就绪")


def _postprocess(out, orig_w, orig_h, conf_thr):
    """保留原始浮点，不做 round（方案 §02 / §08）"""
    boxes = out.get("boxes")
    scores = out.get("scores")
    labels = out.get("labels")
    if boxes is None or scores is None or len(scores) == 0:
        return []
    dets = []
    for i in range(len(scores)):
        s = float(scores[i])
        if s < conf_thr:
            continue
        x1, y1, x2, y2 = [float(v) for v in boxes[i].tolist()]
        x1 = max(0.0, min(x1, float(orig_w)))
        x2 = max(0.0, min(x2, float(orig_w)))
        y1 = max(0.0, min(y1, float(orig_h)))
        y2 = max(0.0, min(y2, float(orig_h)))
        if x2 <= x1 or y2 <= y1:
            continue
        cid = int(labels[i].item())
        dets.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": s,
            "class_id": cid,
            "class_name": class_names[cid] if cid < len(class_names) else "class_%d" % cid,
            "source": "cloud",
        })
    return dets


def detect_rgb(rgb: np.ndarray, conf_thr: float = 0.05):
    """
    唯一的推理入口。rgb 为 HxWx3 uint8 RGB。
    返回 (dets, timing)。timing 记录真实张量尺寸，供方案 §02 "实际前向尺寸"核对。
    """
    t0 = time.time()
    h, w = rgb.shape[:2]
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).float() / 255.0
    tensor = tensor.permute(2, 0, 1)

    with torch.no_grad():
        outputs = model([tensor.to(device)])
    out = {k: v.to("cpu") for k, v in outputs[0].items()}
    dets = _postprocess(out, w, h, conf_thr)

    bmax = [0.0, 0.0]
    if dets:
        bmax = [max(d["bbox"][2] for d in dets), max(d["bbox"][3] for d in dets)]

    timing = {
        "input_tensor_shape": [int(tensor.shape[1]), int(tensor.shape[2])],
        "bbox_max_xy": bmax,
        "ms": round((time.time() - t0) * 1000.0, 2),
    }
    return dets, timing


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "SalienceDETR-SwinL",
            "protocol": codec.VERSION, "meta": meta_info}


@app.post("/detect", response_model=CloudResponse)
async def detect(file: UploadFile = File(...), roi_index: int = Form(default=0),
                 conf_threshold: float = Form(default=0.05)):
    rid = str(uuid.uuid4())[:8]
    data = await file.read()
    if not data:
        raise HTTPException(400, "空图像")
    try:
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(400, "图像解码失败")
        dets, tm = detect_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), conf_threshold)
        return CloudResponse(success=True, request_id=rid, roi_index=roi_index,
                             detections=[DetectionResult(**d) for d in dets],
                             inference_time_ms=tm["ms"])
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/detect_packet", response_model=CloudResponse)
async def detect_packet(file: UploadFile = File(...),
                        roi_index: int = Form(default=0),
                        conf_threshold: float = Form(default=0.05),
                        expect_sha256: str = Form(default="")):
    """
    接收 packet.bin -> 校验 SHA256 -> 解码 -> 推理。
    解码器只看到数据包，不访问原图与 GT（方案 §03 验收项 1）。
    """
    rid = str(uuid.uuid4())[:8]
    t_recv = time.time()
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "空数据包")

    if expect_sha256:
        got = codec.sha256(raw)
        if got != expect_sha256:
            raise HTTPException(400, "SHA256 不一致: got %s" % got)

    try:
        canvas, cmeta = codec.decode(raw)
        dets, tm = detect_rgb(canvas, conf_threshold)
        total = round((time.time() - t_recv) * 1000.0, 2)
        return CloudResponse(success=True, request_id=rid, roi_index=roi_index,
                             detections=[DetectionResult(**d) for d in dets],
                             inference_time_ms=tm["ms"])
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, "%s: %s" % (type(e).__name__, e))


if __name__ == "__main__":
    _port = int(os.environ.get("PV2_PORT", "8002"))
    print("=" * 56)
    print("  Salience DETR Cloud Service (protocol_v2)")
    print("  Ascend 910B4 NPU | port %d" % _port)
    print("=" * 56)
    uvicorn.run(app, host="0.0.0.0", port=_port)
