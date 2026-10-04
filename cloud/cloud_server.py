"""云端 Salience DETR 推理服务 — Ascend 910B4 NPU"""

import time, uuid, sys, os
sys.path.insert(0, '/app/Salience-DETR')

import cv2, numpy as np, torch, torch_npu
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

from util.lazy_load import Config
from util.utils import load_checkpoint, load_state_dict

app = FastAPI(title="Salience DETR Cloud Service", version="1.0")
model = None
device = None
class_names = []

CONFIG_PATH = "configs/salience_detr/salience_detr_swin_l_800_1333.py"
WEIGHT_PATH = "/app/Salience-DETR/checkpoints/salience_detr_swin_l_800_1333/train/2026-07-23-19_59_20/best_ap.pth"

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

@app.on_event("startup")
async def load_model():
    global model, device, class_names
    device = torch.device("npu:0")
    print(f"[Cloud] 推理设备: {device}")

    model = Config(CONFIG_PATH).model
    checkpoint = load_checkpoint(WEIGHT_PATH)
    checkpoint = checkpoint["model"] if "model" in checkpoint else checkpoint
    load_state_dict(model, checkpoint)

    from pycocotools.coco import COCO
    from util.misc import encode_labels
    coco = COCO("/app/data/seadronessee/compressed_version/annotations/instances_val2017.json")
    cat_ids = list(range(max(coco.cats.keys()) + 1))
    classes = tuple(coco.cats.get(c, {"name": "none"})["name"] for c in cat_ids)
    model.register_buffer("_classes_", torch.tensor(encode_labels(classes)))
    class_names = list(classes)
    print(f"[Cloud] 类别: {len(class_names)}, 示例: {class_names[:5]}")

    model.to(device).eval()
    print("[Cloud] NPU 预热中...")
    with torch.no_grad():
        dummy = torch.randn(3, 800, 1333)
        _ = model([dummy.to(device)])
    print("[Cloud] 模型就绪")

def preprocess(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("图像解码失败")
    orig_h, orig_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1)
    return tensor, {"orig_size": (orig_w, orig_h)}

def postprocess(output_dict, orig_size, conf_thr=0.3):
    orig_w, orig_h = orig_size
    boxes = output_dict.get("boxes")
    scores = output_dict.get("scores")
    labels = output_dict.get("labels")
    if boxes is None or scores is None or len(scores) == 0:
        return []

    # DEBUG 输出已关闭，避免批量推理时刷屏
    # print(f"[DEBUG] boxes[0]: {boxes[0].tolist()}")
    # print(f"[DEBUG] orig_size: {orig_size}")
    # print(f"[DEBUG] output keys: {list(output_dict.keys())}")

    dets = []
    for i in range(len(scores)):
        if scores[i] < conf_thr:
            continue

        box = boxes[i].tolist()

        # 模型输出 boxes 已经对应当前输入图像坐标，不再除以 scale。
        # 这里只做边界裁剪，避免少量越界框影响评估。
        x1, y1, x2, y2 = box
        x1 = max(0.0, min(float(x1), float(orig_w)))
        x2 = max(0.0, min(float(x2), float(orig_w)))
        y1 = max(0.0, min(float(y1), float(orig_h)))
        y2 = max(0.0, min(float(y2), float(orig_h)))

        if x2 <= x1 or y2 <= y1:
            continue

        cid = int(labels[i].item())
        cname = class_names[cid] if cid < len(class_names) else f"class_{cid}"
        dets.append({"bbox": [round(x1,1), round(y1,1), round(x2,1), round(y2,1)],
                      "confidence": round(scores[i].item(),4), "class_id": cid,
                      "class_name": cname})
    return dets

@app.get("/health")
async def health():
    return {"status": "healthy", "device": str(device), "model": "SalienceDETR-SwinL"}

@app.post("/detect", response_model=CloudResponse)
async def detect(file: UploadFile = File(...),
                 roi_index: int = Form(default=0),
                 conf_threshold: float = Form(default=0.3)):
    rid = str(uuid.uuid4())[:8]
    t0 = time.time()
    data = await file.read()
    if not data:
        raise HTTPException(400, "空图像")
    try:
        tensor, sinfo = preprocess(data)
        with torch.no_grad():
            outputs = model([tensor.to(device)])
        output_cpu = {k: v.to("cpu") for k, v in outputs[0].items()}
        dets = postprocess(output_cpu, sinfo["orig_size"], conf_threshold)
        ms = (time.time() - t0) * 1000
        return CloudResponse(success=True, request_id=rid, roi_index=roi_index,
                            detections=[DetectionResult(**d) for d in dets],
                            inference_time_ms=round(ms, 2))
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    print("=" * 50)
    print("  Salience DETR Cloud Service (Real Model)")
    print("  Ascend 910B4 NPU")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
