import time
import torch
from ultralytics import YOLO

WEIGHTS = "/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt"
INPUT_SIZE = 640
DEVICE = "npu:0"           # 无 NPU 改 "cpu"

model = YOLO(WEIGHTS)
model.info()

dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
model.model = model.model.to(DEVICE).eval()

with torch.no_grad():
    for _ in range(50):
        _ = model.model(dummy)
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    elif DEVICE.startswith("npu"):
        torch.npu.synchronize()

    n = 100
    t0 = time.perf_counter()
    for _ in range(n):
        _ = model.model(dummy)
        if DEVICE.startswith("cuda"):
            torch.cuda.synchronize()
        elif DEVICE.startswith("npu"):
            torch.npu.synchronize()
    t1 = time.perf_counter()

print(f"\n[Latency] batch=1, {DEVICE}: {(t1-t0)/n*1000:.2f} ms/frame")
print(f"[Input] {INPUT_SIZE}x{INPUT_SIZE}, 不含预处理/后处理")
