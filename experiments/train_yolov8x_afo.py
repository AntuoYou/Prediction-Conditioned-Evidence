"""
Train YOLOv8x on AFO dataset (cloud detector - quick validation).
Usage: python train_yolov8x_afo.py
"""
import os
from ultralytics import YOLO

DATA_YAML = "/app/datasets/afo/coco_format/afo.yaml"
OUTPUT_DIR = "/app/datasets/afo/weights"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Training YOLOv8x on AFO (cloud detector)...")
model = YOLO("yolov8x.pt")

results = model.train(
    data=DATA_YAML,
    epochs=150,
    imgsz=640,
    batch=16,
    device="npu:0",
    project=OUTPUT_DIR,
    name="yolov8x_afo",
    patience=30,
    save=True,
    save_period=10,
    val=True,
    plots=True,
    seed=42,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.15,
)

best_pt = os.path.join(OUTPUT_DIR, "yolov8x_afo", "weights", "best.pt")
print(f"\nBest model: {best_pt}")
print("Running test evaluation...")
model = YOLO(best_pt)
metrics = model.val(data=DATA_YAML, split="test", imgsz=640, device="npu:0")
print(f"\nTest Results:")
print(f"  AP50: {metrics.box.map50:.4f}")
print(f"  mAP:  {metrics.box.map:.4f}")
print("Done!")
