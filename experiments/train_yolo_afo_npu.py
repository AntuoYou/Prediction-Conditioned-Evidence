"""
Train YOLO on AFO dataset with NPU support.
Usage: python train_yolo_afo_npu.py --size s --epochs 150
"""
import os, argparse

# ===== NPU Monkey-patch fixes =====
import torch
import torch_npu
from ultralytics.engine.trainer import BaseTrainer

_orig_get_memory = BaseTrainer._get_memory
def _patched_get_memory(self, fraction=False):
    try:
        return _orig_get_memory(self, fraction)
    except (AssertionError, RuntimeError):
        return 999.0

_orig_clear_memory = BaseTrainer._clear_memory
def _patched_clear_memory(self, fraction=0.5):
    try:
        _orig_clear_memory(self, fraction)
    except (AssertionError, RuntimeError):
        pass

BaseTrainer._get_memory = _patched_get_memory
BaseTrainer._clear_memory = _patched_clear_memory

# Also patch check_amp
from ultralytics.utils import checks
_orig_check_amp = checks.check_amp
def _patched_check_amp(model):
    return False  # NPU doesn't support CUDA AMP
checks.check_amp = _patched_check_amp
# ====================================

from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="s", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--data", default="/app/datasets/afo/coco_format/afo.yaml")
    parser.add_argument("--output", default="/app/datasets/afo/weights")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Training YOLOv8{args.size} on AFO...")
    print(f"  Data: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {args.batch}")
    print(f"  Device: {args.device}")

    model = YOLO(f"yolov8{args.size}.pt")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        amp=False,  # NPU doesn't support CUDA AMP
        project=args.output,
        name=f"yolov8{args.size}_afo",
        patience=30,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        seed=42,
        # Augmentation for aerial/small objects
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5,
        shear=0.0, fliplr=0.5, mosaic=1.0, mixup=0.15,
    )

    # Test evaluation
    best_pt = os.path.join(args.output, f"yolov8{args.size}_afo", "weights", "best.pt")
    print(f"\nBest model: {best_pt}")
    if os.path.exists(best_pt):
        print("Running test evaluation...")
        model = YOLO(best_pt)
        metrics = model.val(data=args.data, split="test", imgsz=args.imgsz,
                           device=args.device, amp=False)
        print(f"\nTest Results:")
        print(f"  AP50: {metrics.box.map50:.4f}")
        print(f"  mAP:  {metrics.box.map:.4f}")

    print("Done!")

if __name__ == "__main__":
    main()
