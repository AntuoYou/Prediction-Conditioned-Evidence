#!/usr/bin/env python3
import os, json
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm

GT_PATH = "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"
IMG_DIR = "/app/data/seadronessee/compressed_version/val2017/"
YOLO_PATH = "/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt"
DEVICE = "npu:0"          # 若 ultralytics 不认 npu，改成 "cpu"
SCORE_THRESHOLD = 0.01
NMS_IOU = 0.7

def main():
    model = YOLO(YOLO_PATH)
    with open(GT_PATH) as f:
        gt = json.load(f)
    images = gt["images"]
    total_boxes = 0
    for img_info in tqdm(images, desc="count"):
        img_path = os.path.join(IMG_DIR, img_info["file_name"])
        r = model(img_path, conf=SCORE_THRESHOLD, iou=NMS_IOU, verbose=False, device=DEVICE)[0]
        if r.boxes is not None:
            total_boxes += len(r.boxes)
    print(f"Total images: {len(images)}")
    print(f"Total edge detections (conf>={SCORE_THRESHOLD}, NMS iou={NMS_IOU}): {total_boxes}")
    print(f"Average boxes per image: {total_boxes/len(images):.2f}")

if __name__ == "__main__":
    main()
