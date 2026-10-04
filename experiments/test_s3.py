#!/usr/bin/env python3
"""最小测试：S3 Full image upper bound"""
import json, os, io, time
import numpy as np
import requests
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLOUD_URL = "http://localhost:8000/detect"
GT_PATH = "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"
IMG_DIR = "/app/data/seadronessee/compressed_version/val2017/"
YOLO_PATH = "/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt"

def send_cloud(img_np, conf=0.05):
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format='JPEG', quality=95)
    buf.seek(0)
    try:
        r = requests.post(CLOUD_URL,
            files={'file': ('img.jpg', buf, 'image/jpeg')},
            data={'conf_threshold': str(conf)}, timeout=60)
        if r.status_code == 200:
            return r.json().get('detections', [])
    except Exception as e:
        print(f"  [WARN] {e}")
    return []

def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2-x1) * (y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)

def nms_fuse(edge_high, cloud_dets, iou_thresh=0.5):
    all_dets = []
    for d in edge_high:
        all_dets.append(d)
    for d in cloud_dets:
        all_dets.append(d)
    all_dets.sort(key=lambda x: x[4], reverse=True)
    keep = []
    for det in all_dets:
        ok = True
        for k in keep:
            if det[5] == k[5] and compute_iou(det[:4], k[:4]) > iou_thresh:
                ok = False
                break
        if ok:
            keep.append(det)
    return keep

def evaluate(results_list, name, img_ids_used):
    """照搬 expF_supplementary.py 的 evaluate 函数"""
    if not results_list:
        return {"AP50": 0, "AP": 0, "AR": 0, "APsmall": 0, "num_dets": 0}
    tmp = f"/tmp/test_s3_{name}.json"
    with open(tmp, "w") as f:
        json.dump(results_list, f)
    try:
        coco_gt = COCO(GT_PATH)
        coco_dt = coco_gt.loadRes(tmp)
        e = COCOeval(coco_gt, coco_dt, 'bbox')
        e.params.imgIds = list(img_ids_used)
        e.evaluate()
        e.accumulate()
        e.summarize()
        return {
            "AP50": float(e.stats[1]),
            "AP": float(e.stats[0]),
            "AR": float(e.stats[8]),
            "APsmall": float(e.stats[3]),
            "num_dets": len(results_list),
        }
    except Exception as ex:
        print(f"  [EVAL ERROR] {ex}")
        return {"AP50": 0, "AP": 0, "AR": 0, "APsmall": 0, "num_dets": len(results_list)}

def main():
    print("Loading GT...")
    with open(GT_PATH) as f:
        gt_coco = json.load(f)
    images_info = gt_coco["images"][:5]  # 5 images for test
    print(f"Images: {len(images_info)}")

    print("Loading YOLO...")
    model = YOLO(YOLO_PATH)

    results_coco = []
    img_ids_used = set()
    total_miss = 0
    total_recovered = 0
    total_gt = 0

    # GT index
    gt_by_img = {}
    for ann in gt_coco["annotations"]:
        img_id = ann["image_id"]
        x, y, w, h = ann["bbox"]
        bbox_xyxy = [x, y, x+w, y+h]
        gt_by_img.setdefault(img_id, []).append((bbox_xyxy, ann["category_id"]))

    for img_info in tqdm(images_info, desc="S3 Full Image"):
        img_id = img_info["id"]
        fname = img_info["file_name"]
        img_path = os.path.join(IMG_DIR, fname)
        img = np.array(Image.open(img_path).convert("RGB"))

        gt_list = gt_by_img.get(img_id, [])
        gt_boxes = [g[0] for g in gt_list]
        gt_cats = [g[1] for g in gt_list]
        total_gt += len(gt_boxes)

        # Edge detection
        r = model(img_path, conf=0.01, verbose=False)[0]
        edge_dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().tolist()
            confs = r.boxes.conf.cpu().numpy().tolist()
            cls_ids = r.boxes.cls.cpu().numpy().tolist()
            for i in range(len(xyxy)):
                edge_dets.append([
                    float(xyxy[i][0]), float(xyxy[i][1]),
                    float(xyxy[i][2]), float(xyxy[i][3]),
                    float(confs[i]), int(cls_ids[i]) + 1  # YOLO 0-indexed -> 1-indexed for COCO
                ])

        edge_high = [d for d in edge_dets if d[4] >= 0.7]

        # Cloud detection (full image)
        cloud_raw = send_cloud(img, 0.05)

        # Convert cloud dets to [x1,y1,x2,y2,score,class]
        cloud_dets = []
        for d in cloud_raw:
            b = d["bbox"]
            cloud_dets.append([
                float(b[0]), float(b[1]),
                float(b[2]), float(b[3]),
                float(d["confidence"]), int(d["class_id"])
            ])

        # Fuse
        fused = nms_fuse(edge_high, cloud_dets)

        # COCO predictions
        for d in fused:
            results_coco.append({
                "image_id": int(img_id),
                "category_id": int(d[5]),
                "bbox": [float(d[0]), float(d[1]), float(d[2]-d[0]), float(d[3]-d[1])],
                "score": float(d[4]),
            })
        img_ids_used.add(int(img_id))

        # Miss recovery
        for gt in gt_boxes:
            hit_edge = False
            for ed in edge_dets:
                if compute_iou(gt, ed[:4]) >= 0.5:
                    hit_edge = True
                    break
            if not hit_edge:
                total_miss += 1
                for fd in fused:
                    if compute_iou(gt, fd[:4]) >= 0.5:
                        total_recovered += 1
                        break

    # Evaluate
    metrics = evaluate(results_coco, "fullimg", img_ids_used)

    print(f"\n{'='*60}")
    print(f"S3 Full Image Results (5 images test)")
    print(f"{'='*60}")
    print(f"  AP50:    {metrics['AP50']:.4f}")
    print(f"  mAP:     {metrics['AP']:.4f}")
    print(f"  APsmall: {metrics['APsmall']:.4f}")
    print(f"  Dets:    {metrics['num_dets']}")
    print(f"  Miss recovery: {total_recovered}/{total_miss} = {total_recovered/max(total_miss,1):.4f}")

if __name__ == "__main__":
    main()
