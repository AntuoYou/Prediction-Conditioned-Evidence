#!/usr/bin/env python3
import os, io, json, time
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
OUTPUT_DIR = "/app/Salience-DETR/experiments/pce_results/results"
DEVICE = "npu:0"          # 无 NPU 改 "cpu"

EDGE_CONF_SPLIT = 0.7
CLOUD_CONF = 0.05
PATCH_GRAY = 110

def send_cloud(img_np, conf=CLOUD_CONF):
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format="JPEG", quality=95)
    buf.seek(0)
    try:
        r = requests.post(CLOUD_URL, files={"file": ("img.jpg", buf, "image/jpeg")},
                          data={"conf_threshold": str(conf)}, timeout=60)
        if r.status_code == 200:
            return r.json().get("detections", [])
    except Exception:
        pass
    return []

def cloud_to_list(cloud_raw):
    res = []
    for d in cloud_raw:
        b = d["bbox"]
        res.append([float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                    float(d["confidence"]), int(d["class_id"])])
    return res

def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2-x1)*(y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter+1e-6)

def nms_fuse(edge_high, cloud_dets, iou_thresh=0.5):
    dets = list(edge_high)+list(cloud_dets)
    dets.sort(key=lambda x: x[4], reverse=True)
    keep = []
    for d in dets:
        if all(not (d[5]==k[5] and compute_iou(d[:4], k[:4])>iou_thresh) for k in keep):
            keep.append(d)
    return keep

def build_exploration_canvas(img, exploration_boxes):
    h, w = img.shape[:2]
    canvas = np.full((h, w, 3), PATCH_GRAY, dtype=np.uint8)
    for b in exploration_boxes:
        x1, y1, x2, y2 = [int(v) for v in b[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            canvas[y1:y2, x1:x2] = img[y1:y2, x1:x2]
    return canvas

def evaluate(results_list, img_ids_used):
    if not results_list:
        return {"AP50":0,"AP":0,"AP75":0,"APsmall":0,"num_dets":0}
    tmp = f"/tmp/exploration_{int(time.time())}.json"
    with open(tmp,"w") as f: json.dump(results_list, f)
    try:
        coco_gt = COCO(GT_PATH); coco_dt = coco_gt.loadRes(tmp)
        e = COCOeval(coco_gt, coco_dt, "bbox")
        e.params.imgIds = list(img_ids_used)
        e.evaluate(); e.accumulate(); e.summarize()
        return {"AP50":float(e.stats[1]),"AP":float(e.stats[0]),"AP75":float(e.stats[2]),
                "APsmall":float(e.stats[3]),"APmedium":float(e.stats[4]),
                "APlarge":float(e.stats[5]),"num_dets":len(results_list)}
    except Exception as ex:
        print(f"[EVAL ERROR] {ex}")
        return {"AP50":0,"AP":0,"AP75":0,"APsmall":0,"num_dets":0}
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    max_images = None if args.full else (args.max_images or 200)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(GT_PATH) as f: gt = json.load(f)
    images = gt["images"]
    if max_images: images = images[:max_images]
    print(f"Images: {len(images)} | mode=exploration-only (E2)")
    model = YOLO(YOLO_PATH)
    results_coco = []; img_ids_used = set()
    for img_info in tqdm(images, desc="exploration-only"):
        img_id = img_info["id"]; fname = img_info["file_name"]
        img_path = os.path.join(IMG_DIR, fname)
        img = np.array(Image.open(img_path).convert("RGB"))
        r = model(img_path, conf=0.01, verbose=False, device=DEVICE)[0]
        edge_dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().tolist()
            confs = r.boxes.conf.cpu().numpy().tolist()
            cls = r.boxes.cls.cpu().numpy().tolist()
            for i in range(len(xyxy)):
                edge_dets.append([float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]),
                                  float(xyxy[i][3]), float(confs[i]), int(cls[i]) + 1])
        edge_high = [d for d in edge_dets if d[4] >= EDGE_CONF_SPLIT]
        exploration_boxes = [d for d in edge_dets if d[4] < EDGE_CONF_SPLIT]
        canvas = build_exploration_canvas(img, exploration_boxes)
        cloud_raw = send_cloud(canvas, CLOUD_CONF)
        cloud_dets = cloud_to_list(cloud_raw)
        fused = nms_fuse(edge_high, cloud_dets)
        for d in fused:
            results_coco.append({"image_id":int(img_id),"category_id":int(d[5]),
                "bbox":[float(d[0]),float(d[1]),float(d[2]-d[0]),float(d[3]-d[1])],
                "score":float(d[4])})
        img_ids_used.add(int(img_id))
    metrics = evaluate(results_coco, img_ids_used)
    metrics["mode"] = "exploration-only"
    metrics["num_images"] = len(images)
    print("\n=== Exploration-only (E2) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    out = os.path.join(OUTPUT_DIR, "exploration_only_results.json")
    with open(out, "w") as f: json.dump(metrics, f, indent=2)
    print(f"[SAVED] {out}")

if __name__ == "__main__":
    main()
