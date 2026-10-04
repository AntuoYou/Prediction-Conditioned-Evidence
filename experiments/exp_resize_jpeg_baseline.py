"""
Resize + JPEG Baseline for SeaDronesSee
=========================================
Usage: python exp_resize_jpeg_baseline.py --phase both
"""
import os, json, io, argparse
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import requests
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

IMG_DIR = "/app/data/seadronessee/compressed_version/val2017/"
GT_PATH = "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"
CLOUD_URL = "http://localhost:8000/detect"
CLOUD_TIMEOUT = 120
OUTPUT_DIR = "/app/Salience-DETR/experiments"

TARGET_PAYLOADS = {"low": 87, "high": 311}
RESIZE_SCALES_SEARCH = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
JPEG_QUALITIES_SEARCH = [40, 50, 60, 70, 80, 85, 90, 95]
NUM_SEARCH_IMAGES = 50
RANDOM_SEED = 42

def compute_resize_jpeg(img, scale, quality):
    H, W = img.shape[:2]
    new_w, new_h = max(1, int(W*scale)), max(1, int(H*scale))
    small = cv2.resize(img, (new_w, new_h))
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).save(buf, format="JPEG", quality=quality)
    payload = len(buf.getvalue())
    buf.seek(0)
    restored_small = cv2.cvtColor(np.array(Image.open(buf)), cv2.COLOR_RGB2BGR)
    restored = cv2.resize(restored_small, (W, H), interpolation=cv2.INTER_LINEAR)
    return restored, payload

def cloud_detect(canvas_rgb, conf=0.05):
    """Send RGB image to cloud detector (matching expE format)."""
    buf = io.BytesIO()
    Image.fromarray(canvas_rgb).save(buf, format='JPEG', quality=95)
    buf.seek(0)
    try:
        r = requests.post(CLOUD_URL,
            files={'file': ('img.jpg', buf, 'image/jpeg')},
            data={'conf_threshold': str(conf)}, timeout=CLOUD_TIMEOUT)
        if r.status_code == 200:
            dets = []
            for d in r.json().get('detections', []):
                # Handle both [x1,y1,x2,y2] and {"bbox": [...]} formats
                if isinstance(d, list):
                    x1,y1,x2,y2,conf,cls = d[0],d[1],d[2],d[3],d[4],d[5]
                else:
                    b = d["bbox"]
                    x1,y1,x2,y2 = b[0],b[1],b[2],b[3]
                    conf = d.get("confidence", d.get("score", 0.5))
                    cls = d.get("class_id", d.get("category_id", 0))
                dets.append([float(x1),float(y1),float(x2),float(y2),float(conf),int(cls)])
            return dets
    except Exception as e:
        print(f"  Cloud error: {e}")
    return []

def evaluate(results_dict, img_ids):
    coco_results = []
    for img_id, dets in results_dict.items():
        for det in dets:
            x1,y1,x2,y2,conf,cls_id = det
            coco_results.append({
                "image_id": int(img_id), "category_id": int(cls_id),
                "bbox": [float(x1),float(y1),float(x2-x1),float(y2-y1)],
                "score": float(conf),
            })
    coco_gt = COCO(GT_PATH)
    if not coco_results:
        return {"AP50":0,"AP":0,"AP75":0,"APS":0,"AR":0}
    coco_dt = coco_gt.loadRes(coco_results)
    e = COCOeval(coco_gt, coco_dt, "bbox")
    e.params.imgIds = list(img_ids)
    e.evaluate()
    e.accumulate()
    e.summarize()
    return {"AP50":e.stats[1],"AP":e.stats[0],"AP75":e.stats[2],"APS":e.stats[3],"AR":e.stats[8]}

def phase_search():
    print(f"\n{'='*60}\nPhase 1: Search resize+JPEG configs\n{'='*60}\n")
    with open(GT_PATH) as f:
        gt = json.load(f)
    all_imgs = gt["images"]
    np.random.seed(RANDOM_SEED)
    idx = np.random.choice(len(all_imgs), min(NUM_SEARCH_IMAGES, len(all_imgs)), replace=False)
    search_imgs = [all_imgs[i] for i in idx]

    results = []
    for scale in RESIZE_SCALES_SEARCH:
        for quality in JPEG_QUALITIES_SEARCH:
            total, n_ok = 0, 0
            for img_info in search_imgs:
                p = os.path.join(IMG_DIR, img_info["file_name"])
                if not os.path.exists(p): continue
                img = cv2.imread(p)
                if img is None: continue
                _, payload = compute_resize_jpeg(img, scale, quality)
                total += payload
                n_ok += 1
            if n_ok > 0:
                avg = total/n_ok/1024
                results.append((scale, quality, avg))
                print(f"  s={scale:.2f} q={quality:2d}: {avg:7.1f} KB")

    best = {}
    for name, target in TARGET_PAYLOADS.items():
        best_cfg, best_diff = None, float('inf')
        for s,q,p in results:
            diff = abs(p-target)
            if p > target: diff *= 1.1
            if diff < best_diff:
                best_diff, best_cfg = diff, (s,q,p)
        best[name] = {"resize_scale":best_cfg[0],"jpeg_quality":best_cfg[1],
                      "avg_payload_kb":best_cfg[2],"target_kb":target}
        print(f"\n  Best {name} (target={target}KB): s={best_cfg[0]:.2f}, q={best_cfg[1]}, {best_cfg[2]:.1f}KB")

    out = os.path.join(OUTPUT_DIR, "resize_jpeg_search.json")
    with open(out, "w") as f:
        json.dump({"targets":TARGET_PAYLOADS,
                   "all_configs":[{"resize_scale":s,"jpeg_quality":q,"avg_payload_kb":p} for s,q,p in results],
                   "best_configs":best}, f, indent=2)
    print(f"\n  Saved: {out}")
    return best

def phase_eval(best_configs):
    print(f"\n{'='*60}\nPhase 2: Full evaluation\n{'='*60}\n")
    with open(GT_PATH) as f:
        all_imgs = json.load(f)["images"]
    print(f"Total images: {len(all_imgs)}")

    all_metrics = {}
    for name, cfg in best_configs.items():
        s, q, target = cfg["resize_scale"], cfg["jpeg_quality"], cfg["target_kb"]
        print(f"\n--- {name} (target={target}KB, s={s}, q={q}) ---")
        results_dict, img_ids, total_pl = {}, [], 0
        for info in tqdm(all_imgs, desc=f"RJ-{name}"):
            p = os.path.join(IMG_DIR, info["file_name"])
            if not os.path.exists(p): continue
            img = cv2.imread(p)
            if img is None: continue
            canvas, pl = compute_resize_jpeg(img, s, q)
            total_pl += pl
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            dets = cloud_detect(rgb)
            results_dict[info["id"]] = dets
            img_ids.append(info["id"])
        m = evaluate(results_dict, img_ids)
        n = max(1, len(img_ids))
        m["payload_kb"] = total_pl/n/1024
        m["resize_scale"], m["jpeg_quality"], m["target_kb"] = s, q, target
        all_metrics[f"resize_jpeg_{name}"] = m
        print(f"  AP50={m['AP50']:.4f}, mAP={m['AP']:.4f}, APS={m['APS']:.4f}, payload={m['payload_kb']:.1f}KB")

    out = os.path.join(OUTPUT_DIR, "resize_jpeg_baseline.json")
    with open(out, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Saved: {out}")
    print(f"\n{'='*70}")
    print(f"{'Method':<25} {'AP50':>7} {'mAP':>7} {'APS':>7} {'KB':>8}")
    print(f"{'-'*70}")
    for n, m in all_metrics.items():
        print(f"{n:<25} {m['AP50']:>7.4f} {m['AP']:>7.4f} {m['APS']:>7.4f} {m['payload_kb']:>8.1f}")
    print(f"{'='*70}")
    return all_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="both", choices=["search","eval","both"])
    args = parser.parse_args()
    print("Resize + JPEG Baseline (SeaDronesSee)")

    best = None
    if args.phase in ["search", "both"]:
        best = phase_search()
    if args.phase in ["eval", "both"]:
        if best is None:
            p = os.path.join(OUTPUT_DIR, "resize_jpeg_search.json")
            with open(p) as f: best = json.load(f)["best_configs"]
        phase_eval(best)
    print("\nDone!")

if __name__ == "__main__":
    main()
