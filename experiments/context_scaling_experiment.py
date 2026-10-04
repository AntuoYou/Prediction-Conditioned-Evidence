import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLOUD_URL = "http://localhost:8000/detect"

def expand_bbox_xywh(bbox_xywh, ratio, img_w, img_h):
    x, y, w, h = bbox_xywh
    cx = x + w / 2.0
    cy = y + h / 2.0
    nw = w * ratio
    nh = h * ratio
    rx1 = max(0.0, cx - nw / 2.0)
    ry1 = max(0.0, cy - nh / 2.0)
    rx2 = min(float(img_w), cx + nw / 2.0)
    ry2 = min(float(img_h), cy + nh / 2.0)
    return [int(rx1), int(ry1), int(rx2), int(ry2)]

def encode_image(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()

def detect_cloud(img_bytes, orig_w, orig_h, timeout=30, roi_offset=None, roi_scale=None):
    files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
    data = {"conf_threshold": "0.05"}
    try:
        resp = requests.post(CLOUD_URL, files=files, data=data, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        return []

    dets = []
    for d in result.get("detections", []):
        bbox = d.get("bbox", {})
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        elif isinstance(bbox, dict):
            x1, y1, x2, y2 = float(bbox.get("x1", 0)), float(bbox.get("y1", 0)), float(bbox.get("x2", 0)), float(bbox.get("y2", 0))
        else:
            continue

        x1 = max(0, min(x1, orig_w))
        x2 = max(0, min(x2, orig_w))
        y1 = max(0, min(y1, orig_h))
        y2 = max(0, min(y2, orig_h))
        if x2 <= x1 or y2 <= y1:
            continue

        class_id = int(d.get("class_id", 0))
        confidence = float(d.get("confidence", 0))
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)

        if roi_offset is not None and roi_scale is not None:
            x1 = roi_offset[0] + x1 * roi_scale[0]
            y1 = roi_offset[1] + y1 * roi_scale[1]
            w = w * roi_scale[0]
            h = h * roi_scale[1]

        dets.append({
            "image_id": 0,
            "category_id": class_id,
            "bbox": [x1, y1, w, h],
            "score": confidence,
        })
    return dets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="/app/data/seadronessee/compressed_version/annotations/instances_val2017.json")
    parser.add_argument("--img-root", default="/app/data/seadronessee/compressed_version/val2017")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", default="experiments/context_scaling")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading GT...")
    coco_gt = COCO(args.gt)
    all_anns = coco_gt.loadAnns(coco_gt.getAnnIds())
    valid_anns = [a for a in all_anns if a["area"] > 100]
    sampled = random.sample(valid_anns, min(args.num_samples, len(valid_anns)))

    by_image = {}
    for ann in sampled:
        img_id = ann["image_id"]
        if img_id not in by_image:
            by_image[img_id] = []
        by_image[img_id].append(ann)

    image_ids = list(by_image.keys())
    print(f"Sampled {len(sampled)} annotations from {len(image_ids)} images")

    context_ratios = [1.0, 1.5, 3.0, 5.0]
    group_names = ["A_roi_only", "B_ctx_1.5x", "C_ctx_3x", "D_ctx_5x"]

    all_results = {name: [] for name in group_names}
    all_results["E_full"] = []

    stats = {name: {"samples": 0, "dets": 0, "errors": 0} for name in group_names}
    stats["E_full"] = {"samples": 0, "dets": 0, "errors": 0}

    for idx, img_id in enumerate(image_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        img_path = Path(args.img_root) / file_name
        if not img_path.exists():
            img_path = Path(args.img_root) / Path(file_name).name
        if not img_path.exists():
            matches = list(Path(args.img_root).rglob(Path(file_name).name))
            if matches:
                img_path = matches[0]
            else:
                continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        anns_for_img = by_image[img_id]

        # === E: 全图 ===
        full_bytes = encode_image(img)
        full_dets = detect_cloud(full_bytes, img_w, img_h)
        for d in full_dets:
            d["image_id"] = img_id
        all_results["E_full"].extend(full_dets)
        stats["E_full"]["samples"] += 1
        stats["E_full"]["dets"] += len(full_dets)

        # === A/B/C/D: 不同 context ratio ===
        for ann in anns_for_img:
            bbox_xywh = ann["bbox"]

            for ratio, name in zip(context_ratios, group_names):
                roi_xyxy = expand_bbox_xywh(bbox_xywh, ratio, img_w, img_h)
                x1, y1, x2, y2 = roi_xyxy
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    stats[name]["errors"] += 1
                    continue

                crop_bytes = encode_image(crop)
                roi_w_orig = x2 - x1
                roi_h_orig = y2 - y1
                dets = detect_cloud(
                    crop_bytes, img_w, img_h,
                    roi_offset=(x1, y1),
                    roi_scale=(roi_w_orig / max(1, crop.shape[1]), roi_h_orig / max(1, crop.shape[0]))
                )
                for d in dets:
                    d["image_id"] = img_id
                all_results[name].extend(dets)
                stats[name]["samples"] += 1
                stats[name]["dets"] += len(dets)

        if (idx + 1) % 10 == 0:
            elapsed = idx + 1
            print(f"[{elapsed}/{len(image_ids)}] done")

    # 保存结果
    for name in group_names + ["E_full"]:
        out_path = f"{args.out_prefix}_{name}_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results[name], f)
        print(f"Saved: {out_path} ({len(all_results[name])} dets, samples={stats[name]['samples']})")

    with open(f"{args.out_prefix}_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # 评估每组
    print("\n" + "=" * 80)
    print("Context Scaling Experiment Results")
    print("=" * 80)

    for name in ["E_full"] + group_names:
        dets = all_results[name]
        print(f"\n--- {name} ---")
        print(f"  Samples: {stats[name]['samples']}, Total dets: {stats[name]['dets']}")

        if not dets:
            print("  No detections, skipping eval")
            continue

        pred_ids = {d["image_id"] for d in dets}
        gt_imgs = coco_gt.loadImgs(list(pred_ids))
        gt_image_ids = {img["id"] for img in gt_imgs}
        gt_anns_sub = [a for a in coco_gt.loadAnns(coco_gt.getAnnIds()) if a["image_id"] in gt_image_ids]

        subset_gt = {
            "images": gt_imgs,
            "annotations": gt_anns_sub,
            "categories": coco_gt.dataset["categories"],
        }

        import tempfile, os
        gt_f = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        json.dump(subset_gt, gt_f)
        gt_f.close()

        dt_f = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        json.dump(dets, dt_f)
        dt_f.close()

        try:
            coco_eval_gt = COCO(gt_f.name)
            coco_dt = coco_eval_gt.loadRes(dt_f.name)
            coco_eval = COCOeval(coco_eval_gt, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        finally:
            os.unlink(gt_f.name)
            os.unlink(dt_f.name)

    print("\n" + "=" * 80)
    print("Summary table:")
    print("=" * 80)
    print(f"{'Group':<15} {'Samples':>8} {'Dets':>8}")
    for name in ["E_full"] + group_names:
        print(f"{name:<15} {stats[name]['samples']:>8} {stats[name]['dets']:>8}")

if __name__ == "__main__":
    main()
