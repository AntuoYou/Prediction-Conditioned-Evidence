import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import requests
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLOUD_URL = "http://localhost:8000/detect"

def encode_image(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()

def detect_cloud(img_bytes, orig_w, orig_h, timeout=30):
    files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
    data = {"conf_threshold": "0.05"}
    try:
        resp = requests.post(CLOUD_URL, files=files, data=data, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        print(f"  Cloud error: {e}")
        return []

    dets = []
    for d in result.get("detections", []):
        bbox = d.get("bbox", {})
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        elif isinstance(bbox, dict):
            x1, y1, x2, y2 = float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"])
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

        dets.append({
            "image_id": 0,
            "category_id": class_id,
            "bbox": [x1, y1, w, h],
            "score": confidence,
        })
    return dets

def create_fixed_scale_roi(img, bbox_xywh, target_ratio, canvas_size, img_w, img_h):
    """
    构造一个固定目标尺度的 ROI 输入：
    - 把目标 bbox resize 到占 canvas 的 target_ratio
    - 周围用原图真实背景填充
    """
    x, y, bw, bh = bbox_xywh
    cx, cy = x + bw / 2, y + bh / 2

    # 计算 canvas 上目标应该有多大
    canvas_area = canvas_size * canvas_size
    target_area = canvas_area * target_ratio
    scale = np.sqrt(target_area / max(1, bw * bh))
    new_w = int(bw * scale)
    new_h = int(bh * scale)

    # 从原图 crop 出足够大的区域来做背景
    ctx_ratio = 3.0  # 背景区域取 3x context
    crop_w = int(bw * ctx_ratio)
    crop_h = int(bh * ctx_ratio)
    crop_x1 = max(0, int(cx - crop_w / 2))
    crop_y1 = max(0, int(cy - crop_h / 2))
    crop_x2 = min(img_w, crop_x1 + crop_w)
    crop_y2 = min(img_h, crop_y1 + crop_h)
    crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # Resize crop 到 canvas_size
    if crop.size == 0:
        return None, None
    bg = cv2.resize(crop, (canvas_size, canvas_size))

    # 在 bg 中央贴上 resize 后的目标区域
    paste_x1 = (canvas_size - new_w) // 2
    paste_y1 = (canvas_size - new_h) // 2
    paste_x2 = paste_x1 + new_w
    paste_y2 = paste_y1 + new_h

    # 从原图截取目标区域并 resize
    obj_crop = img[int(y):int(y+bh), int(x):int(x+bw)]
    if obj_crop.size == 0:
        return None, None
    obj_resized = cv2.resize(obj_crop, (new_w, new_h))

    # 贴到背景上
    bg[paste_y1:paste_y2, paste_x1:paste_x2] = obj_resized

    # 计算坐标映射：canvas 上的目标坐标 → 原图坐标
    # 这里只是近似，因为我们是从原图截取的
    offset_x = crop_x1 + (crop_x2 - crop_x1) * (paste_x1 / canvas_size)
    offset_y = crop_y1 + (crop_y2 - crop_y1) * (paste_y1 / canvas_size)
    scale_x = (crop_x2 - crop_x1) / canvas_size * (new_w / max(1, obj_resized.shape[1]))
    scale_y = (crop_y2 - crop_y1) / canvas_size * (new_h / max(1, obj_resized.shape[0]))

    return bg, (offset_x, offset_y, scale_x, scale_y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="/app/data/seadronessee/compressed_version/annotations/instances_val2017.json")
    parser.add_argument("--img-root", default="/app/data/seadronessee/compressed_version/val2017")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", default="experiments/scale_vs_context")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading GT...")
    coco_gt = COCO(args.gt)
    all_anns = coco_gt.loadAnns(coco_gt.getAnnIds())
    valid_anns = [a for a in all_anns if 500 < a["area"] < 10000]  # 中等大小目标
    sampled = random.sample(valid_anns, min(args.num_samples, len(valid_anns)))

    by_image = {}
    for ann in sampled:
        img_id = ann["image_id"]
        if img_id not in by_image:
            by_image[img_id] = []
        by_image[img_id].append(ann)

    image_ids = list(by_image.keys())
    print(f"Sampled {len(sampled)} annotations from {len(image_ids)} images")

    # 实验组
    configs = [
        ("A_1x_roi", 1.0, None),      # 原始 1x ROI
        ("B_1x_fixed_scale", 1.0, 0.05),  # 1x ROI 但固定目标占 5%
        ("C_3x_fixed_scale", 3.0, 0.05),  # 3x ROI 但固定目标占 5%
        ("D_5x_roi", 5.0, None),      # 原始 5x ROI
        ("E_full", None, None),       # 全图
    ]

    all_results = {name: [] for name, _, _ in configs}
    stats = {name: {"samples": 0, "dets": 0} for name, _, _ in configs}

    for idx, img_id in enumerate(image_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        img_w, img_h = img_info["width"], img_info["height"]

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

        # E: 全图
        full_bytes = encode_image(img)
        full_dets = detect_cloud(full_bytes, img_w, img_h)
        for d in full_dets:
            d["image_id"] = img_id
        all_results["E_full"].extend(full_dets)
        stats["E_full"]["samples"] += 1
        stats["E_full"]["dets"] += len(full_dets)

        for ann in anns_for_img:
            bbox_xywh = ann["bbox"]
            x, y, bw, bh = bbox_xywh
            cx, cy = x + bw / 2, y + bh / 2

            for name, ctx_ratio, target_ratio in configs:
                if name == "E_full":
                    continue

                if target_ratio is None:
                    # 原始 ROI：按 context ratio 扩展
                    crop_w = int(bw * ctx_ratio)
                    crop_h = int(bh * ctx_ratio)
                    crop_x1 = max(0, int(cx - crop_w / 2))
                    crop_y1 = max(0, int(cy - crop_h / 2))
                    crop_x2 = min(img_w, crop_x1 + crop_w)
                    crop_y2 = min(img_h, crop_y1 + crop_h)
                    crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

                    if crop.size == 0:
                        continue

                    crop_bytes = encode_image(crop)
                    dets = detect_cloud(crop_bytes, img_w, img_h)

                    # 坐标映射
                    scale_x = (crop_x2 - crop_x1) / max(1, crop.shape[1])
                    scale_y = (crop_y2 - crop_y1) / max(1, crop.shape[0])
                    for d in dets:
                        d["bbox"][0] = crop_x1 + d["bbox"][0] * scale_x
                        d["bbox"][1] = crop_y1 + d["bbox"][1] * scale_y
                        d["bbox"][2] = d["bbox"][2] * scale_x
                        d["bbox"][3] = d["bbox"][3] * scale_y

                else:
                    # 固定目标尺度版本
                    canvas, mapping = create_fixed_scale_roi(
                        img, bbox_xywh, target_ratio, 800, img_w, img_h
                    )
                    if canvas is None or mapping is None:
                        continue

                    offset_x, offset_y, scale_x, scale_y = mapping
                    crop_bytes = encode_image(canvas)
                    dets = detect_cloud(crop_bytes, img_w, img_h)

                    for d in dets:
                        d["bbox"][0] = offset_x + d["bbox"][0] * scale_x
                        d["bbox"][1] = offset_y + d["bbox"][1] * scale_y
                        d["bbox"][2] = d["bbox"][2] * scale_x
                        d["bbox"][3] = d["bbox"][3] * scale_y

                for d in dets:
                    d["image_id"] = img_id
                all_results[name].extend(dets)
                stats[name]["samples"] += 1
                stats[name]["dets"] += len(dets)

        if (idx + 1) % 10 == 0:
            print(f"[{idx+1}/{len(image_ids)}] done")

    # 保存和评估
    for name, _, _ in configs:
        out_path = f"{args.out_prefix}_{name}_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results[name], f)

    with open(f"{args.out_prefix}_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 80)
    for name, _, _ in configs:
        dets = all_results[name]
        print(f"\n--- {name} ---")
        print(f"  Samples: {stats[name]['samples']}, Dets: {stats[name]['dets']}")

        if not dets:
            print("  No detections")
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

if __name__ == "__main__":
    main()
