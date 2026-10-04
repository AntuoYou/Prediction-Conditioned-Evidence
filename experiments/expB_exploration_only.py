#!/usr/bin/env python3
"""E2 (corrected): Exploration-only —— 论文口径（不确定性热力图 + 邻近度/纹理/低置信度密度）

底本: expB 低分辨率背景 + JPEG 编码 + 云端细化 API（传输量口径与 expB 完全一致）
探索补丁: pce_experiments/exploration.py:compute_exploration_patches
          （论文 Section 4.4, Eq.12；需 edge_output / support_regions / support_mask）

与 run_supplementary.py (S4) 的真实调用链对齐:
  edge_out        = YOLO 检测 -> EdgeOutput
  support_regions = generate_support_regions(edge_out, 0.5, 16)
  support_mask    = get_support_mask(support_regions, image_shape, 16)
  exploration_patches = compute_exploration_patches(img, edge_out, support_regions,
                                                    support_mask, budget_patches, ...)
canvas = 低分辨率背景 + 只贴 exploration 补丁（不贴 support ROI）= 纯 exploration-only
"""
import sys, os, io, json, argparse
import numpy as np
import cv2
import requests
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 项目模块（论文真实实现）
sys.path.insert(0, "/app/Salience-DETR")
sys.path.insert(0, "/app/Salience-DETR/pce_experiments")
from edge_detector import EdgeOutput, Detection
from support_region import generate_support_regions, get_support_mask
from exploration import compute_exploration_patches

YOLO_MODEL_PATH = "/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt"
CLOUD_URL = "http://localhost:8000/detect"
GT_PATH = "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"
IMG_DIR = "/app/data/seadronessee/compressed_version/val2017/"
LOW_RES_SCALE = 8
DEFAULT_BUDGET = 115  # 论文 S4 总 patch 预算: budget_kb=87 -> 89088//772 = 115


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
        print(f"  云端错误: {e}")
    return []


def apply_lowres_bg(img):
    """纯低分辨率背景（8x 下采样再上采样），与 expB 背景口径一致"""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // LOW_RES_SCALE), max(1, h // LOW_RES_SCALE)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def build_edge_output(img, img_id, model):
    """用与 expB 同款 YOLO（conf=0.01）跑检测，封装成论文的 EdgeOutput。"""
    H, W = img.shape[:2]
    r = model(img, conf=0.01, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return EdgeOutput(image_id=str(img_id), image_shape=(H, W), detections=[])
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    labels = r.boxes.cls.cpu().numpy().astype(int)
    dets = [Detection(bbox=tuple(b.tolist()), label=int(l), score=float(s))
            for b, s, l in zip(boxes, scores, labels)]
    return EdgeOutput(image_id=str(img_id), image_shape=(H, W), detections=dets)


def make_exploration_canvas(img, edge_out, budget):
    """低分辨率背景 + 论文 exploration 补丁（纯 exploration-only，不送 support ROI）"""
    canvas = apply_lowres_bg(img)
    support_regions = generate_support_regions(edge_out, 0.5, 16)
    support_mask = get_support_mask(support_regions, edge_out.image_shape, 16)
    patches = compute_exploration_patches(
        img, edge_out, support_regions, support_mask,
        budget_patches=budget, patch_size=16,
        w_distance=0.4, w_texture=0.3, w_lowconf=0.3)
    for p in patches:
        x, y, s = int(p.x), int(p.y), int(p.size)
        x2, y2 = min(img.shape[1], x + s), min(img.shape[0], y + s)
        if x2 > x and y2 > y:
            canvas[y:y2, x:x2] = img[y:y2, x:x2]
    return canvas, patches


def calc_payload(img, patches):
    """传输量口径与 expB 完全一致: 低分辨率全局 JPEG + 各 exploration 补丁 JPEG"""
    h, w = img.shape[:2]
    total = 0
    small = cv2.resize(img, (max(1, w // LOW_RES_SCALE), max(1, h // LOW_RES_SCALE)),
                       cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(small).save(buf, format='JPEG', quality=95)
    total += len(buf.getvalue())
    for p in patches:
        x, y, s = int(p.x), int(p.y), int(p.size)
        crop = img[max(0, y):min(h, y + s), max(0, x):min(w, x + s)]
        if crop.size > 0:
            b = io.BytesIO()
            Image.fromarray(crop).save(b, format='JPEG', quality=95)
            total += len(b.getvalue())
    return total


def evaluate(results_list, name, img_ids_used):
    if not results_list:
        return {"AP50": 0, "AP": 0, "AP75": 0, "AR": 0, "APsmall": 0,
                "APmedium": 0, "APlarge": 0, "num_dets": 0}
    tmp = f"/tmp/expB_exploration_only_{name}.json"
    with open(tmp, "w") as f:
        json.dump(results_list, f)
    try:
        coco_gt = COCO(GT_PATH)
        coco_dt = coco_gt.loadRes(tmp)
        e = COCOeval(coco_gt, coco_dt, 'bbox')
        e.params.imgIds = list(img_ids_used)
        e.evaluate(); e.accumulate(); e.summarize()
        return {
            "AP50": float(e.stats[1]), "AP": float(e.stats[0]), "AP75": float(e.stats[2]),
            "AR": float(e.stats[8]),
            "APsmall": float(e.stats[3]), "APmedium": float(e.stats[4]), "APlarge": float(e.stats[5]),
            "num_dets": len(results_list),
        }
    except Exception as ex:
        print(f"  {name} 评估失败: {ex}")
        return {"AP50": 0, "AP": 0, "AP75": 0, "AR": 0, "APsmall": 0,
                "APmedium": 0, "APlarge": 0, "num_dets": len(results_list)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                        help="exploration 补丁数；论文 S4 总预算=115 (budget_kb=87, lam=0.6)")
    args = parser.parse_args()

    print("=" * 70)
    print("E2 (corrected): Exploration-only —— 论文口径（不确定性热力图）")
    print(f"  exploration budget = {args.budget} patches")
    print("=" * 70)

    with open(GT_PATH) as f:
        gt_coco = json.load(f)
    images = gt_coco["images"][:args.max_images]
    print(f"图片数: {len(images)}")

    model = YOLO(YOLO_MODEL_PATH)
    results = []
    img_ids_used = set()
    total_payload = 0
    total_patches = 0

    for img_info in tqdm(images, desc="处理中"):
        img_id = img_info["id"]
        fname = img_info["file_name"]
        img = np.array(Image.open(os.path.join(IMG_DIR, fname)).convert("RGB"))

        edge_out = build_edge_output(img, img_id, model)
        canvas, patches = make_exploration_canvas(img, edge_out, args.budget)
        total_patches += len(patches)
        cloud_dets = send_cloud(canvas, 0.05)
        for d in cloud_dets:
            b = d["bbox"]
            results.append({
                "image_id": int(img_id),
                "category_id": int(d.get("class_id", 1)),
                "bbox": [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])],
                "score": float(d.get("confidence", 0)),
            })
        total_payload += calc_payload(img, patches)
        img_ids_used.add(int(img_id))

    metrics = evaluate(results, "exploration_only", img_ids_used)
    n = len(img_ids_used)
    avg_bytes = total_payload / n if n else 0

    print("\n" + "=" * 70)
    print("E2 (corrected) 结果")
    print("=" * 70)
    print(f"  AP50    : {metrics['AP50']:.4f}")
    print(f"  mAP     : {metrics['AP']:.4f}")
    print(f"  AP75    : {metrics['AP75']:.4f}")
    print(f"  APsmall : {metrics['APsmall']:.4f}")
    print(f"  APmed   : {metrics['APmedium']:.4f}")
    print(f"  APlarge : {metrics['APlarge']:.4f}")
    print(f"  AR      : {metrics['AR']:.4f}")
    print(f"  检测数   : {metrics['num_dets']}")
    if n:
        print(f"  平均 patch 数/图 : {total_patches / n:.1f}")
    print(f"  传输量  : {avg_bytes/1024:.1f} KB/img")

    with open("/app/Salience-DETR/experiments/expB_exploration_only.json", "w") as f:
        json.dump({
            "mode": "exploration-only (uncertainty heatmap, expB background, paper-equivalent budget)",
            "budget_patches": args.budget,
            "metrics": metrics,
            "payload_bytes_avg": avg_bytes,
            "avg_patches_per_img": (total_patches / n) if n else 0,
            "num_images": n,
            "low_res_scale": LOW_RES_SCALE,
        }, f, indent=2)
    print("\n结果已保存: experiments/expB_exploration_only.json")


if __name__ == "__main__":
    main()
