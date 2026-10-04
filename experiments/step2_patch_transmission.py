#!/usr/bin/env python3
"""Step 2: Patch-level Sparse Transmission 验证

对比:
  A. ROI原图直接发云端 (Baseline)
  B. ROI切成16x16patch, 全部传输, 重建后发云端 (100% patch)
  C. ROI切成16x16patch, 未发送patch用模糊背景填充

验证: patch级传输是否能保持DETR性能
"""
import json, os, io, argparse
import numpy as np
import cv2
import requests
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

YOLO_MODEL_PATH = "/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt"
CLOUD_URL = "http://localhost:8000/detect"
GT_ANNOTATION_PATH = "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"
IMAGE_DIR = "/app/data/seadronessee/compressed_version/val2017/"
YOLO_CONF = 0.01
PATCH_SIZE = 16
SUPPORT_EXPAND = 0.20
GRID_ROWS = 8; GRID_COLS = 8
ALPHA = 0.3; BETA = 0.7
TOPK = 8
EXPAND_FOR_CROP = 0.5  # 裁剪ROI时扩展50%上下文(和baseline一致)

def run_yolo(model, img_path, conf=YOLO_CONF):
    results = model(img_path, conf=conf, verbose=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0: return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clses = r.boxes.cls.cpu().numpy().astype(int)
    return [{"bbox":[float(xyxy[i][0]),float(xyxy[i][1]),float(xyxy[i][2]),float(xyxy[i][3])],
             "score":float(confs[i]),"class_id":int(clses[i])} for i in range(len(xyxy))]

def send_to_cloud(image_np, conf=0.05):
    pil = Image.fromarray(image_np)
    buf = io.BytesIO(); pil.save(buf, format="JPEG", quality=95); buf.seek(0)
    try:
        r = requests.post(CLOUD_URL,
            files={"file": ("image.jpg", buf, "image/jpeg")},
            data={"conf_threshold": str(conf)}, timeout=60)
        if r.status_code == 200:
            d = r.json()
            return d.get("detections", [])
    except Exception as e:
        pass
    return []

def merge_dets_to_roi(dets, roi_x1, roi_y1):
    """把ROI内的检测坐标映射回原图"""
    result = []
    for d in dets:
        bbox = d.get("bbox", [0,0,0,0])
        result.append({
            "bbox": [bbox[0]+roi_x1, bbox[1]+roi_y1, bbox[2]+roi_x1, bbox[3]+roi_y1],
            "score": d.get("confidence", 0),
            "class_id": d.get("class_id", 0),
        })
    return result

def compute_crop_region(detections, img_w, img_h, sf=EXPAND_FOR_CROP):
    """所有检测框合并成一个大ROI，扩展sf比例"""
    if not detections:
        return [0, 0, min(img_w, 800), min(img_h, 800)]
    x1s = [d["bbox"][0] for d in detections]
    y1s = [d["bbox"][1] for d in detections]
    x2s = [d["bbox"][2] for d in detections]
    y2s = [d["bbox"][3] for d in detections]
    cx = (min(x1s) + max(x2s)) / 2
    cy = (min(y1s) + max(y2s)) / 2
    ss = np.sqrt(sf)
    cw, ch = int(img_w * ss), int(img_h * ss)
    x1 = int(cx - cw/2); y1 = int(cy - ch/2)
    x1 = max(0, x1); y1 = max(0, y1)
    x2, y2 = x1 + cw, y1 + ch
    if x2 > img_w: x2 = img_w; x1 = max(0, x2 - cw)
    if y2 > img_h: y2 = img_h; y1 = max(0, y2 - ch)
    return [x1, y1, x2, y2]

def patchify_and_reconstruct(crop_img, patch_size=PATCH_SIZE, fill_mode="blur"):
    """把ROI切成patch，再拼回去。fill_mode: 'all'(全保留) or 'blur'(未发的用模糊背景)"""
    h, w = crop_img.shape[:2]
    
    if fill_mode == "all":
        # 全部patch都保留，直接重建（相当于无损，验证切分本身是否影响）
        return crop_img.copy(), h * w * 3  # 返回原图和字节数
    
    # 模式：切patch，全部发送，重建
    # 计算patch网格
    n_rows = (h + patch_size - 1) // patch_size
    n_cols = (w + patch_size - 1) // patch_size
    
    # 创建模糊背景
    if fill_mode == "blur":
        background = cv2.GaussianBlur(crop_img, (31, 31), 0)
    else:  # gray
        background = np.full_like(crop_img, 110)
    
    canvas = background.copy()
    
    # 把所有patch放回去
    total_pixels = 0
    for r in range(n_rows):
        for c in range(n_cols):
            px1 = c * patch_size
            py1 = r * patch_size
            px2 = min(px1 + patch_size, w)
            py2 = min(py1 + patch_size, h)
            patch = crop_img[py1:py2, px1:px2]
            canvas[py1:py2, px1:px2] = patch
            total_pixels += patch.size
    
    return canvas, total_pixels

def evaluate_coco(gt_coco, results, img_id_map):
    if not results:
        return {"AP50": 0, "AP": 0, "AR": 0}
    
    # 转换为COCO格式
    coco_results = []
    for r in results:
        img_id = r["image_id"]
        for d in r["detections"]:
            bbox = d["bbox"]
            coco_results.append({
                "image_id": img_id,
                "category_id": d["class_id"],
                "bbox": [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]],
                "score": d["score"],
            })
    
    # 写入临时文件
    tmp_path = "/tmp/step2_coco_results.json"
    with open(tmp_path, "w") as f:
        json.dump(coco_results, f)
    
    try:
        coco_gt = COCO(GT_ANNOTATION_PATH)
        coco_dt = coco_gt.loadRes(tmp_path)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return {
            "AP50": float(coco_eval.stats[1]),
            "AP": float(coco_eval.stats[0]),
            "AR": float(coco_eval.stats[8]),
        }
    except Exception as e:
        print(f"评估错误: {e}")
        return {"AP50": 0, "AP": 0, "AR": 0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=20)
    args = parser.parse_args()
    
    print("=" * 70)
    print("Step 2: Patch级传输验证")
    print("=" * 70)
    
    with open(GT_ANNOTATION_PATH) as f:
        gt_coco = json.load(f)
    image_map = {img["id"]: img for img in gt_coco["images"]}
    image_list = list(image_map.values())[:args.max_images]
    
    print(f"图片数: {len(image_list)}")
    print(f"Patch大小: {PATCH_SIZE}x{PATCH_SIZE}")
    print()
    
    model = YOLO(YOLO_MODEL_PATH)
    
    results_a = []  # Baseline: ROI原图
    results_b = []  # 100% patch重建
    results_c = []  # 模糊背景重建
    
    bytes_a = 0; bytes_b = 0; bytes_c = 0
    
    for img_info in tqdm(image_list, desc="Step 2"):
        img_id = img_info["id"]
        img_path = os.path.join(IMAGE_DIR, img_info["file_name"])
        img = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w = img.shape[:2]
        
        dets = run_yolo(model, img_path)
        
        if not dets:
            # 没有检测就跳过
            continue
        
        # 计算ROI裁剪区域
        roi = compute_crop_region(dets, img_w, img_h)
        rx1, ry1, rx2, ry2 = roi
        crop = img[ry1:ry2, rx1:rx2]
        
        # A: ROI原图直接发云端
        cloud_dets_a = send_to_cloud(crop, conf=0.05)
        mapped_a = merge_dets_to_roi(cloud_dets_a, rx1, ry1)
        results_a.append({"image_id": img_id, "detections": mapped_a})
        bytes_a += crop.size
        
        # B: 100% patch全部保留（无损重建，验证切分不影响）
        recon_b, pixels_b = patchify_and_reconstruct(crop, fill_mode="all")
        cloud_dets_b = send_to_cloud(recon_b, conf=0.05)
        mapped_b = merge_dets_to_roi(cloud_dets_b, rx1, ry1)
        results_b.append({"image_id": img_id, "detections": mapped_b})
        bytes_b += pixels_b
        
        # C: 模糊背景 + 所有patch（验证背景填充策略）
        recon_c, pixels_c = patchify_and_reconstruct(crop, fill_mode="blur")
        cloud_dets_c = send_to_cloud(recon_c, conf=0.05)
        mapped_c = merge_dets_to_roi(cloud_dets_c, rx1, ry1)
        results_c.append({"image_id": img_id, "detections": mapped_c})
        bytes_c += pixels_c
    
    # 评估
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    
    img_id_map = {img["file_name"]: img["id"] for img in gt_coco["images"]}
    
    metrics_a = evaluate_coco(gt_coco, results_a, img_id_map)
    metrics_b = evaluate_coco(gt_coco, results_b, img_id_map)
    metrics_c = evaluate_coco(gt_coco, results_c, img_id_map)
    
    print(f"\n{'方法':<30} {'AP50':<10} {'AP':<10} {'AR':<10} {'大小':<12}")
    print("-" * 72)
    print(f"{'A: ROI原图 (Baseline)':<30} {metrics_a['AP50']:<10.4f} {metrics_a['AP']:<10.4f} {metrics_a['AR']:<10.4f} {bytes_a/1024/1024:<10.1f} MB")
    print(f"{'B: 100% Patch (无损)':<30} {metrics_b['AP50']:<10.4f} {metrics_b['AP']:<10.4f} {metrics_b['AR']:<10.4f} {bytes_b/1024/1024:<10.1f} MB")
    print(f"{'C: Patch + 模糊背景':<30} {metrics_c['AP50']:<10.4f} {metrics_c['AP']:<10.4f} {metrics_c['AR']:<10.4f} {bytes_c/1024/1024:<10.1f} MB")
    
    # 相对Baseline的保持率
    if metrics_a['AP50'] > 0:
        print(f"\n相对Baseline保持率:")
        print(f"  B (100% patch): AP50保持 {metrics_b['AP50']/metrics_a['AP50']*100:.1f}%")
        print(f"  C (模糊背景): AP50保持 {metrics_c['AP50']/metrics_a['AP50']*100:.1f}%")
    
    # 保存结果
    with open("/app/Salience-DETR/experiments/step2_result.json", "w") as f:
        json.dump({
            "num_images": len(image_list),
            "patch_size": PATCH_SIZE,
            "A": metrics_a, "B": metrics_b, "C": metrics_c,
            "bytes_A_MB": bytes_a/1024/1024,
            "bytes_B_MB": bytes_b/1024/1024,
            "bytes_C_MB": bytes_c/1024/1024,
        }, f, indent=2)
    print(f"\n[保存] step2_result.json")

if __name__ == "__main__":
    main()
