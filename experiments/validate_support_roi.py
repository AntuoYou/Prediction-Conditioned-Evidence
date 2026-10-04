#!/usr/bin/env python3
"""验证Support ROI在云端DETR的检测能力
对比：50% ROI整块裁剪 vs Support ROI拼贴
"""
import json, os, io, argparse
import numpy as np
import requests
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

YOLO_MODEL_PATH = "/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt"
CLOUD_URL = "http://localhost:8000/detect"
GT_PATH = "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"
IMG_DIR = "/app/data/seadronessee/compressed_version/val2017/"
EXPAND_RATIO = 0.2  # Support ROI扩展比例

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

def compute_roi_50(dets, img_w, img_h):
    if not dets:
        return [0, 0, min(img_w, 800), min(img_h, 800)]
    x1s = [d[0] for d in dets]; y1s = [d[1] for d in dets]
    x2s = [d[2] for d in dets]; y2s = [d[3] for d in dets]
    cx = (min(x1s) + max(x2s)) / 2
    cy = (min(y1s) + max(y2s)) / 2
    ss = np.sqrt(0.5)
    cw, ch = int(img_w * ss), int(img_h * ss)
    x1 = int(max(0, cx - cw/2)); y1 = int(max(0, cy - ch/2))
    x2, y2 = x1 + cw, y1 + ch
    if x2 > img_w: x2 = img_w; x1 = max(0, x2 - cw)
    if y2 > img_h: y2 = img_h; y1 = max(0, y2 - ch)
    return [int(x1), int(y1), int(x2), int(y2)]

def make_support_canvas(img, dets, expand=EXPAND_RATIO):
    """把每个检测框扩展成Support ROI，贴到灰色画布上
    返回: 画布图, 偏移映射(画布坐标 -> 原图坐标的偏移量)
    """
    h, w = img.shape[:2]
    
    # 灰色背景
    canvas = np.full_like(img, 114)  # YOLO默认灰
    
    # 计算每个Support ROI
    supports = []
    for det in dets:
        x1, y1, x2, y2 = det
        bw, bh = x2 - x1, y2 - y1
        sx1 = int(max(0, x1 - bw * expand))
        sy1 = int(max(0, y1 - bh * expand))
        sx2 = int(min(w, x2 + bw * expand))
        sy2 = int(min(h, y2 + bh * expand))
        supports.append((sx1, sy1, sx2, sy2))
    
    # 把Support ROI贴到画布上（位置不变，背景是灰色）
    for sx1, sy1, sx2, sy2 in supports:
        canvas[sy1:sy2, sx1:sx2] = img[sy1:sy2, sx1:sx2]
    
    return canvas, supports

def evaluate(results_list, name, img_ids_used):
    """COCO评估"""
    if not results_list:
        print(f"  {name}: 无结果")
        return {"AP50": 0, "AP": 0, "AR": 0, "num_dets": 0}
    
    tmp = f"/tmp/val_support_{name}.json"
    with open(tmp, "w") as f:
        json.dump(results_list, f)
    
    try:
        coco_gt = COCO(GT_PATH)
        # 只评估用到的图片
        coco_gt.imgs = {k: v for k, v in coco_gt.imgs.items() if k in img_ids_used}
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
            "num_dets": len(results_list),
        }
    except Exception as ex:
        print(f"  {name} 评估失败: {ex}")
        import traceback; traceback.print_exc()
        return {"AP50": 0, "AP": 0, "AR": 0, "num_dets": len(results_list)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=50)
    args = parser.parse_args()
    
    print("=" * 60)
    print("验证: Support ROI 云端检测能力")
    print("=" * 60)
    
    with open(GT_PATH) as f:
        gt_coco = json.load(f)
    images = gt_coco["images"][:args.max_images]
    print(f"图片数: {len(images)}")
    
    model = YOLO(YOLO_MODEL_PATH)
    
    results_50 = []     # 50% ROI
    results_sup = []    # Support ROI拼贴
    img_ids_used = set()
    
    total_50_pixels = 0
    total_sup_pixels = 0
    total_img_pixels = 0
    
    for img_info in tqdm(images, desc="处理中"):
        img_id = img_info["id"]
        fname = img_info["file_name"]
        img_path = os.path.join(IMG_DIR, fname)
        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]
        total_img_pixels += h * w
        
        # YOLO检测（低置信）
        results = model(img_path, conf=0.01, verbose=False)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        
        xyxy = r.boxes.xyxy.cpu().numpy().tolist()
        dets = [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in xyxy]
        
        # 方案1: 50% ROI整块裁剪
        roi = compute_roi_50(dets, w, h)
        rx1, ry1, rx2, ry2 = roi
        crop_50 = img[ry1:ry2, rx1:rx2]
        total_50_pixels += crop_50.shape[0] * crop_50.shape[1]
        
        cloud_50 = send_cloud(crop_50, 0.05)
        for d in cloud_50:
            b = d["bbox"]
            results_50.append({
                "image_id": int(img_id),
                "category_id": int(d.get("class_id", 1)),
                "bbox": [float(b[0]+rx1), float(b[1]+ry1), float(b[2]-b[0]), float(b[3]-b[1])],
                "score": float(d.get("confidence", 0)),
            })
        
        # 方案2: Support ROI拼贴
        canvas_sup, supports = make_support_canvas(img, dets)
        # 计算Support ROI的像素总量
        sup_pix = sum((sx2-sx1)*(sy2-sy1) for sx1,sy1,sx2,sy2 in supports)
        total_sup_pixels += sup_pix
        
        # 整张画布发云端（坐标和原图一致，不需要偏移）
        cloud_sup = send_cloud(canvas_sup, 0.05)
        for d in cloud_sup:
            b = d["bbox"]
            results_sup.append({
                "image_id": int(img_id),
                "category_id": int(d.get("class_id", 1)),
                "bbox": [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])],
                "score": float(d.get("confidence", 0)),
            })
        
        img_ids_used.add(int(img_id))
    
    # 评估
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    
    print(f"\n传输量统计（相对于原图）:")
    print(f"  50% ROI:     {total_50_pixels/total_img_pixels*100:.1f}%")
    print(f"  Support ROI: {total_sup_pixels/total_img_pixels*100:.1f}%")
    
    print(f"\n检测数量:")
    print(f"  50% ROI:     {len(results_50)} 个")
    print(f"  Support ROI: {len(results_sup)} 个")
    
    print("\n--- 50% ROI ---")
    m_50 = evaluate(results_50, "roi50", img_ids_used)
    
    print("\n--- Support ROI ---")
    m_sup = evaluate(results_sup, "support", img_ids_used)
    
    print(f"\n{'方案':<20} {'AP50':<10} {'AP':<10} {'AR':<10} {'传输量':<10}")
    print("-" * 60)
    print(f"{'50% ROI':<20} {m_50['AP50']:<10.4f} {m_50['AP']:<10.4f} {m_50['AR']:<10.4f} {total_50_pixels/total_img_pixels*100:.1f}%")
    print(f"{'Support ROI':<20} {m_sup['AP50']:<10.4f} {m_sup['AP']:<10.4f} {m_sup['AR']:<10.4f} {total_sup_pixels/total_img_pixels*100:.1f}%")
    
    if m_50['AP50'] > 0:
        print(f"\nSupport ROI 相对 50% ROI:")
        print(f"  AP50保持率: {m_sup['AP50']/m_50['AP50']*100:.1f}%")
        print(f"  传输量比例: {total_sup_pixels/total_50_pixels*100:.1f}%")
    
    with open("/app/Salience-DETR/experiments/support_roi_result.json", "w") as f:
        json.dump({
            "roi50": m_50,
            "support": m_sup,
            "transfer_50_pct": total_50_pixels/total_img_pixels*100,
            "transfer_sup_pct": total_sup_pixels/total_img_pixels*100,
        }, f, indent=2)

if __name__ == "__main__":
    main()
