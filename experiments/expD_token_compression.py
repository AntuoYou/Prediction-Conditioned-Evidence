#!/usr/bin/env python3
"""Experiment D: Token Compression (Phase 1: 200 images trend)

基于 Support + Exploration 双路径，测试不同Token压缩比例。

D0: Full ROI (基线, Support+Exp不压缩)
D1: 100% tokens (patch化基线)
D2: 50% tokens
D3: 25% tokens
D4: 12.5% tokens
D5: 6.25% tokens

Token选择策略: 按patch的L2范数排序（高信息含量优先）
"""
import json, os, io, argparse
import numpy as np
import cv2
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
EXPAND_RATIO = 0.1
LOW_RES_SCALE = 8
GRID_ROWS = 6
GRID_COLS = 10
ALPHA = 0.3
BETA = 0.7
TOPK_EXPLORE = 8
PATCH_SIZE = 16  # 16x16 patch

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
        pass
    return []

def make_support_rois(img, dets, expand=EXPAND_RATIO):
    h, w = img.shape[:2]
    supports = []
    for det in dets:
        x1, y1, x2, y2 = det
        bw, bh = x2 - x1, y2 - y1
        sx1 = int(max(0, x1 - bw * expand))
        sy1 = int(max(0, y1 - bh * expand))
        sx2 = int(min(w, x2 + bw * expand))
        sy2 = int(min(h, y2 + bh * expand))
        if sx2 > sx1 and sy2 > sy1:
            supports.append((sx1, sy1, sx2, sy2))
    return supports

def compute_support_coverage(supports, gx1, gy1, gx2, gy2):
    grid_area = (gx2 - gx1) * (gy2 - gy1)
    if grid_area <= 0:
        return 1.0
    covered = 0
    for sx1, sy1, sx2, sy2 in supports:
        ix1 = max(gx1, sx1); iy1 = max(gy1, sy1)
        ix2 = min(gx2, sx2); iy2 = min(gy2, sy2)
        if ix2 > ix1 and iy2 > iy1:
            covered += (ix2 - ix1) * (iy2 - iy1)
    return min(1.0, covered / grid_area)

def generate_exploration_rois(supports, img_w, img_h, topk=TOPK_EXPLORE):
    grid_w = img_w / GRID_COLS
    grid_h = img_h / GRID_ROWS
    max_dist = (img_w**2 + img_h**2) ** 0.5
    if supports:
        sup_centers = [((s[0]+s[2])/2, (s[1]+s[3])/2) for s in supports]
    else:
        sup_centers = [(img_w/2, img_h/2)]
    
    candidates = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            gx1 = int(c * grid_w)
            gy1 = int(r * grid_h)
            gx2 = int((c + 1) * grid_w)
            gy2 = int((r + 1) * grid_h)
            coverage = compute_support_coverage(supports, gx1, gy1, gx2, gy2)
            gcx = (gx1 + gx2) / 2
            gcy = (gy1 + gy2) / 2
            min_dist = min(((sc[0]-gcx)**2 + (sc[1]-gcy)**2)**0.5 for sc in sup_centers)
            score = ALPHA * (1 - coverage) + BETA * (1 - min_dist / max_dist)
            candidates.append((gx1, gy1, gx2, gy2, score))
    
    candidates.sort(key=lambda x: x[4], reverse=True)
    return [(c[0], c[1], c[2], c[3]) for c in candidates[:topk]]

def make_lowres_canvas(img, rois):
    """低分辨率背景 + ROI叠加（完整ROI，不做patch选择）"""
    h, w = img.shape[:2]
    small_w = max(1, w // LOW_RES_SCALE)
    small_h = max(1, h // LOW_RES_SCALE)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    canvas = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    for rx1, ry1, rx2, ry2 in rois:
        if rx2 > rx1 and ry2 > ry1:
            canvas[ry1:ry2, rx1:rx2] = img[ry1:ry2, rx1:rx2]
    return canvas

def extract_patches_from_rois(img, rois, patch_size=PATCH_SIZE):
    """从所有ROI中提取patch，返回patch列表和对应的位置"""
    h, w = img.shape[:2]
    patches = []  # (x, y, patch_data, score)
    
    for rx1, ry1, rx2, ry2 in rois:
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        
        # 在ROI范围内按网格切patch
        # 对齐到patch_size边界
        start_x = rx1 - (rx1 % patch_size) if rx1 % patch_size != 0 else rx1
        start_y = ry1 - (ry1 % patch_size) if ry1 % patch_size != 0 else ry1
        
        for py in range(start_y, ry2, patch_size):
            for px in range(start_x, rx2, patch_size):
                px2 = min(px + patch_size, w)
                py2 = min(py + patch_size, h)
                
                if px2 <= px or py2 <= py:
                    continue
                
                patch = img[py:py2, px:px2]
                
                # 跳过太小的边缘patch
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    continue
                
                # 计算patch的信息含量（L2范数作为proxy）
                score = float(np.linalg.norm(patch.astype(float)))
                
                patches.append({
                    "x": px, "y": py,
                    "w": px2 - px, "h": py2 - py,
                    "data": patch,
                    "score": score,
                })
    
    return patches

def make_patch_canvas(img, rois, token_ratio, patch_size=PATCH_SIZE):
    """基于patch选择的画布
    
    token_ratio: 1.0 = 100% patch, 0.5 = 50%, etc.
    """
    h, w = img.shape[:2]
    
    # 低分辨率背景
    small_w = max(1, w // LOW_RES_SCALE)
    small_h = max(1, h // LOW_RES_SCALE)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    canvas = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 提取所有ROI内的patch
    patches = extract_patches_from_rois(img, rois, patch_size)
    
    if not patches:
        return canvas, 0
    
    # 按score排序，选top token_ratio
    patches_sorted = sorted(patches, key=lambda p: p["score"], reverse=True)
    n_keep = max(1, int(len(patches_sorted) * token_ratio))
    selected = patches_sorted[:n_keep]
    
    # 把选中的patch贴到画布上
    for p in selected:
        canvas[p["y"]:p["y"]+p["h"], p["x"]:p["x"]+p["w"]] = p["data"]
    
    return canvas, len(selected)

def calc_payload_roi(img, rois):
    """完整ROI传输量（低分辨率背景 + ROI JPEG）"""
    h, w = img.shape[:2]
    total = 0
    # 低分辨率全局
    small_w = max(1, w // LOW_RES_SCALE)
    small_h = max(1, h // LOW_RES_SCALE)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(small).save(buf, format='JPEG', quality=95)
    total += len(buf.getvalue())
    # ROI
    for rx1, ry1, rx2, ry2 in rois:
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        crop = img[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            continue
        buf = io.BytesIO()
        Image.fromarray(crop).save(buf, format='JPEG', quality=95)
        total += len(buf.getvalue())
    return total

def calc_payload_token(img, rois, token_ratio, patch_size=PATCH_SIZE):
    """Token传输量 = 整个画布的JPEG大小（和实际传输一致）"""
    canvas, n_patches = make_patch_canvas(img, rois, token_ratio, patch_size)
    buf = io.BytesIO()
    Image.fromarray(canvas).save(buf, format='JPEG', quality=95)
    return len(buf.getvalue())

def evaluate(results_list, name, img_ids_used):
    if not results_list:
        return {"AP50": 0, "AP": 0, "AR": 0, "APsmall": 0, "APmedium": 0, "APlarge": 0, "AP75": 0, "num_dets": 0}
    tmp = f"/tmp/expD_{name}.json"
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
            "AP75": float(e.stats[2]),
            "APsmall": float(e.stats[3]),
            "APmedium": float(e.stats[4]),
            "APlarge": float(e.stats[5]),
            "AR": float(e.stats[8]),
            "ARsmall": float(e.stats[11]),
            "num_dets": len(results_list),
        }
    except Exception as ex:
        print(f"  {name} 评估失败: {ex}")
        return {"AP50": 0, "AP": 0, "AR": 0, "APsmall": 0, "APmedium": 0, "APlarge": 0, "AP75": 0, "num_dets": len(results_list)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=200)
    args = parser.parse_args()
    
    print("=" * 70)
    print("Experiment D: Token Compression (Phase 1: Trend)")
    print("=" * 70)
    
    with open(GT_PATH) as f:
        gt_coco = json.load(f)
    images = gt_coco["images"][:args.max_images]
    print(f"图片数: {len(images)}")
    
    model = YOLO(YOLO_MODEL_PATH)
    
    # 测试的Token比例
    ratios = [
        ("D0_fullroi", None),      # 完整ROI（基线，不做patch）
        ("D1_100pct", 1.0),        # 100% patch
        ("D2_50pct", 0.5),
        ("D3_25pct", 0.25),
        ("D4_12pct", 0.125),
        ("D5_6pct", 0.0625),
    ]
    
    results = {name: [] for name, _ in ratios}
    img_ids_used = set()
    total_payloads = {name: 0 for name, _ in ratios}
    
    for img_info in tqdm(images, desc="处理中"):
        img_id = img_info["id"]
        fname = img_info["file_name"]
        img_path = os.path.join(IMG_DIR, fname)
        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]
        
        results_yolo = model(img_path, conf=0.01, verbose=False)
        r = results_yolo[0]
        
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().tolist()
            dets = [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in xyxy]
        else:
            dets = []
        
        supports = make_support_rois(img, dets)
        explorations = generate_exploration_rois(supports, w, h)
        all_rois = supports + explorations
        
        for name, ratio in ratios:
            if ratio is None:
                # 完整ROI
                canvas = make_lowres_canvas(img, all_rois)
                total_payloads[name] += calc_payload_roi(img, all_rois)
            else:
                # Token压缩
                canvas, n_patches = make_patch_canvas(img, all_rois, ratio)
                total_payloads[name] += calc_payload_token(img, all_rois, ratio)
            
            cloud_dets = send_cloud(canvas, 0.05)
            for d in cloud_dets:
                b = d["bbox"]
                results[name].append({
                    "image_id": int(img_id),
                    "category_id": int(d.get("class_id", 1)),
                    "bbox": [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])],
                    "score": float(d.get("confidence", 0)),
                })
        
        img_ids_used.add(int(img_id))
    
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    
    metrics = {}
    for name, _ in ratios:
        print(f"\n--- {name} ---")
        metrics[name] = evaluate(results[name], name, img_ids_used)
    
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    
    print(f"\n{'方案':<20} {'AP50':<8} {'AP':<8} {'APsmall':<8} {'APmed':<8} {'APlarge':<8} {'AR':<8} {'检测数':<8} {'传输量KB':<10}")
    print("-" * 100)
    n = len(img_ids_used)
    for name, ratio in ratios:
        m = metrics[name]
        avg_kb = total_payloads[name] / n / 1024
        label = name
        print(f"{label:<20} {m['AP50']:<8.4f} {m['AP']:<8.4f} {m['APsmall']:<8.4f} {m['APmedium']:<8.4f} {m['APlarge']:<8.4f} {m['AR']:<8.4f} {m['num_dets']:<8} {avg_kb:<10.1f}")
    
    # 相对D0的保持率
    base_ap50 = metrics["D0_fullroi"]["AP50"]
    base_apsmall = metrics["D0_fullroi"]["APsmall"]
    base_ar = metrics["D0_fullroi"]["AR"]
    
    print(f"\n相对完整ROI (D0) 的保持率:")
    print(f"{'方案':<20} {'AP50':<10} {'APsmall':<10} {'AR':<10} {'Payload':<10}")
    print("-" * 60)
    base_payload = total_payloads["D0_fullroi"]
    for name, ratio in ratios:
        m = metrics[name]
        if base_ap50 > 0:
            p_ap50 = m['AP50']/base_ap50*100
            p_apsmall = m['APsmall']/base_apsmall*100 if base_apsmall > 0 else 0
            p_ar = m['AR']/base_ar*100 if base_ar > 0 else 0
            p_payload = total_payloads[name]/base_payload*100
            print(f"{name:<20} {p_ap50:<9.1f}% {p_apsmall:<9.1f}% {p_ar:<9.1f}% {p_payload:<9.1f}%")
    
    with open("/app/Salience-DETR/experiments/expD_token_compression.json", "w") as f:
        json.dump({
            "ratios": {name: str(ratio) for name, ratio in ratios},
            "metrics": metrics,
            "payloads_bytes": total_payloads,
            "avg_payload_kb": {name: total_payloads[name]/n/1024 for name, _ in ratios},
            "num_images": n,
            "config": {
                "patch_size": PATCH_SIZE,
                "expand_ratio": EXPAND_RATIO,
                "low_res_scale": LOW_RES_SCALE,
                "alpha": ALPHA,
                "beta": BETA,
                "topk_explore": TOPK_EXPLORE,
                "selection": "L2-norm scoring",
            },
        }, f, indent=2)
    
    print(f"\n结果已保存: expD_token_compression.json")

if __name__ == "__main__":
    main()
