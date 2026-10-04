#!/usr/bin/env python3
"""Experiment D v2: Patch-level Evidence Compression

通信模型:
  ROI → 16x16 patchify → 按比例选择 → packed canvas → 一次JPEG
  传输: packed JPEG + binary mask + ROI coordinates
  云端: decode → mask恢复位置 → 叠加低分辨率背景 → DETR

Token选择策略: Uniform grid sampling (第一版, 不用复杂scoring)
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
PATCH_SIZE = 16

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
    except:
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
            gx1 = int(c * grid_w); gy1 = int(r * grid_h)
            gx2 = int((c + 1) * grid_w); gy2 = int((r + 1) * grid_h)
            coverage = compute_support_coverage(supports, gx1, gy1, gx2, gy2)
            gcx = (gx1 + gx2) / 2; gcy = (gy1 + gy2) / 2
            min_dist = min(((sc[0]-gcx)**2 + (sc[1]-gcy)**2)**0.5 for sc in sup_centers)
            score = ALPHA * (1 - coverage) + BETA * (1 - min_dist / max_dist)
            candidates.append((gx1, gy1, gx2, gy2, score))
    candidates.sort(key=lambda x: x[4], reverse=True)
    return [(c[0], c[1], c[2], c[3]) for c in candidates[:topk]]

# ============================================================
# Patch-level Evidence Compression
# ============================================================

def patchify_roi(img, roi, patch_size=PATCH_SIZE):
    """把ROI切成16x16 patch网格
    
    Returns:
        patches: list of (row, col, patch_data)
        n_rows, n_cols: patch网格尺寸
    """
    rx1, ry1, rx2, ry2 = roi
    roi_w = rx2 - rx1
    roi_h = ry2 - ry1
    
    if roi_w <= 0 or roi_h <= 0:
        return [], 0, 0
    
    n_cols = (roi_w + patch_size - 1) // patch_size
    n_rows = (roi_h + patch_size - 1) // patch_size
    
    patches = []
    for r in range(n_rows):
        for c in range(n_cols):
            px = rx1 + c * patch_size
            py = ry1 + r * patch_size
            px2 = min(px + patch_size, rx2)
            py2 = min(py + patch_size, ry2)
            
            # padding到patch_size x patch_size
            patch = img[py:py2, px:px2]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                pad_h = patch_size - patch.shape[0]
                pad_w = patch_size - patch.shape[1]
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
            
            patches.append((r, c, patch))
    
    return patches, n_rows, n_cols

def uniform_select(n_rows, n_cols, ratio):
    """均匀网格采样: 按ratio比例选择patch位置
    
    Returns: set of (row, col) selected positions
    """
    total = n_rows * n_cols
    n_keep = max(1, int(total * ratio))
    
    if ratio >= 1.0:
        return set((r, c) for r in range(n_rows) for c in range(n_cols))
    
    # 均匀采样: 每隔step取一个
    step = max(1, int(1.0 / ratio))
    selected = set()
    for r in range(0, n_rows, step):
        for c in range(0, n_cols, step):
            if len(selected) < n_keep:
                selected.add((r, c))
    
    # 如果不够, 补充
    if len(selected) < n_keep:
        for r in range(n_rows):
            for c in range(n_cols):
                if (r, c) not in selected and len(selected) < n_keep:
                    selected.add((r, c))
    
    return selected

def pack_and_transmit(img, rois, token_ratio, patch_size=PATCH_SIZE):
    """完整的patch压缩+传输流程
    
    1. 每个ROI: patchify → uniform select → pack
    2. 传输: 低分辨率背景 + packed JPEG + masks + ROI coords
    
    Returns:
        canvas: 云端恢复后的完整画布 (低分辨率背景 + sparse patches)
        payload_bytes: 实际传输字节数
    """
    h, w = img.shape[:2]
    
    # === 1. 低分辨率全局背景 ===
    small_w = max(1, w // LOW_RES_SCALE)
    small_h = max(1, h // LOW_RES_SCALE)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    canvas = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 低分辨率背景的JPEG
    buf = io.BytesIO()
    Image.fromarray(small).save(buf, format='JPEG', quality=95)
    bg_jpeg_bytes = len(buf.getvalue())
    
    total_payload = bg_jpeg_bytes
    
    # === 2. 每个ROI: patchify → select → pack ===
    all_packed_patches = []  # (patch_data, target_x, target_y)
    
    for roi in rois:
        rx1, ry1, rx2, ry2 = roi
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        
        patches, n_rows, n_cols = patchify_roi(img, roi, patch_size)
        if not patches:
            continue
        
        if token_ratio >= 1.0:
            # 100%: 全选
            selected_positions = set((r, c) for r, c, _ in patches)
        else:
            selected_positions = uniform_select(n_rows, n_cols, token_ratio)
        
        # 收集选中的patch + 在原图中的绝对坐标
        for r, c, patch_data in patches:
            if (r, c) in selected_positions:
                # patch在原图中的绝对位置
                abs_x = rx1 + c * patch_size
                abs_y = ry1 + r * patch_size
                all_packed_patches.append((patch_data, abs_x, abs_y))
    
    if not all_packed_patches:
        return canvas, total_payload + 4  # header only
    
    # === 3. Pack: 把选中的patch紧凑排列 ===
    n_patches = len(all_packed_patches)
    ps = patch_size
    
    # 计算packed canvas尺寸 (正方形排列最紧凑)
    packed_cols = int(np.ceil(np.sqrt(n_patches)))
    packed_rows = int(np.ceil(n_patches / packed_cols))
    
    packed_w = packed_cols * ps
    packed_h = packed_rows * ps
    
    # 创建packed canvas
    packed_canvas = np.full((packed_h, packed_w, 3), 114, dtype=np.uint8)
    
    for i, (patch_data, abs_x, abs_y) in enumerate(all_packed_patches):
        pr = i // packed_cols
        pc = i % packed_cols
        packed_canvas[pr*ps:(pr+1)*ps, pc*ps:(pc+1)*ps] = patch_data
    
    # === 4. JPEG编码packed canvas (一次!) ===
    buf = io.BytesIO()
    Image.fromarray(packed_canvas).save(buf, format='JPEG', quality=85)
    packed_jpeg_bytes = len(buf.getvalue())
    packed_jpeg_data = buf.getvalue()
    
    total_payload += packed_jpeg_bytes
    
    # === 5. metadata: 每个patch的坐标 ===
    # 每个patch需要: abs_x (uint16), abs_y (uint16) = 4 bytes
    # + n_patches header (4 bytes)
    coord_bytes = n_patches * 4 + 4
    total_payload += coord_bytes
    
    # === 6. 云端恢复 ===
    # 解码packed JPEG
    packed_decoded = np.array(Image.open(io.BytesIO(packed_jpeg_data)).convert('RGB'))
    
    # 用坐标把patch放回canvas
    for i, (patch_data, abs_x, abs_y) in enumerate(all_packed_patches):
        pr = i // packed_cols
        pc = i % packed_cols
        decoded_patch = packed_decoded[pr*ps:(pr+1)*ps, pc*ps:(pc+1)*ps]
        
        # 放回原图坐标位置 (处理边界)
        ph, pw = min(ps, h - abs_y), min(ps, w - abs_x)
        if ph > 0 and pw > 0:
            canvas[abs_y:abs_y+ph, abs_x:abs_x+pw] = decoded_patch[:ph, :pw]
    
    return canvas, total_payload

def make_lowres_canvas_full(img, rois):
    """完整ROI画布 (D0基线, 不做patch)"""
    h, w = img.shape[:2]
    small_w = max(1, w // LOW_RES_SCALE)
    small_h = max(1, h // LOW_RES_SCALE)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    canvas = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    for rx1, ry1, rx2, ry2 in rois:
        if rx2 > rx1 and ry2 > ry1:
            canvas[ry1:ry2, rx1:rx2] = img[ry1:ry2, rx1:rx2]
    return canvas

def calc_payload_full_roi(img, rois):
    """D0: 完整ROI的传输量 = 低分辨率JPEG + 每个ROI的JPEG"""
    h, w = img.shape[:2]
    total = 0
    small_w = max(1, w // LOW_RES_SCALE)
    small_h = max(1, h // LOW_RES_SCALE)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(small).save(buf, format='JPEG', quality=95)
    total += len(buf.getvalue())
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

def evaluate(results_list, name, img_ids_used):
    if not results_list:
        return {"AP50": 0, "AP": 0, "AR": 0, "APsmall": 0, "APmedium": 0, "APlarge": 0, "AP75": 0, "num_dets": 0}
    tmp = f"/tmp/expD2_{name}.json"
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
    print("Experiment D v2: Patch-level Evidence Compression")
    print("通信模型: Packed Patch Canvas + 一次JPEG + 坐标metadata")
    print("选择策略: Uniform grid sampling")
    print("=" * 70)
    
    with open(GT_PATH) as f:
        gt_coco = json.load(f)
    images = gt_coco["images"][:args.max_images]
    print(f"图片数: {len(images)}")
    
    model = YOLO(YOLO_MODEL_PATH)
    
    ratios = [
        ("D0_fullroi", None),
        ("D1_100pct", 1.0),
        ("D2_50pct", 0.50),
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
                # D0: 完整ROI
                canvas = make_lowres_canvas_full(img, all_rois)
                total_payloads[name] += calc_payload_full_roi(img, all_rois)
            else:
                # D1~D5: packed patch compression
                canvas, payload = pack_and_transmit(img, all_rois, ratio)
                total_payloads[name] += payload
            
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
    
    n = len(img_ids_used)
    print(f"\n{'方案':<18} {'AP50':<8} {'AP':<8} {'APsmall':<8} {'APmed':<8} {'APlarge':<8} {'AR':<8} {'检测数':<8} {'传输量KB':<10}")
    print("-" * 95)
    for name, ratio in ratios:
        m = metrics[name]
        avg_kb = total_payloads[name] / n / 1024
        print(f"{name:<18} {m['AP50']:<8.4f} {m['AP']:<8.4f} {m['APsmall']:<8.4f} {m['APmedium']:<8.4f} {m['APlarge']:<8.4f} {m['AR']:<8.4f} {m['num_dets']:<8} {avg_kb:<10.1f}")
    
    # 相对D0
    base_ap50 = metrics["D0_fullroi"]["AP50"]
    base_apsmall = metrics["D0_fullroi"]["APsmall"]
    base_payload = total_payloads["D0_fullroi"]
    
    print(f"\n相对完整ROI (D0) 的保持率:")
    print(f"{'方案':<18} {'AP50':<10} {'APsmall':<10} {'Payload':<10}")
    print("-" * 50)
    for name, ratio in ratios:
        m = metrics[name]
        if base_ap50 > 0:
            p_ap50 = m['AP50']/base_ap50*100
            p_apsmall = m['APsmall']/base_apsmall*100 if base_apsmall > 0 else 0
            p_payload = total_payloads[name]/base_payload*100 if base_payload > 0 else 0
            print(f"{name:<18} {p_ap50:<9.1f}% {p_apsmall:<9.1f}% {p_payload:<9.1f}%")
    
    with open("/app/Salience-DETR/experiments/expD_token_v2.json", "w") as f:
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
                "selection": "uniform_grid",
                "serialization": "packed_canvas_jpeg + coordinates",
            },
        }, f, indent=2)
    
    print(f"\n结果已保存: expD_token_v2.json")

if __name__ == "__main__":
    main()
