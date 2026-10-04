#!/usr/bin/env python3
"""Experiment F: 补充实验

F1: Edge-miss Recovery Rate (边缘漏检恢复率)
F2: TP/FP/FN/Precision/Recall (Uniform 25% vs Evidence-aware 25%)
F3: External Baselines (完整图传输 / 仅低分辨率 / 随机ROI)
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
LAMBDA_S = 0.6
LAMBDA_E = 0.4
IOU_THRESHOLD = 0.5

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

def compute_iou(box1, box2):
    """计算两个xyxy框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)

def match_detections_to_gt(detections, gt_boxes, iou_thresh=IOU_THRESHOLD):
    """将检测结果匹配到GT，返回 (matched_gt_indices, unmatched_det_indices)"""
    matched_gt = set()
    unmatched_det = []
    
    for di, det in enumerate(detections):
        best_iou = 0
        best_gt = -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = compute_iou(det, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = gi
        if best_iou >= iou_thresh and best_gt >= 0:
            matched_gt.add(best_gt)
        else:
            unmatched_det.append(di)
    
    return matched_gt, unmatched_det

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

def generate_random_rois(img_w, img_h, topk=TOPK_EXPLORE):
    """随机选择网格区域作为baseline"""
    cells = []
    grid_w = img_w / GRID_COLS
    grid_h = img_h / GRID_ROWS
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            gx1 = int(c * grid_w); gy1 = int(r * grid_h)
            gx2 = int((c + 1) * grid_w); gy2 = int((r + 1) * grid_h)
            cells.append((gx1, gy1, gx2, gy2))
    import random
    random.seed(42)
    return random.sample(cells, min(topk, len(cells)))

def get_roi_centers(rois):
    return [((r[0]+r[2])/2, (r[1]+r[3])/2) for r in rois]

def make_lowres_canvas(img, rois):
    h, w = img.shape[:2]
    small_w = max(1, w // LOW_RES_SCALE)
    small_h = max(1, h // LOW_RES_SCALE)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    canvas = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    for rx1, ry1, rx2, ry2 in rois:
        if rx2 > rx1 and ry2 > ry1:
            canvas[ry1:ry2, rx1:rx2] = img[ry1:ry2, rx1:rx2]
    return canvas

def make_lowres_only(img):
    """仅低分辨率背景，无高分辨率证据"""
    h, w = img.shape[:2]
    small_w = max(1, w // LOW_RES_SCALE)
    small_h = max(1, h // LOW_RES_SCALE)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    canvas = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return canvas

def patchify_roi(img, roi, patch_size=PATCH_SIZE):
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
            patch = img[py:py2, px:px2]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                pad_h = patch_size - patch.shape[0]
                pad_w = patch_size - patch.shape[1]
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
            patches.append((r, c, patch, px, py))
    return patches, n_rows, n_cols

def uniform_select(n_rows, n_cols, ratio):
    total = n_rows * n_cols
    n_keep = max(1, int(total * ratio))
    if ratio >= 1.0:
        return set((r, c) for r in range(n_rows) for c in range(n_cols))
    step = max(1, int(1.0 / ratio))
    selected = set()
    for r in range(0, n_rows, step):
        for c in range(0, n_cols, step):
            if len(selected) < n_keep:
                selected.add((r, c))
    if len(selected) < n_keep:
        for r in range(n_rows):
            for c in range(n_cols):
                if (r, c) not in selected and len(selected) < n_keep:
                    selected.add((r, c))
    return selected

def evidence_aware_select(patches_info, sup_centers, exp_centers, img_w, img_h, ratio):
    max_dist = (img_w**2 + img_h**2) ** 0.5
    n_total = len(patches_info)
    n_keep = max(1, int(n_total * ratio))
    scored = []
    for p in patches_info:
        r, c, patch_data, abs_x, abs_y = p
        pcx = abs_x + PATCH_SIZE / 2
        pcy = abs_y + PATCH_SIZE / 2
        if sup_centers:
            dist_s = min(((sc[0]-pcx)**2 + (sc[1]-pcy)**2)**0.5 for sc in sup_centers)
            s_support = 1 - dist_s / max_dist
        else:
            s_support = 0
        if exp_centers:
            dist_e = min(((ec[0]-pcx)**2 + (ec[1]-pcy)**2)**0.5 for ec in exp_centers)
            s_exploration = 1 - dist_e / max_dist
        else:
            s_exploration = 0
        score = LAMBDA_S * s_support + LAMBDA_E * s_exploration
        scored.append((r, c, patch_data, abs_x, abs_y, score))
    scored.sort(key=lambda x: x[5], reverse=True)
    return set((x[0], x[1]) for x in scored[:n_keep])

def pack_and_transmit(img, rois, token_ratio, selection_mode="uniform",
                      sup_centers=None, exp_centers=None):
    h, w = img.shape[:2]
    small_w = max(1, w // LOW_RES_SCALE)
    small_h = max(1, h // LOW_RES_SCALE)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    canvas = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    total_payload = 0
    buf = io.BytesIO()
    Image.fromarray(small).save(buf, format='JPEG', quality=95)
    total_payload += len(buf.getvalue())
    all_packed_patches = []
    for roi in rois:
        rx1, ry1, rx2, ry2 = roi
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        patches, n_rows, n_cols = patchify_roi(img, roi)
        if not patches:
            continue
        patches_info = [(p[0], p[1], p[2], p[3], p[4]) for p in patches]
        if token_ratio >= 1.0:
            selected_positions = set((r, c) for r, c, _, _, _ in patches_info)
        elif selection_mode == "uniform":
            selected_positions = uniform_select(n_rows, n_cols, token_ratio)
        elif selection_mode == "evidence":
            selected_positions = evidence_aware_select(
                patches_info, sup_centers, exp_centers, w, h, token_ratio)
        else:
            selected_positions = uniform_select(n_rows, n_cols, token_ratio)
        for r, c, patch_data, abs_x, abs_y in patches:
            if (r, c) in selected_positions:
                all_packed_patches.append((patch_data, abs_x, abs_y))
    if not all_packed_patches:
        return canvas, total_payload + 4
    n_patches = len(all_packed_patches)
    ps = PATCH_SIZE
    packed_cols = int(np.ceil(np.sqrt(n_patches)))
    packed_rows = int(np.ceil(n_patches / packed_cols))
    packed_w = packed_cols * ps
    packed_h = packed_rows * ps
    packed_canvas = np.full((packed_h, packed_w, 3), 114, dtype=np.uint8)
    for i, (patch_data, abs_x, abs_y) in enumerate(all_packed_patches):
        pr = i // packed_cols
        pc = i % packed_cols
        packed_canvas[pr*ps:(pr+1)*ps, pc*ps:(pc+1)*ps] = patch_data
    buf = io.BytesIO()
    Image.fromarray(packed_canvas).save(buf, format='JPEG', quality=85)
    packed_jpeg_data = buf.getvalue()
    total_payload += len(packed_jpeg_data)
    total_payload += n_patches * 4 + 4
    packed_decoded = np.array(Image.open(io.BytesIO(packed_jpeg_data)).convert('RGB'))
    for i, (patch_data, abs_x, abs_y) in enumerate(all_packed_patches):
        pr = i // packed_cols
        pc = i % packed_cols
        decoded_patch = packed_decoded[pr*ps:(pr+1)*ps, pc*ps:(pc+1)*ps]
        ph, pw = min(ps, h - abs_y), min(ps, w - abs_x)
        if ph > 0 and pw > 0:
            canvas[abs_y:abs_y+ph, abs_x:abs_x+pw] = decoded_patch[:ph, :pw]
    return canvas, total_payload

def calc_payload_full_roi(img, rois):
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
        return {"AP50": 0, "AP": 0, "AR": 0, "APsmall": 0, "num_dets": 0}
    tmp = f"/tmp/expF_{name}.json"
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
    except:
        return {"AP50": 0, "AP": 0, "AR": 0, "APsmall": 0, "num_dets": len(results_list)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=1600)
    args = parser.parse_args()
    
    print("=" * 70)
    print("Experiment F: 补充实验")
    print("F1: Edge-miss Recovery Rate")
    print("F2: TP/FP/FN/Precision/Recall")
    print("F3: External Baselines")
    print("=" * 70)
    
    with open(GT_PATH) as f:
        gt_coco = json.load(f)
    images = gt_coco["images"][:args.max_images]
    print(f"图片数: {len(images)}")
    
    # 构建GT索引: image_id -> list of (bbox_xyxy, category_id)
    gt_by_img = {}
    for ann in gt_coco["annotations"]:
        img_id = ann["image_id"]
        x, y, w, h = ann["bbox"]
        bbox = [x, y, x+w, y+h]
        gt_by_img.setdefault(img_id, []).append((bbox, ann["category_id"]))
    
    model = YOLO(YOLO_MODEL_PATH)
    
    # ============================================================
    # F1 + F2: Edge-miss Recovery + TP/FP/FN
    # ============================================================
    
    # 统计
    total_gt = 0
    edge_matched = 0
    edge_missed = 0
    
    # F1: edge-miss recovery
    # 对Support+Exp和Support+Exp+Token25%(Uniform/Evidence)分别统计恢复
    recovery_sup_exp = {"matched": 0, "total_missed": 0}
    recovery_uniform25 = {"matched": 0, "total_missed": 0}
    recovery_evidence25 = {"matched": 0, "total_missed": 0}
    
    # F2: TP/FP/FN at multiple confidence thresholds
    conf_thresholds = [0.1, 0.2, 0.3, 0.5]
    tp_fp_fn = {
        "uniform25": {ct: {"tp": 0, "fp": 0, "fn": 0} for ct in conf_thresholds},
        "evidence25": {ct: {"tp": 0, "fp": 0, "fn": 0} for ct in conf_thresholds},
    }
    
    # F3: External baselines COCO results
    results_fullimg = []   # 完整图传云端
    results_lowres = []    # 仅低分辨率
    results_random_roi = [] # 随机ROI
    results_sup_exp = []   # Support+Exploration (参考)
    
    img_ids_used = set()
    
    for img_info in tqdm(images, desc="处理中"):
        img_id = img_info["id"]
        fname = img_info["file_name"]
        img_path = os.path.join(IMG_DIR, fname)
        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]
        
        # GT
        gt_list = gt_by_img.get(img_id, [])
        gt_boxes = [g[0] for g in gt_list]
        gt_cats = [g[1] for g in gt_list]
        total_gt += len(gt_boxes)
        
        # YOLO检测
        results_yolo = model(img_path, conf=0.01, verbose=False)
        r = results_yolo[0]
        
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().tolist()
            dets = [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in xyxy]
        else:
            dets = []
        
        # F1: 计算edge-missed GT
        edge_matched_gt, _ = match_detections_to_gt(dets, gt_boxes, IOU_THRESHOLD)
        edge_missed_gt = set(range(len(gt_boxes))) - edge_matched_gt
        edge_matched += len(edge_matched_gt)
        edge_missed += len(edge_missed_gt)
        
        # 构建Support + Exploration
        supports = make_support_rois(img, dets)
        explorations = generate_exploration_rois(supports, w, h)
        all_rois = supports + explorations
        sup_centers = get_roi_centers(supports)
        exp_centers = get_roi_centers(explorations)
        
        # === F3: External baselines ===
        
        # 3a: 完整图传云端
        cloud_full = send_cloud(img, 0.05)
        for d in cloud_full:
            b = d["bbox"]
            results_fullimg.append({
                "image_id": int(img_id),
                "category_id": int(d.get("class_id", 1)),
                "bbox": [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])],
                "score": float(d.get("confidence", 0)),
            })
        
        # 3b: 仅低分辨率
        canvas_lowres = make_lowres_only(img)
        cloud_lowres = send_cloud(canvas_lowres, 0.05)
        for d in cloud_lowres:
            b = d["bbox"]
            results_lowres.append({
                "image_id": int(img_id),
                "category_id": int(d.get("class_id", 1)),
                "bbox": [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])],
                "score": float(d.get("confidence", 0)),
            })
        
        # 3c: 随机ROI
        random_rois = generate_random_rois(w, h, TOPK_EXPLORE)
        all_random_rois = supports + random_rois  # Support + Random Exploration
        canvas_random = make_lowres_canvas(img, all_random_rois)
        cloud_random = send_cloud(canvas_random, 0.05)
        for d in cloud_random:
            b = d["bbox"]
            results_random_roi.append({
                "image_id": int(img_id),
                "category_id": int(d.get("class_id", 1)),
                "bbox": [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])],
                "score": float(d.get("confidence", 0)),
            })
        
        # 3d: Support+Exploration (参考)
        canvas_sup_exp = make_lowres_canvas(img, all_rois)
        cloud_sup_exp = send_cloud(canvas_sup_exp, 0.05)
        for d in cloud_sup_exp:
            b = d["bbox"]
            results_sup_exp.append({
                "image_id": int(img_id),
                "category_id": int(d.get("class_id", 1)),
                "bbox": [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])],
                "score": float(d.get("confidence", 0)),
            })
        
        # === F1: Edge-miss recovery ===
        # Support+Exploration的云端检测
        sup_exp_dets = [[d["bbox"][0], d["bbox"][1], d["bbox"][0]+d["bbox"][2], d["bbox"][1]+d["bbox"][3]]
                       for d in cloud_sup_exp if d.get("confidence", 0) >= 0.1]
        recovered_sup_exp, _ = match_detections_to_gt(sup_exp_dets, 
                                                       [gt_boxes[i] for i in edge_missed_gt],
                                                       IOU_THRESHOLD)
        recovery_sup_exp["matched"] += len(recovered_sup_exp)
        recovery_sup_exp["total_missed"] += len(edge_missed_gt)
        
        # === F2: Token 25% Uniform vs Evidence-aware ===
        canvas_u25, _ = pack_and_transmit(img, all_rois, 0.25, "uniform", sup_centers, exp_centers)
        cloud_u25 = send_cloud(canvas_u25, 0.05)
        
        canvas_e25, _ = pack_and_transmit(img, all_rois, 0.25, "evidence", sup_centers, exp_centers)
        cloud_e25 = send_cloud(canvas_e25, 0.05)
        
        # F1: recovery for token variants
        u25_dets = [[d["bbox"][0], d["bbox"][1], d["bbox"][0]+d["bbox"][2], d["bbox"][1]+d["bbox"][3]]
                    for d in cloud_u25 if d.get("confidence", 0) >= 0.1]
        recovered_u25, _ = match_detections_to_gt(u25_dets,
                                                    [gt_boxes[i] for i in edge_missed_gt],
                                                    IOU_THRESHOLD)
        recovery_uniform25["matched"] += len(recovered_u25)
        recovery_uniform25["total_missed"] += len(edge_missed_gt)
        
        e25_dets = [[d["bbox"][0], d["bbox"][1], d["bbox"][0]+d["bbox"][2], d["bbox"][1]+d["bbox"][3]]
                    for d in cloud_e25 if d.get("confidence", 0) >= 0.1]
        recovered_e25, _ = match_detections_to_gt(e25_dets,
                                                   [gt_boxes[i] for i in edge_missed_gt],
                                                   IOU_THRESHOLD)
        recovery_evidence25["matched"] += len(recovered_e25)
        recovery_evidence25["total_missed"] += len(edge_missed_gt)
        
        # F2: TP/FP/FN at multiple thresholds
        for ct in conf_thresholds:
            for method, cloud_dets in [("uniform25", cloud_u25), ("evidence25", cloud_e25)]:
                filtered_dets = [[d["bbox"][0], d["bbox"][1], d["bbox"][0]+d["bbox"][2], d["bbox"][1]+d["bbox"][3]]
                                for d in cloud_dets if d.get("confidence", 0) >= ct]
                matched_gt, unmatched_dets = match_detections_to_gt(filtered_dets, gt_boxes, IOU_THRESHOLD)
                tp = len(matched_gt)
                fp = len(unmatched_dets)
                fn = len(gt_boxes) - tp
                tp_fp_fn[method][ct]["tp"] += tp
                tp_fp_fn[method][ct]["fp"] += fp
                tp_fp_fn[method][ct]["fn"] += fn
        
        img_ids_used.add(int(img_id))
    
    n = len(img_ids_used)
    
    # ============================================================
    # 输出结果
    # ============================================================
    
    print("\n" + "=" * 70)
    print("F1: Edge-miss Recovery Rate")
    print("=" * 70)
    
    print(f"\n总GT目标数: {total_gt}")
    print(f"边缘端匹配数: {edge_matched} ({edge_matched/total_gt*100:.1f}%)")
    print(f"边缘端漏检数: {edge_missed} ({edge_missed/total_gt*100:.1f}%)")
    
    r_sup_exp = recovery_sup_exp["matched"] / recovery_sup_exp["total_missed"] * 100 if recovery_sup_exp["total_missed"] > 0 else 0
    r_u25 = recovery_uniform25["matched"] / recovery_uniform25["total_missed"] * 100 if recovery_uniform25["total_missed"] > 0 else 0
    r_e25 = recovery_evidence25["matched"] / recovery_evidence25["total_missed"] * 100 if recovery_evidence25["total_missed"] > 0 else 0
    
    print(f"\n{'方法':<30} {'恢复数':<10} {'漏检总数':<10} {'恢复率':<10}")
    print("-" * 65)
    print(f"{'Support+Exploration':<30} {recovery_sup_exp['matched']:<10} {recovery_sup_exp['total_missed']:<10} {r_sup_exp:<10.1f}%")
    print(f"{'+Token 25% (Uniform)':<30} {recovery_uniform25['matched']:<10} {recovery_uniform25['total_missed']:<10} {r_u25:<10.1f}%")
    print(f"{'+Token 25% (Evidence-aware)':<30} {recovery_evidence25['matched']:<10} {recovery_evidence25['total_missed']:<10} {r_e25:<10.1f}%")
    
    print("\n" + "=" * 70)
    print("F2: TP/FP/FN/Precision/Recall (25% Token)")
    print("=" * 70)
    
    print(f"\n{'Conf阈值':<10} {'方法':<15} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<12} {'Recall':<12}")
    print("-" * 80)
    for ct in conf_thresholds:
        for method in ["uniform25", "evidence25"]:
            s = tp_fp_fn[method][ct]
            precision = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0
            recall = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0
            label = "Uniform" if method == "uniform25" else "Evidence"
            print(f"{ct:<10} {label:<15} {s['tp']:<8} {s['fp']:<8} {s['fn']:<8} {precision:<12.4f} {recall:<12.4f}")
    
    print("\n" + "=" * 70)
    print("F3: External Baselines (COCO评估)")
    print("=" * 70)
    
    m_full = evaluate(results_fullimg, "fullimg", img_ids_used)
    m_lowres = evaluate(results_lowres, "lowres", img_ids_used)
    m_random = evaluate(results_random_roi, "random_roi", img_ids_used)
    m_sup_exp = evaluate(results_sup_exp, "sup_exp", img_ids_used)
    
    print(f"\n{'方法':<30} {'AP50':<10} {'mAP':<10} {'APsmall':<10} {'AR':<10} {'检测数':<10}")
    print("-" * 80)
    print(f"{'Full image (上界)':<30} {m_full['AP50']:<10.4f} {m_full['AP']:<10.4f} {m_full['APsmall']:<10.4f} {m_full['AR']:<10.4f} {m_full['num_dets']:<10}")
    print(f"{'Low-res only (无证据)':<30} {m_lowres['AP50']:<10.4f} {m_lowres['AP']:<10.4f} {m_lowres['APsmall']:<10.4f} {m_lowres['AR']:<10.4f} {m_lowres['num_dets']:<10}")
    print(f"{'Random ROI (Support+Random)':<30} {m_random['AP50']:<10.4f} {m_random['AP']:<10.4f} {m_random['APsmall']:<10.4f} {m_random['AR']:<10.4f} {m_random['num_dets']:<10}")
    print(f"{'Support+Exploration (本文)':<30} {m_sup_exp['AP50']:<10.4f} {m_sup_exp['AP']:<10.4f} {m_sup_exp['APsmall']:<10.4f} {m_sup_exp['AR']:<10.4f} {m_sup_exp['num_dets']:<10}")
    
    # 保存
    with open("/app/Salience-DETR/experiments/expF_supplementary.json", "w") as f:
        json.dump({
            "F1_edge_miss_recovery": {
                "total_gt": total_gt,
                "edge_matched": edge_matched,
                "edge_missed": edge_missed,
                "recovery": {
                    "support_exploration": {"matched": recovery_sup_exp["matched"], "total_missed": recovery_sup_exp["total_missed"], "rate": r_sup_exp},
                    "uniform_25": {"matched": recovery_uniform25["matched"], "total_missed": recovery_uniform25["total_missed"], "rate": r_u25},
                    "evidence_25": {"matched": recovery_evidence25["matched"], "total_missed": recovery_evidence25["total_missed"], "rate": r_e25},
                }
            },
            "F2_tp_fp_fn": tp_fp_fn,
            "F3_external_baselines": {
                "full_image": m_full,
                "low_res_only": m_lowres,
                "random_roi": m_random,
                "support_exploration": m_sup_exp,
            },
            "num_images": n,
        }, f, indent=2)
    
    print(f"\n结果已保存: expF_supplementary.json")

if __name__ == "__main__":
    main()
