#!/usr/bin/env python3
"""修复 F1 + F2: Edge-miss Recovery + TP/FP/FN"""
import json, os, io, argparse
import numpy as np
import cv2
import requests
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm

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

def match_tp_fp(detections, gt_boxes, iou_thresh=IOU_THRESHOLD):
    """Hungarian-style matching: 按置信度排序，逐个分配最优GT
    
    Returns: (tp_count, fp_count, matched_gt_indices_set)
    """
    # detections: list of [x1, y1, x2, y2, score]
    # gt_boxes: list of [x1, y1, x2, y2]
    
    # 按score降序排列
    detections_sorted = sorted(detections, key=lambda x: x[4], reverse=True)
    
    matched_gt = set()
    tp = 0
    fp = 0
    
    for det in detections_sorted:
        det_box = det[:4]
        best_iou = 0
        best_gt = -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = compute_iou(det_box, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = gi
        if best_iou >= iou_thresh and best_gt >= 0:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1
    
    return tp, fp, matched_gt

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
        return canvas, 0
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
    packed_decoded = np.array(Image.open(io.BytesIO(packed_jpeg_data)).convert('RGB'))
    for i, (patch_data, abs_x, abs_y) in enumerate(all_packed_patches):
        pr = i // packed_cols
        pc = i % packed_cols
        decoded_patch = packed_decoded[pr*ps:(pr+1)*ps, pc*ps:(pc+1)*ps]
        ph, pw = min(ps, h - abs_y), min(ps, w - abs_x)
        if ph > 0 and pw > 0:
            canvas[abs_y:abs_y+ph, abs_x:abs_x+pw] = decoded_patch[:ph, :pw]
    return canvas, len(packed_jpeg_data) + n_patches * 4 + 4

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=1600)
    args = parser.parse_args()
    
    print("=" * 70)
    print("F1+F2 修复版: Edge-miss Recovery + TP/FP/FN")
    print("=" * 70)
    
    with open(GT_PATH) as f:
        gt_coco = json.load(f)
    images = gt_coco["images"][:args.max_images]
    print(f"图片数: {len(images)}")
    
    gt_by_img = {}
    for ann in gt_coco["annotations"]:
        img_id = ann["image_id"]
        x, y, w, h = ann["bbox"]
        bbox = [x, y, x+w, y+h]
        gt_by_img.setdefault(img_id, []).append(bbox)
    
    model = YOLO(YOLO_MODEL_PATH)
    
    # 统计
    total_gt = 0
    total_edge_det = 0
    edge_missed_total = 0
    
    # 恢复率统计
    recovery_stats = {
        "support_only": {"matched_missed": 0, "total_missed": 0},
        "support_exp": {"matched_missed": 0, "total_missed": 0},
        "token_uniform_25": {"matched_missed": 0, "total_missed": 0},
        "token_evidence_25": {"matched_missed": 0, "total_missed": 0},
    }
    
    # TP/FP/FN at confidence=0.1 (和COCO评估一致)
    conf_thresh = 0.1
    stats_pr = {
        "support_only": {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0},
        "support_exp": {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0},
        "token_uniform_25": {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0},
        "token_evidence_25": {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0},
    }
    
    for img_info in tqdm(images, desc="处理中"):
        img_id = img_info["id"]
        fname = img_info["file_name"]
        img_path = os.path.join(IMG_DIR, fname)
        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]
        
        gt_boxes = gt_by_img.get(img_id, [])
        total_gt += len(gt_boxes)
        
        if len(gt_boxes) == 0:
            continue
        
        # YOLO检测 (conf=0.01)
        results_yolo = model(img_path, conf=0.01, verbose=False)
        r = results_yolo[0]
        
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().tolist()
            confs = r.boxes.conf.cpu().numpy().tolist()
            dets = [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in xyxy]
            dets_with_score = [[float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(c)]
                              for x, c in zip(xyxy, confs)]
        else:
            dets = []
            dets_with_score = []
        
        total_edge_det += len(dets)
        
        # 计算edge端匹配（IoU=0.5, 按score排序匹配）
        edge_tp, edge_fp, edge_matched_gt_idx = match_tp_fp(dets_with_score, gt_boxes, IOU_THRESHOLD)
        edge_missed_idx = set(range(len(gt_boxes))) - edge_matched_gt_idx
        edge_missed_total += len(edge_missed_idx)
        
        # 漏检GT的boxes
        missed_gt_boxes = [gt_boxes[i] for i in edge_missed_idx]
        
        supports = make_support_rois(img, dets)
        explorations = generate_exploration_rois(supports, w, h)
        all_rois = supports + explorations
        sup_centers = get_roi_centers(supports)
        exp_centers = get_roi_centers(explorations)
        
        # === 方法1: Support only ===
        canvas_sup = make_lowres_canvas(img, supports)
        cloud_sup = send_cloud(canvas_sup, conf_thresh)
        sup_dets = [[d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3], d.get("confidence", 0)]
                    for d in cloud_sup if d.get("confidence", 0) >= conf_thresh]
        
        tp, fp, matched_gt = match_tp_fp(sup_dets, gt_boxes, IOU_THRESHOLD)
        stats_pr["support_only"]["tp"] += tp
        stats_pr["support_only"]["fp"] += fp
        stats_pr["support_only"]["fn"] += len(gt_boxes) - tp
        stats_pr["support_only"]["total_gt"] += len(gt_boxes)
        
        # 漏检恢复: 看edge漏检的GT中，有多少被云端检测到
        if missed_gt_boxes:
            tp_missed, _, _ = match_tp_fp(sup_dets, missed_gt_boxes, IOU_THRESHOLD)
            recovery_stats["support_only"]["matched_missed"] += tp_missed
            recovery_stats["support_only"]["total_missed"] += len(missed_gt_boxes)
        
        # === 方法2: Support + Exploration ===
        canvas_sup_exp = make_lowres_canvas(img, all_rois)
        cloud_sup_exp = send_cloud(canvas_sup_exp, conf_thresh)
        sup_exp_dets = [[d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3], d.get("confidence", 0)]
                        for d in cloud_sup_exp if d.get("confidence", 0) >= conf_thresh]
        
        tp, fp, matched_gt = match_tp_fp(sup_exp_dets, gt_boxes, IOU_THRESHOLD)
        stats_pr["support_exp"]["tp"] += tp
        stats_pr["support_exp"]["fp"] += fp
        stats_pr["support_exp"]["fn"] += len(gt_boxes) - tp
        stats_pr["support_exp"]["total_gt"] += len(gt_boxes)
        
        if missed_gt_boxes:
            tp_missed, _, _ = match_tp_fp(sup_exp_dets, missed_gt_boxes, IOU_THRESHOLD)
            recovery_stats["support_exp"]["matched_missed"] += tp_missed
            recovery_stats["support_exp"]["total_missed"] += len(missed_gt_boxes)
        
        # === 方法3: Token 25% Uniform ===
        canvas_u25, _ = pack_and_transmit(img, all_rois, 0.25, "uniform", sup_centers, exp_centers)
        cloud_u25 = send_cloud(canvas_u25, conf_thresh)
        u25_dets = [[d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3], d.get("confidence", 0)]
                   for d in cloud_u25 if d.get("confidence", 0) >= conf_thresh]
        
        tp, fp, matched_gt = match_tp_fp(u25_dets, gt_boxes, IOU_THRESHOLD)
        stats_pr["token_uniform_25"]["tp"] += tp
        stats_pr["token_uniform_25"]["fp"] += fp
        stats_pr["token_uniform_25"]["fn"] += len(gt_boxes) - tp
        stats_pr["token_uniform_25"]["total_gt"] += len(gt_boxes)
        
        if missed_gt_boxes:
            tp_missed, _, _ = match_tp_fp(u25_dets, missed_gt_boxes, IOU_THRESHOLD)
            recovery_stats["token_uniform_25"]["matched_missed"] += tp_missed
            recovery_stats["token_uniform_25"]["total_missed"] += len(missed_gt_boxes)
        
        # === 方法4: Token 25% Evidence-aware ===
        canvas_e25, _ = pack_and_transmit(img, all_rois, 0.25, "evidence", sup_centers, exp_centers)
        cloud_e25 = send_cloud(canvas_e25, conf_thresh)
        e25_dets = [[d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3], d.get("confidence", 0)]
                   for d in cloud_e25 if d.get("confidence", 0) >= conf_thresh]
        
        tp, fp, matched_gt = match_tp_fp(e25_dets, gt_boxes, IOU_THRESHOLD)
        stats_pr["token_evidence_25"]["tp"] += tp
        stats_pr["token_evidence_25"]["fp"] += fp
        stats_pr["token_evidence_25"]["fn"] += len(gt_boxes) - tp
        stats_pr["token_evidence_25"]["total_gt"] += len(gt_boxes)
        
        if missed_gt_boxes:
            tp_missed, _, _ = match_tp_fp(e25_dets, missed_gt_boxes, IOU_THRESHOLD)
            recovery_stats["token_evidence_25"]["matched_missed"] += tp_missed
            recovery_stats["token_evidence_25"]["total_missed"] += len(missed_gt_boxes)
    
    # ============================================================
    # 输出
    # ============================================================
    print("\n" + "=" * 70)
    print("F1: Edge-miss Recovery Rate (IoU=0.5)")
    print("=" * 70)
    
    print(f"\n总GT目标数: {total_gt}")
    print(f"边缘端检测数(conf≥0.01): {total_edge_det}")
    print(f"边缘端漏检数: {edge_missed_total} ({edge_missed_total/total_gt*100:.1f}%)")
    print(f"  (YOLO检测后，IoU≥0.5未匹配到GT的目标数)")
    
    print(f"\n{'方法':<30} {'恢复数':<10} {'漏检总数':<10} {'恢复率':<12}")
    print("-" * 70)
    
    labels = {
        "support_only": "Support only",
        "support_exp": "Support + Exploration",
        "token_uniform_25": "+Token 25% (Uniform)",
        "token_evidence_25": "+Token 25% (Evidence-aware)",
    }
    for key in ["support_only", "support_exp", "token_uniform_25", "token_evidence_25"]:
        s = recovery_stats[key]
        rate = s["matched_missed"] / s["total_missed"] * 100 if s["total_missed"] > 0 else 0
        print(f"{labels[key]:<30} {s['matched_missed']:<10} {s['total_missed']:<10} {rate:<12.1f}%")
    
    print("\n" + "=" * 70)
    print("F2: Precision/Recall at confidence=0.1")
    print("=" * 70)
    
    print(f"\n{'方法':<30} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<12} {'Recall':<12}")
    print("-" * 80)
    for key in ["support_only", "support_exp", "token_uniform_25", "token_evidence_25"]:
        s = stats_pr[key]
        precision = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0
        recall = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0
        print(f"{labels[key]:<30} {s['tp']:<8} {s['fp']:<8} {s['fn']:<8} {precision:<12.4f} {recall:<12.4f}")
    
    print(f"\nUniform vs Evidence-aware (25% token):")
    su = stats_pr["token_uniform_25"]
    se = stats_pr["token_evidence_25"]
    p_u = su["tp"] / (su["tp"] + su["fp"]) if (su["tp"] + su["fp"]) > 0 else 0
    p_e = se["tp"] / (se["tp"] + se["fp"]) if (se["tp"] + se["fp"]) > 0 else 0
    r_u = su["tp"] / (su["tp"] + su["fn"]) if (su["tp"] + su["fn"]) > 0 else 0
    r_e = se["tp"] / (se["tp"] + se["fn"]) if (se["tp"] + se["fn"]) > 0 else 0
    print(f"  Precision: {p_u:.4f} → {p_e:.4f} ({(p_e/p_u-1)*100:+.1f}%)")
    print(f"  Recall:    {r_u:.4f} → {r_e:.4f} ({(r_e/r_u-1)*100:+.1f}%)")
    print(f"  FP:        {su['fp']} → {se['fp']} ({(se['fp']/su['fp']-1)*100:+.1f}%)")
    print(f"  TP:        {su['tp']} → {se['tp']} ({(se['tp']/su['tp']-1)*100:+.1f}%)")
    
    # 保存
    with open("/app/Salience-DETR/experiments/expF1_fix.json", "w") as f:
        json.dump({
            "F1_recovery": {
                "total_gt": total_gt,
                "edge_missed": edge_missed_total,
                "edge_missed_pct": edge_missed_total / total_gt * 100,
                "methods": {key: {
                    "matched_missed": recovery_stats[key]["matched_missed"],
                    "total_missed": recovery_stats[key]["total_missed"],
                    "recovery_rate": recovery_stats[key]["matched_missed"] / recovery_stats[key]["total_missed"] * 100 if recovery_stats[key]["total_missed"] > 0 else 0,
                } for key in recovery_stats},
            },
            "F2_precision_recall": {
                "conf_threshold": conf_thresh,
                "methods": {key: {
                    "tp": stats_pr[key]["tp"],
                    "fp": stats_pr[key]["fp"],
                    "fn": stats_pr[key]["fn"],
                    "precision": stats_pr[key]["tp"] / (stats_pr[key]["tp"] + stats_pr[key]["fp"]) if (stats_pr[key]["tp"] + stats_pr[key]["fp"]) > 0 else 0,
                    "recall": stats_pr[key]["tp"] / (stats_pr[key]["tp"] + stats_pr[key]["fn"]) if (stats_pr[key]["tp"] + stats_pr[key]["fn"]) > 0 else 0,
                } for key in stats_pr},
            }
        }, f, indent=2)
    
    print(f"\n结果已保存: expF1_fix.json")

if __name__ == "__main__":
    main()
