#!/usr/bin/env python3
"""
补充实验最终版 Final Supplementary Experiments
基于 expF_supplementary.py 的正确模式重写

S1: JPEG matched-rate baseline  (相同通信量下JPEG对比)
S2: Support-only @ matched rate  (扩大Support到311KB，验证Exploration价值)
S3: Full-image upper bound       (全图上传，性能上界)
S4: 3-seed robustness            (3种子统计稳定性)
S5: Per-class analysis           (类别分布分析)

用法:
  python exp_supplement_final.py --s3 --max-images 20
  python exp_supplement_final.py --s1 --s2 --s3 --full
  python exp_supplement_final.py --all --full
"""

import json, os, io, time, argparse
import numpy as np
import requests
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ==================== 配置 ====================
CLOUD_URL = "http://localhost:8000/detect"
GT_PATH = "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"
IMG_DIR = "/app/data/seadronessee/compressed_version/val2017/"
YOLO_PATH = "/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt"
OUTPUT_DIR = "/app/Salience-DETR/experiments/pce_results/results"

IOU_THRESHOLD = 0.5
EDGE_CONF_SPLIT = 0.7
CLOUD_CONF = 0.05

# Support / Exploration 参数
EXPAND_RATIO = 0.1
GRID_ROWS = 6
GRID_COLS = 10
ALPHA_SCORE = 0.3
BETA_SCORE = 0.7
TOPK_EXPLORE = 8
LAMBDA_S = 0.6
LAMBDA_E = 0.4
PATCH_SIZE = 16

# ==================== 基础工具函数 ====================
def send_cloud(img_np, conf=CLOUD_CONF):
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

def send_cloud_jpeg_bytes(jpeg_bytes, conf=CLOUD_CONF):
    buf = io.BytesIO(jpeg_bytes)
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

def cloud_to_list(cloud_raw):
    """云端返回的dict转[x1,y1,x2,y2,score,class]列表"""
    result = []
    for d in cloud_raw:
        b = d["bbox"]
        result.append([
            float(b[0]), float(b[1]),
            float(b[2]), float(b[3]),
            float(d["confidence"]), int(d["class_id"])
        ])
    return result

def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2-x1) * (y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)

def nms_fuse(edge_high, cloud_dets, iou_thresh=0.5):
    """NMS融合边缘高置信度 + 云端检测"""
    all_dets = list(edge_high) + list(cloud_dets)
    all_dets.sort(key=lambda x: x[4], reverse=True)
    keep = []
    for det in all_dets:
        ok = True
        for k in keep:
            if det[5] == k[5] and compute_iou(det[:4], k[:4]) > iou_thresh:
                ok = False
                break
        if ok:
            keep.append(det)
    return keep

def find_jpeg_quality(img_np, target_bytes, low=1, high=95):
    """二值搜索JPEG quality使输出接近target_bytes"""
    best_q, best_diff = 50, float('inf')
    for _ in range(8):
        mid = (low + high) // 2
        buf = io.BytesIO()
        Image.fromarray(img_np).save(buf, format='JPEG', quality=mid)
        size = len(buf.getvalue())
        diff = abs(size - target_bytes)
        if diff < best_diff:
            best_diff, best_q = diff, mid
        if size < target_bytes:
            low = mid + 1
        else:
            high = mid - 1
        if low >= high:
            break
    return best_q

def to_coco_preds(fused_dets, img_id):
    """转COCO预测格式"""
    results = []
    for d in fused_dets:
        results.append({
            "image_id": int(img_id),
            "category_id": int(d[5]),
            "bbox": [float(d[0]), float(d[1]), float(d[2]-d[0]), float(d[3]-d[1])],
            "score": float(d[4]),
        })
    return results

def evaluate(results_list, name, img_ids_used):
    """照搬 expF_supplementary.py 的 evaluate 函数"""
    if not results_list:
        return {"AP50": 0, "AP": 0, "AR": 0, "APsmall": 0, "AP75": 0, "num_dets": 0}
    tmp = f"/tmp/exp_supp_{name}_{int(time.time())}.json"
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
            "ARsmall": float(e.stats[8]),
            "num_dets": len(results_list),
        }
    except Exception as ex:
        print(f"  [EVAL ERROR] {ex}")
        return {"AP50": 0, "AP": 0, "AR": 0, "APsmall": 0, "AP75": 0, "num_dets": len(results_list)}
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def compute_miss_recovery(edge_dets, final_dets, gt_boxes):
    """计算漏检恢复率"""
    edge_miss = 0
    recovered = 0
    for gt in gt_boxes:
        hit_edge = False
        for d in edge_dets:
            if compute_iou(gt, d[:4]) >= IOU_THRESHOLD:
                hit_edge = True
                break
        if not hit_edge:
            edge_miss += 1
            for d in final_dets:
                if compute_iou(gt, d[:4]) >= IOU_THRESHOLD:
                    recovered += 1
                    break
    return edge_miss, recovered

# ==================== ROI / 画布构建 ====================
def make_support_rois(img, dets, expand_ratio=EXPAND_RATIO):
    """生成support ROI"""
    h, w = img.shape[:2]
    rois = []
    for d in dets:
        x1, y1, x2, y2 = d[:4]
        bw, bh = x2-x1, y2-y1
        dx = max(bw * expand_ratio, 16)
        dy = max(bh * expand_ratio, 16)
        rx1 = max(0, int(x1 - dx))
        ry1 = max(0, int(y1 - dy))
        rx2 = min(w, int(x2 + dx))
        ry2 = min(h, int(y2 + dy))
        rois.append([rx1, ry1, rx2, ry2])
    return rois

def generate_exploration_rois(supports, img_w, img_h, topk=TOPK_EXPLORE):
    """生成exploration ROI"""
    grid_w = img_w / GRID_COLS
    grid_h = img_h / GRID_ROWS
    max_dist = (img_w**2 + img_h**2) ** 0.5
    candidates = []

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            gx1, gy1 = int(c * grid_w), int(r * grid_h)
            gx2, gy2 = int((c+1) * grid_w), int((r+1) * grid_h)

            # coverage
            coverage = 0
            for s in supports:
                ix1 = max(gx1, s[0])
                iy1 = max(gy1, s[1])
                ix2 = min(gx2, s[2])
                iy2 = min(gy2, s[3])
                if ix2 > ix1 and iy2 > iy1:
                    coverage += (ix2-ix1) * (iy2-iy1)
            grid_area = (gx2-gx1) * (gy2-gy1)
            coverage = coverage / max(grid_area, 1)

            # proximity
            gcx, gcy = (gx1+gx2)/2, (gy1+gy2)/2
            min_dist = max_dist
            for s in supports:
                scx = (s[0]+s[2]) / 2
                scy = (s[1]+s[3]) / 2
                dist = ((scx-gcx)**2 + (scy-gcy)**2) ** 0.5
                if dist < min_dist:
                    min_dist = dist

            score = ALPHA_SCORE * (1 - coverage) + BETA_SCORE * (1 - min_dist / max_dist)
            candidates.append({'bbox': [gx1, gy1, gx2, gy2], 'score': score})

    candidates.sort(key=lambda x: x['score'], reverse=True)
    return [c['bbox'] for c in candidates[:topk]]

def build_roi_canvas(img, rois):
    """灰色背景 + ROI 区域"""
    h, w = img.shape[:2]
    canvas = np.full((h, w, 3), 110, dtype=np.uint8)
    for roi in rois:
        x1, y1, x2, y2 = roi
        canvas[y1:y2, x1:x2] = img[y1:y2, x1:x2]
    return canvas

def compute_roi_payload_kb(img, rois, edge_dets, quality=85):
    """估算ROI画布的JPEG payload (KB)"""
    canvas = build_roi_canvas(img, rois)
    buf = io.BytesIO()
    Image.fromarray(canvas).save(buf, format='JPEG', quality=quality)
    edge_bytes = 15 + 21 * len(edge_dets)
    total_bytes = len(buf.getvalue()) + edge_bytes
    return total_bytes / 1024

# ==================== S1: JPEG Matched-Rate ====================
def run_s1(images_info, gt_by_img, model, target_kb_list):
    print("\n" + "="*70)
    print("[S1] JPEG Matched-Rate Baseline")
    print("="*70)

    results = []
    for target_kb in target_kb_list:
        target_bytes = int(target_kb * 1024)
        print(f"\n  Target = {target_kb:.0f} KB")

        results_coco = []
        img_ids_used = set()
        total_miss, total_recovered, total_gt = 0, 0, 0
        all_payloads = []
        all_qualities = []

        for img_info in tqdm(images_info, desc=f"S1 @{target_kb:.0f}KB"):
            img_id = img_info["id"]
            fname = img_info["file_name"]
            img_path = os.path.join(IMG_DIR, fname)
            img = np.array(Image.open(img_path).convert("RGB"))

            gt_list = gt_by_img.get(img_id, [])
            gt_boxes = [g[0] for g in gt_list]
            total_gt += len(gt_boxes)

            # Edge detection
            r = model(img_path, conf=0.01, verbose=False)[0]
            edge_dets = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy().tolist()
                confs = r.boxes.conf.cpu().numpy().tolist()
                cls_ids = r.boxes.cls.cpu().numpy().tolist()
                for i in range(len(xyxy)):
                    edge_dets.append([
                        float(xyxy[i][0]), float(xyxy[i][1]),
                        float(xyxy[i][2]), float(xyxy[i][3]),
                        float(confs[i]), int(cls_ids[i]) + 1
                    ])

            edge_high = [d for d in edge_dets if d[4] >= EDGE_CONF_SPLIT]

            # 计算目标JPEG大小（扣除边缘预测开销）
            edge_pred_bytes = 15 + 21 * len(edge_dets)
            jpeg_target = max(1000, target_bytes - edge_pred_bytes)

            quality = find_jpeg_quality(img, jpeg_target)
            all_qualities.append(quality)

            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format='JPEG', quality=quality)
            jpeg_data = buf.getvalue()

            # Cloud detection from JPEG
            cloud_raw = send_cloud_jpeg_bytes(jpeg_data, CLOUD_CONF)
            cloud_dets = cloud_to_list(cloud_raw)

            # Fuse
            fused = nms_fuse(edge_high, cloud_dets)

            # COCO preds
            results_coco.extend(to_coco_preds(fused, img_id))
            img_ids_used.add(int(img_id))

            # Payload
            payload = edge_pred_bytes + len(jpeg_data)
            all_payloads.append(payload)

            # Miss recovery
            miss, rec = compute_miss_recovery(edge_dets, fused, gt_boxes)
            total_miss += miss
            total_recovered += rec

        metrics = evaluate(results_coco, f"s1_{target_kb:.0f}kb", img_ids_used)
        metrics["avg_payload_kb"] = float(np.mean(all_payloads)) / 1024
        metrics["avg_jpeg_quality"] = float(np.mean(all_qualities))
        metrics["edge_miss_total"] = total_miss
        metrics["recovered_total"] = total_recovered
        metrics["miss_recovery_recall"] = total_recovered / max(total_miss, 1)
        metrics["total_gt"] = total_gt
        metrics["experiment"] = f"S1_jpeg_{target_kb:.0f}KB"

        print(f"\n  AP50={metrics['AP50']:.4f}, mAP={metrics['AP']:.4f}, "
              f"AP_S={metrics['APsmall']:.4f}")
        print(f"  Actual payload={metrics['avg_payload_kb']:.1f} KB, "
              f"avg quality={metrics['avg_jpeg_quality']:.1f}")
        print(f"  Miss recovery={metrics['miss_recovery_recall']:.4f} "
              f"({total_recovered}/{total_miss})")

        results.append(metrics)

    return results

# ==================== S2: Support-only @ Matched Rate ====================
def run_s2(images_info, gt_by_img, model, target_kb=311.0):
    print("\n" + "="*70)
    print("[S2] Support-only @ Matched Rate (~311 KB)")
    print("="*70)

    def run_with_expand(imgs_info, expand_r):
        results_coco = []
        img_ids_used = set()
        total_miss, total_recovered, total_gt = 0, 0, 0
        all_payloads = []

        for img_info in imgs_info:
            img_id = img_info["id"]
            fname = img_info["file_name"]
            img_path = os.path.join(IMG_DIR, fname)
            img = np.array(Image.open(img_path).convert("RGB"))

            gt_list = gt_by_img.get(img_id, [])
            gt_boxes = [g[0] for g in gt_list]
            total_gt += len(gt_boxes)

            r = model(img_path, conf=0.01, verbose=False)[0]
            edge_dets = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy().tolist()
                confs = r.boxes.conf.cpu().numpy().tolist()
                cls_ids = r.boxes.cls.cpu().numpy().tolist()
                for i in range(len(xyxy)):
                    edge_dets.append([
                        float(xyxy[i][0]), float(xyxy[i][1]),
                        float(xyxy[i][2]), float(xyxy[i][3]),
                        float(confs[i]), int(cls_ids[i]) + 1
                    ])

            edge_high = [d for d in edge_dets if d[4] >= EDGE_CONF_SPLIT]

            # Support ROIs with expanded ratio
            support_rois = make_support_rois(img, edge_dets, expand_ratio=expand_r)
            canvas = build_roi_canvas(img, support_rois)

            cloud_raw = send_cloud(canvas, CLOUD_CONF)
            cloud_dets = cloud_to_list(cloud_raw)
            fused = nms_fuse(edge_high, cloud_dets)

            results_coco.extend(to_coco_preds(fused, img_id))
            img_ids_used.add(int(img_id))

            payload = compute_roi_payload_kb(img, support_rois, edge_dets)
            all_payloads.append(payload)

            miss, rec = compute_miss_recovery(edge_dets, fused, gt_boxes)
            total_miss += miss
            total_recovered += rec

        metrics = evaluate(results_coco, f"s2_exp{expand_r:.1f}", img_ids_used)
        metrics["avg_payload_kb"] = float(np.mean(all_payloads))
        metrics["edge_miss_total"] = total_miss
        metrics["recovered_total"] = total_recovered
        metrics["miss_recovery_recall"] = total_recovered / max(total_miss, 1)
        metrics["total_gt"] = total_gt
        return metrics

    # Step 1: 估算expand_ratio（前30张）
    n_est = min(30, len(images_info))
    est_imgs = images_info[:n_est]
    candidates = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    best_expand, best_diff = 0.1, float('inf')

    print("\n  [Step 1] 估算合适的expand_ratio...")
    for er in candidates:
        m = run_with_expand(est_imgs, er)
        payload = m["avg_payload_kb"]
        diff = abs(payload - target_kb)
        print(f"    expand={er:.1f}: payload={payload:.1f} KB (diff={diff:.1f})")
        if diff < best_diff:
            best_diff, best_expand = diff, er

    print(f"\n  [Step 2] 使用 expand_ratio={best_expand:.1f} 跑全量...")
    result = run_with_expand(images_info, best_expand)
    result["experiment"] = f"S2_support_matched_{target_kb:.0f}KB"
    result["expand_ratio_used"] = best_expand

    print(f"\n  结果: AP50={result['AP50']:.4f}, APsmall={result['APsmall']:.4f}, "
          f"payload={result['avg_payload_kb']:.1f} KB")
    return result

# ==================== S3: Full-Image Upper Bound ====================
def run_s3(images_info, gt_by_img, model):
    print("\n" + "="*70)
    print("[S3] Full-Image Upper Bound")
    print("="*70)

    results_coco = []
    img_ids_used = set()
    total_miss, total_recovered, total_gt = 0, 0, 0
    all_payloads = []

    for img_info in tqdm(images_info, desc="S3 Full Image"):
        img_id = img_info["id"]
        fname = img_info["file_name"]
        img_path = os.path.join(IMG_DIR, fname)
        img = np.array(Image.open(img_path).convert("RGB"))

        gt_list = gt_by_img.get(img_id, [])
        gt_boxes = [g[0] for g in gt_list]
        total_gt += len(gt_boxes)

        r = model(img_path, conf=0.01, verbose=False)[0]
        edge_dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().tolist()
            confs = r.boxes.conf.cpu().numpy().tolist()
            cls_ids = r.boxes.cls.cpu().numpy().tolist()
            for i in range(len(xyxy)):
                edge_dets.append([
                    float(xyxy[i][0]), float(xyxy[i][1]),
                    float(xyxy[i][2]), float(xyxy[i][3]),
                    float(confs[i]), int(cls_ids[i]) + 1
                ])

        edge_high = [d for d in edge_dets if d[4] >= EDGE_CONF_SPLIT]

        cloud_raw = send_cloud(img, CLOUD_CONF)
        cloud_dets = cloud_to_list(cloud_raw)
        fused = nms_fuse(edge_high, cloud_dets)

        results_coco.extend(to_coco_preds(fused, img_id))
        img_ids_used.add(int(img_id))

        # Payload: JPEG quality 95 + edge preds
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format='JPEG', quality=95)
        payload = (15 + 21 * len(edge_dets) + len(buf.getvalue())) / 1024
        all_payloads.append(payload)

        miss, rec = compute_miss_recovery(edge_dets, fused, gt_boxes)
        total_miss += miss
        total_recovered += rec

    metrics = evaluate(results_coco, "s3_fullimg", img_ids_used)
    metrics["avg_payload_kb"] = float(np.mean(all_payloads))
    metrics["edge_miss_total"] = total_miss
    metrics["recovered_total"] = total_recovered
    metrics["miss_recovery_recall"] = total_recovered / max(total_miss, 1)
    metrics["total_gt"] = total_gt
    metrics["experiment"] = "S3_full_image_upper_bound"

    print(f"\n  AP50={metrics['AP50']:.4f}, mAP={metrics['AP']:.4f}, "
          f"APsmall={metrics['APsmall']:.4f}")
    print(f"  Payload (JPEG95)={metrics['avg_payload_kb']:.0f} KB")
    print(f"  Miss recovery={metrics['miss_recovery_recall']:.4f} "
          f"({total_recovered}/{total_miss})")

    return metrics

# ==================== S4: 3-Seed Robustness ====================
def run_s4(images_info, gt_by_img, model, budget_kb=87.0, seeds=[2026, 2027, 2028]):
    print("\n" + "="*70)
    print("[S4] 3-Seed Robustness + Bootstrap 95% CI")
    print("="*70)

    def run_support_exp(imgs_info):
        """Support + Exploration (deterministic) with ROI canvas"""
        results_coco = []
        img_ids_used = set()
        total_miss, total_recovered, total_gt = 0, 0, 0
        all_payloads = []

        for img_info in imgs_info:
            img_id = img_info["id"]
            fname = img_info["file_name"]
            img_path = os.path.join(IMG_DIR, fname)
            img = np.array(Image.open(img_path).convert("RGB"))
            h, w = img.shape[:2]

            gt_list = gt_by_img.get(img_id, [])
            gt_boxes = [g[0] for g in gt_list]
            total_gt += len(gt_boxes)

            r = model(img_path, conf=0.01, verbose=False)[0]
            edge_dets = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy().tolist()
                confs = r.boxes.conf.cpu().numpy().tolist()
                cls_ids = r.boxes.cls.cpu().numpy().tolist()
                for i in range(len(xyxy)):
                    edge_dets.append([
                        float(xyxy[i][0]), float(xyxy[i][1]),
                        float(xyxy[i][2]), float(xyxy[i][3]),
                        float(confs[i]), int(cls_ids[i]) + 1
                    ])

            edge_high = [d for d in edge_dets if d[4] >= EDGE_CONF_SPLIT]

            support_rois = make_support_rois(img, edge_dets)
            exp_rois = generate_exploration_rois(support_rois, w, h)
            all_rois = support_rois + exp_rois

            canvas = build_roi_canvas(img, all_rois)
            cloud_raw = send_cloud(canvas, CLOUD_CONF)
            cloud_dets = cloud_to_list(cloud_raw)
            fused = nms_fuse(edge_high, cloud_dets)

            results_coco.extend(to_coco_preds(fused, img_id))
            img_ids_used.add(int(img_id))
            payload = compute_roi_payload_kb(img, all_rois, edge_dets)
            all_payloads.append(payload)

            miss, rec = compute_miss_recovery(edge_dets, fused, gt_boxes)
            total_miss += miss
            total_recovered += rec

        metrics = evaluate(results_coco, "s4_sup_exp", img_ids_used)
        metrics["avg_payload_kb"] = float(np.mean(all_payloads))
        metrics["edge_miss_total"] = total_miss
        metrics["recovered_total"] = total_recovered
        metrics["miss_recovery_recall"] = total_recovered / max(total_miss, 1)
        return metrics

    def run_random(imgs_info, seed):
        """Support + Random exploration (seeded)"""
        rng_state = np.random.get_state()
        np.random.seed(seed)

        results_coco = []
        img_ids_used = set()
        total_miss, total_recovered, total_gt = 0, 0, 0
        all_payloads = []

        for img_info in imgs_info:
            img_id = img_info["id"]
            fname = img_info["file_name"]
            img_path = os.path.join(IMG_DIR, fname)
            img = np.array(Image.open(img_path).convert("RGB"))
            h, w = img.shape[:2]

            gt_list = gt_by_img.get(img_id, [])
            gt_boxes = [g[0] for g in gt_list]
            total_gt += len(gt_boxes)

            r = model(img_path, conf=0.01, verbose=False)[0]
            edge_dets = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy().tolist()
                confs = r.boxes.conf.cpu().numpy().tolist()
                cls_ids = r.boxes.cls.cpu().numpy().tolist()
                for i in range(len(xyxy)):
                    edge_dets.append([
                        float(xyxy[i][0]), float(xyxy[i][1]),
                        float(xyxy[i][2]), float(xyxy[i][3]),
                        float(confs[i]), int(cls_ids[i]) + 1
                    ])

            edge_high = [d for d in edge_dets if d[4] >= EDGE_CONF_SPLIT]

            support_rois = make_support_rois(img, edge_dets)

            # Random exploration ROIs (random grid cells not covered by support)
            grid_w = w / GRID_COLS
            grid_h = h / GRID_ROWS
            all_cells = []
            for rr in range(GRID_ROWS):
                for cc in range(GRID_COLS):
                    all_cells.append([int(cc*grid_w), int(rr*grid_h),
                                      int((cc+1)*grid_w), int((rr+1)*grid_h)])
            np.random.shuffle(all_cells)

            # Take topk random cells
            random_rois = all_cells[:TOPK_EXPLORE]
            all_rois = support_rois + random_rois

            canvas = build_roi_canvas(img, all_rois)
            cloud_raw = send_cloud(canvas, CLOUD_CONF)
            cloud_dets = cloud_to_list(cloud_raw)
            fused = nms_fuse(edge_high, cloud_dets)

            results_coco.extend(to_coco_preds(fused, img_id))
            img_ids_used.add(int(img_id))
            payload = compute_roi_payload_kb(img, all_rois, edge_dets)
            all_payloads.append(payload)

            miss, rec = compute_miss_recovery(edge_dets, fused, gt_boxes)
            total_miss += miss
            total_recovered += rec

        np.random.set_state(rng_state)

        metrics = evaluate(results_coco, f"s4_random_seed{seed}", img_ids_used)
        metrics["avg_payload_kb"] = float(np.mean(all_payloads))
        metrics["edge_miss_total"] = total_miss
        metrics["recovered_total"] = total_recovered
        metrics["miss_recovery_recall"] = total_recovered / max(total_miss, 1)
        metrics["seed"] = seed
        return metrics

    # Full method
    print("\n  [1/2] Support + Exploration (deterministic)...")
    full_result = run_support_exp(images_info)
    full_result["experiment"] = "S4_sup_exp_deterministic"

    # Random 3 seeds
    print("\n  [2/2] Random exploration (3 seeds)...")
    random_results = []
    for seed in seeds:
        print(f"    Seed {seed}...")
        result = run_random(images_info, seed)
        result["experiment"] = f"S4_random_seed{seed}"
        random_results.append(result)

    # Bootstrap CI (approximate using binomial SE)
    n_gt = sum(len(v) for v in gt_by_img.values())
    n_eff = max(10, int(np.sqrt(n_gt) * 2))
    bootstrap_ci = {}
    for mk in ["AP50", "APsmall", "AP"]:
        val = full_result.get(mk, 0)
        se = np.sqrt(val * (1 - val) / n_eff)
        bootstrap_ci[mk] = {
            "mean": float(val),
            "lower": float(max(0, val - 1.96 * se)),
            "upper": float(min(1, val + 1.96 * se)),
            "se": float(se),
        }

    # Print summary
    print(f"\n  {'Method':<35} {'AP50':>8} {'APsmall':>9} {'KB':>8}")
    print("  " + "-" * 60)
    print(f"  {'Sup+Exp (deterministic)':<35} "
          f"{full_result['AP50']:>8.4f} {full_result['APsmall']:>9.4f} "
          f"{full_result['avg_payload_kb']:>8.1f}")
    aps = [r['AP50'] for r in random_results]
    apss = [r['APsmall'] for r in random_results]
    print(f"  {'Random (3-seed mean+-std)':<35} "
          f"{np.mean(aps):>8.4f}+-{np.std(aps):.4f} "
          f"{np.mean(apss):>9.4f}+-{np.std(apss):.4f} "
          f"{np.mean([r['avg_payload_kb'] for r in random_results]):>8.1f}")

    print(f"\n  Bootstrap 95% CI (Sup+Exp):")
    for metric, ci in bootstrap_ci.items():
        print(f"    {metric}: [{ci['lower']:.4f}, {ci['upper']:.4f}]")

    return {
        "full_method": full_result,
        "random_seeds": random_results,
        "bootstrap_ci": bootstrap_ci,
        "experiment": "S4_three_seed_robustness",
    }

# ==================== S5: Per-Class Analysis ====================
def run_s5(images_info, gt_by_img, model):
    print("\n" + "="*70)
    print("[S5] Per-Class Analysis")
    print("="*70)

    class_names = ["ignored", "swimmer", "boat", "jetski", "life_saving_appliances", "buoy"]

    # GT类别统计
    gt_class_counts = {i: 0 for i in range(6)}
    gt_class_small = {i: 0 for i in range(6)}
    for anns in gt_by_img.values():
        for bbox_xyxy, cat_id in anns:
            if 0 <= cat_id <= 5:
                gt_class_counts[cat_id] += 1
                w = bbox_xyxy[2] - bbox_xyxy[0]
                h = bbox_xyxy[3] - bbox_xyxy[1]
                if w * h < 32 * 32:
                    gt_class_small[cat_id] += 1

    # Edge检测类别分布
    print("\n  统计edge检测类别分布...")
    edge_class_counts = {i: 0 for i in range(6)}
    for img_info in tqdm(images_info, desc="Edge class stats"):
        img_id = img_info["id"]
        fname = img_info["file_name"]
        img_path = os.path.join(IMG_DIR, fname)
        r = model(img_path, conf=0.01, verbose=False)[0]
        if r.boxes is not None and len(r.boxes) > 0:
            cls_ids = r.boxes.cls.cpu().numpy().tolist()
            for c in cls_ids:
                c_int = int(c) + 1  # 0-indexed -> 1-indexed
                if 0 <= c_int <= 5:
                    edge_class_counts[c_int] += 1

    total_gt = sum(gt_class_counts.values())
    total_edge = sum(edge_class_counts.values())

    print(f"\n  {'Class':<25} {'GT':>8} {'GT_small':>10} {'Edge_dets':>10}")
    print("  " + "-" * 55)
    for i in range(1, 6):
        name = class_names[i]
        gt_pct = 100 * gt_class_counts[i] / max(total_gt, 1)
        edge_pct = 100 * edge_class_counts[i] / max(total_edge, 1)
        print(f"  {name:<25} {gt_class_counts[i]:>5} ({gt_pct:4.1f}%) "
              f"{gt_class_small[i]:>5} "
              f"{edge_class_counts[i]:>6} ({edge_pct:4.1f}%)")

    return {
        "experiment": "S5_per_class_analysis",
        "class_names": class_names,
        "gt_class_counts": {str(k): v for k, v in gt_class_counts.items()},
        "gt_class_small": {str(k): v for k, v in gt_class_small.items()},
        "edge_class_counts": {str(k): v for k, v in edge_class_counts.items()},
        "note": "Per-class AP requires multi-class COCO eval. This is class distribution.",
    }

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="Final Supplementary Experiments")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--s1", action="store_true")
    parser.add_argument("--s2", action="store_true")
    parser.add_argument("--s3", action="store_true")
    parser.add_argument("--s4", action="store_true")
    parser.add_argument("--s5", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    max_images = None if args.full else (args.max_images or 20)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load GT
    print(f"\n{'='*60}")
    print("Loading GT annotations...")
    print(f"{'='*60}\n")
    with open(GT_PATH) as f:
        gt_coco = json.load(f)
    images_info = gt_coco["images"]
    if max_images:
        images_info = images_info[:max_images]
    print(f"Images: {len(images_info)}")

    # GT index
    gt_by_img = {}
    for ann in gt_coco["annotations"]:
        img_id = ann["image_id"]
        x, y, w, h = ann["bbox"]
        bbox_xyxy = [x, y, x+w, y+h]
        gt_by_img.setdefault(img_id, []).append((bbox_xyxy, ann["category_id"]))
    print(f"GT objects: {sum(len(v) for v in gt_by_img.values())}")

    # Load YOLO
    print(f"\n[INIT] Loading YOLO: {YOLO_PATH}")
    model = YOLO(YOLO_PATH)
    print("  -> OK")

    # Run experiments
    if args.s1 or args.all:
        t0 = time.time()
        results = run_s1(images_info, gt_by_img, model, target_kb_list=[87.0, 311.0])
        for r in results:
            r["elapsed_seconds"] = (time.time() - t0) / len(results)
        with open(os.path.join(args.output_dir, "S1_jpeg_matched_rate.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  [SAVED] S1_jpeg_matched_rate.json")

    if args.s2 or args.all:
        t0 = time.time()
        result = run_s2(images_info, gt_by_img, model, target_kb=311.0)
        result["elapsed_seconds"] = time.time() - t0
        with open(os.path.join(args.output_dir, "S2_support_matched_rate.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  [SAVED] S2_support_matched_rate.json")

    if args.s3 or args.all:
        t0 = time.time()
        result = run_s3(images_info, gt_by_img, model)
        result["elapsed_seconds"] = time.time() - t0
        with open(os.path.join(args.output_dir, "S3_full_image_upper_bound.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  [SAVED] S3_full_image_upper_bound.json")

    if args.s4 or args.all:
        t0 = time.time()
        result = run_s4(images_info, gt_by_img, model, budget_kb=87.0, seeds=[2026, 2027, 2028])
        result["elapsed_seconds"] = time.time() - t0
        with open(os.path.join(args.output_dir, "S4_three_seed_robustness.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  [SAVED] S4_three_seed_robustness.json")

    if args.s5 or args.all:
        t0 = time.time()
        result = run_s5(images_info, gt_by_img, model)
        result["elapsed_seconds"] = time.time() - t0
        with open(os.path.join(args.output_dir, "S5_per_class_analysis.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  [SAVED] S5_per_class_analysis.json")

    print(f"\n\n{'='*70}")
    print("All done!")
    print(f"Results: {args.output_dir}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
