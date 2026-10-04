#!/usr/bin/env python3
"""
补充实验 v2 (直接基于YOLO + requests + pycocotools，不依赖自制模块)

S1: JPEG matched-rate baseline (整图JPEG在相同通信量下对比)
S2: Support-only @ matched rate (扩大Support到311KB，验证Exploration价值)
S3: Full-image upper bound (全图上传，建立性能上界)

用法:
  python exp_supplement_v2.py --s3 --max-images 20    # 测试
  python exp_supplement_v2.py --s1 --s2 --s3 --full    # 全量
"""

import json, os, io, argparse, time
import numpy as np
import requests
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ==================== 配置 ====================
YOLO_MODEL_PATH = "/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt"
CLOUD_URL = "http://localhost:8000/detect"
GT_PATH = "/app/data/seadronessee/compressed_version/annotations/instances_val2017.json"
IMG_DIR = "/app/data/seadronessee/compressed_version/val2017/"
OUTPUT_DIR = "/app/Salience-DETR/experiments/pce_results/results"

EXPAND_RATIO = 0.1      # support扩展比
LOW_RES_SCALE = 8
GRID_ROWS = 6
GRID_COLS = 10
ALPHA_SCORE = 0.3       # exploration coverage权重
BETA_SCORE = 0.7        # exploration proximity权重
TOPK_EXPLORE = 8
PATCH_SIZE = 16
IOU_THRESHOLD = 0.5
CLOUD_CONF = 0.05

# ==================== 工具函数 ====================
def send_cloud(img_np, conf=CLOUD_CONF):
    """发送图像到云端DETR，返回检测框列表 [x1,y1,x2,y2,score,label]"""
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format='JPEG', quality=95)
    buf.seek(0)
    try:
        r = requests.post(CLOUD_URL,
            files={'file': ('img.jpg', buf, 'image/jpeg')},
            data={'conf_threshold': str(conf)}, timeout=60)
        if r.status_code == 200:
            raw = r.json().get('detections', [])
            result = []
            for d in raw:
                b = d['bbox']
                # bbox is already xyxy format: [x1, y1, x2, y2]
                result.append([float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(d['confidence']), int(d['class_id'])])
            return result
    except Exception as e:
        print(f"  [WARN] cloud request failed: {e}")
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

def nms_fuse(edge_high, cloud_dets, iou_thresh=0.5):
    """NMS融合边缘高置信度 + 云端检测"""
    all_dets = []
    for d in edge_high:
        all_dets.append([d[0], d[1], d[2], d[3], d[4], d[5]])
    for d in cloud_dets:
        all_dets.append([d[0], d[1], d[2], d[3], d[4], d[5]])

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

def compute_support_regions(edge_dets, img_h, img_w, expand_ratio):
    """生成support ROI区域（扩展后的检测框）"""
    regions = []
    for d in edge_dets:
        x1, y1, x2, y2 = d[:4]
        w, h = x2 - x1, y2 - y1
        dx = max(w * expand_ratio, 16)
        dy = max(h * expand_ratio, 16)
        rx1 = max(0, int(x1 - dx))
        ry1 = max(0, int(y1 - dy))
        rx2 = min(img_w, int(x2 + dx))
        ry2 = min(img_h, int(y2 + dy))
        regions.append([rx1, ry1, rx2, ry2])
    return regions

def generate_exploration_rois(supports, img_w, img_h, topk=TOPK_EXPLORE):
    """生成exploration ROI（基于coverage + proximity评分）"""
    grid_w = img_w / GRID_COLS
    grid_h = img_h / GRID_ROWS
    max_dist = (img_w**2 + img_h**2) ** 0.5
    candidates = []

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            gx1, gy1 = int(c * grid_w), int(r * grid_h)
            gx2, gy2 = int((c+1) * grid_w), int((r+1) * grid_h)

            # coverage: 被support覆盖的比例
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

            # proximity: 到最近support的距离
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

def build_evidence_canvas(img, support_rois, exploration_rois, mode='full'):
    """
    构建证据画布（灰色背景 + ROI裁剪区域）
    mode: 'full' = 所有像素都传, 'jpeg' = 返回JPEG字节
    """
    h, w = img.shape[:2]
    canvas = np.full((h, w, 3), 110, dtype=np.uint8)

    all_rois = support_rois + exploration_rois
    for roi in all_rois:
        x1, y1, x2, y2 = roi
        canvas[y1:y2, x1:x2] = img[y1:y2, x1:x2]

    return canvas

def evaluate_coco(all_preds, gt_coco, img_id_map):
    """COCO评估，返回AP50, mAP, APsmall等"""
    if not all_preds:
        return {k: 0.0 for k in ['AP50', 'mAP50_95', 'AP_S', 'AR_S']}

    # 转换COCO格式的预测
    coco_preds = []
    for img_key, dets in all_preds.items():
        coco_id = img_id_map.get(img_key, 0)
        for d in dets:
            coco_preds.append({
                'image_id': coco_id,
                'category_id': 1,
                'bbox': [float(d[0]), float(d[1]), float(d[2]-d[0]), float(d[3]-d[1])],
                'score': float(d[4]) if hasattr(d[4], 'item') else d[4],
            })

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as gt_f:
        json.dump(gt_coco, gt_f)
        gt_path = gt_f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as pred_f:
        json.dump(coco_preds, pred_f)
        pred_path = pred_f.name

    try:
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(pred_path)
        e = COCOeval(coco_gt, coco_dt, 'bbox')
        e.evaluate()
        e.accumulate()
        e.summarize()
        s = e.stats
        result = {
            'AP50': float(s[1]),
            'mAP50_95': float(s[0]),
            'AP75': float(s[2]),
            'AP_S': float(s[3]),
            'AP_M': float(s[4]),
            'AP_L': float(s[5]),
            'AR_S': float(s[8]),
        }
    finally:
        os.unlink(gt_path)
        os.unlink(pred_path)

    return result

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

def load_dataset(max_images=None):
    """加载数据集"""
    print(f"[LOAD] GT: {GT_PATH}")
    coco = COCO(GT_PATH)
    img_ids = sorted(coco.getImgIds())
    if max_images:
        img_ids = img_ids[:max_images]

    images = []
    image_keys = []
    gt_per_image = {}
    img_id_map = {}  # filename -> coco img_id

    for img_id in tqdm(img_ids, desc='Loading images'):
        info = coco.loadImgs(img_id)[0]
        fname = info['file_name']
        img_path = os.path.join(IMG_DIR, fname)
        if not os.path.exists(img_path):
            continue
        img = np.array(Image.open(img_path).convert('RGB'))
        h, w = img.shape[:2]

        images.append(img)
        key = os.path.splitext(fname)[0]
        image_keys.append(key)
        img_id_map[key] = img_id

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_boxes = []
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            b = ann['bbox']
            gt_boxes.append([b[0], b[1], b[0]+b[2], b[1]+b[3]])
        gt_per_image[key] = gt_boxes

    print(f"  -> {len(images)} images, {sum(len(v) for v in gt_per_image.values())} GT objects")
    return images, image_keys, coco.dataset, gt_per_image, img_id_map

# ==================== S1: JPEG Matched-Rate ====================
def run_s1(images, image_keys, gt_coco, gt_per_image, img_id_map,
           yolo, target_kb_list):
    print("\n" + "="*70)
    print("[S1] JPEG Matched-Rate Baseline")
    print("="*70)

    results = []
    for target_kb in target_kb_list:
        target_bytes = int(target_kb * 1024)
        print(f"\n  Target = {target_kb:.0f} KB")

        all_final = {}
        all_payloads = []
        all_qualities = []
        edge_dets_map = {}

        for img, key in tqdm(zip(images, image_keys), total=len(images),
                             desc=f"S1@{target_kb:.0f}KB"):
            # Edge detection
            res = yolo(img, verbose=False)[0]
            edge_dets = []
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                score = float(b.conf[0].cpu())
                label = int(b.cls[0].cpu())
                edge_dets.append([x1, y1, x2, y2, score, label])
            edge_dets_map[key] = edge_dets
            edge_high = [d for d in edge_dets if d[4] >= 0.7]

            # 计算目标JPEG大小（扣除协议开销）
            edge_pred_bytes = 15 + 21 * len(edge_dets)
            jpeg_target = max(1000, target_bytes - edge_pred_bytes)

            quality = find_jpeg_quality(img, jpeg_target)
            all_qualities.append(quality)

            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format='JPEG', quality=quality)
            jpeg_data = buf.getvalue()

            # Cloud detection from JPEG
            cloud_dets = send_cloud(np.array(Image.open(io.BytesIO(jpeg_data)).convert('RGB')))

            # Fuse
            fused = nms_fuse(edge_high, cloud_dets)
            all_final[key] = fused

            # Payload
            payload = edge_pred_bytes + len(jpeg_data)
            all_payloads.append(payload)

        # Evaluate
        metrics = evaluate_coco(all_final, gt_coco, img_id_map)

        # Miss recovery
        total_miss, total_recovered, total_gt = 0, 0, 0
        for key in image_keys:
            gt_boxes = gt_per_image.get(key, [])
            miss, rec = compute_miss_recovery(edge_dets_map[key], all_final[key], gt_boxes)
            total_miss += miss
            total_recovered += rec
            total_gt += len(gt_boxes)

        metrics['edge_miss_total'] = total_miss
        metrics['recovered_total'] = total_recovered
        metrics['miss_recovery_recall'] = total_recovered / max(total_miss, 1)
        metrics['total_gt'] = total_gt
        metrics['avg_payload_kb'] = float(np.mean(all_payloads)) / 1024
        metrics['avg_jpeg_quality'] = float(np.mean(all_qualities))
        metrics['experiment'] = f"S1_jpeg_{target_kb:.0f}KB"

        print(f"  AP50={metrics['AP50']:.4f}, mAP={metrics['mAP50_95']:.4f}, "
              f"AP_S={metrics['AP_S']:.4f}")
        print(f"  Actual payload={metrics['avg_payload_kb']:.1f} KB, "
              f"avg quality={metrics['avg_jpeg_quality']:.1f}")
        print(f"  Miss recovery={metrics['miss_recovery_recall']:.4f} "
              f"({total_recovered}/{total_miss})")

        results.append(metrics)

    return results

# ==================== S2: Support-only @ Matched Rate ====================
def run_s2(images, image_keys, gt_coco, gt_per_image, img_id_map,
           yolo, target_kb=311.0):
    print("\n" + "="*70)
    print("[S2] Support-only @ Matched Rate")
    print("="*70)

    target_bytes = int(target_kb * 1024)

    def run_with_alpha(imgs, keys, alpha):
        all_final = {}
        all_payloads = []
        edge_dets_map = {}

        for img, key in zip(imgs, keys):
            res = yolo(img, verbose=False)[0]
            edge_dets = []
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                score = float(b.conf[0].cpu())
                label = int(b.cls[0].cpu())
                edge_dets.append([x1, y1, x2, y2, score, label])
            edge_dets_map[key] = edge_dets
            edge_high = [d for d in edge_dets if d[4] >= 0.7]

            h, w = img.shape[:2]
            support_rois = compute_support_regions(edge_dets, h, w, alpha)

            # 构建support画布
            canvas = build_evidence_canvas(img, support_rois, [])

            # 计算payload
            # ROI JPEG: 只压缩ROI区域 + 坐标
            roi_pixels = 0
            for roi in support_rois:
                roi_pixels += (roi[2]-roi[0]) * (roi[3]-roi[1])

            payload = 15 + 21 * len(edge_dets) + 16 * len(support_rois) + roi_pixels * 0.5  # 估算

            # 直接传画布给云端
            cloud_dets = send_cloud(canvas)
            fused = nms_fuse(edge_high, cloud_dets)
            all_final[key] = fused
            all_payloads.append(payload)

        metrics = evaluate_coco(all_final, gt_coco, img_id_map)
        total_miss, total_recovered = 0, 0
        for key in keys:
            gt_boxes = gt_per_image.get(key, [])
            miss, rec = compute_miss_recovery(edge_dets_map[key], all_final[key], gt_boxes)
            total_miss += miss
            total_recovered += rec
        metrics['edge_miss_total'] = total_miss
        metrics['recovered_total'] = total_recovered
        metrics['miss_recovery_recall'] = total_recovered / max(total_miss, 1)
        metrics['avg_payload_kb'] = float(np.mean(all_payloads)) / 1024
        return metrics

    # Step 1: 估算alpha（前30张）
    n_est = min(30, len(images))
    est_imgs, est_keys = images[:n_est], image_keys[:n_est]

    alpha_candidates = [0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    best_alpha, best_diff = 0.1, float('inf')

    print("\n  [Step 1] 估算合适的expand_ratio...")
    for alpha in alpha_candidates:
        m = run_with_alpha(est_imgs, est_keys, alpha)
        payload = m['avg_payload_kb']
        diff = abs(payload - target_kb)
        print(f"    expand={alpha:.1f}: payload={payload:.1f} KB (diff={diff:.1f})")
        if diff < best_diff:
            best_diff, best_alpha = diff, alpha

    print(f"\n  [Step 2] 使用 expand_ratio={best_alpha:.1f} 跑全量...")
    result = run_with_alpha(images, image_keys, best_alpha)
    result['experiment'] = f"S2_support_matched_{target_kb:.0f}KB"
    result['expand_ratio_used'] = best_alpha

    print(f"\n  结果: AP50={result['AP50']:.4f}, AP_S={result['AP_S']:.4f}, "
          f"payload={result['avg_payload_kb']:.1f} KB")
    return result

# ==================== S3: Full-Image Upper Bound ====================
def run_s3(images, image_keys, gt_coco, gt_per_image, img_id_map, yolo):
    print("\n" + "="*70)
    print("[S3] Full-Image Upper Bound")
    print("="*70)

    all_final = {}
    all_payloads = []
    edge_dets_map = {}

    for img, key in tqdm(zip(images, image_keys), total=len(images), desc="S3 Full Image"):
        res = yolo(img, verbose=False)[0]
        edge_dets = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            score = float(b.conf[0].cpu())
            label = int(b.cls[0].cpu())
            edge_dets.append([x1, y1, x2, y2, score, label])
        edge_dets_map[key] = edge_dets
        edge_high = [d for d in edge_dets if d[4] >= 0.7]

        # Full image cloud detection
        cloud_dets = send_cloud(img)
        fused = nms_fuse(edge_high, cloud_dets)
        all_final[key] = fused

        # Payload: JPEG quality 95 + edge preds
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format='JPEG', quality=95)
        payload = 15 + 21 * len(edge_dets) + len(buf.getvalue())
        all_payloads.append(payload)

    # Build COCO prediction list
    results_coco = []
    img_ids_used = set()
    for key, dets in all_final.items():
        coco_id = img_id_map.get(key, 0)
        img_ids_used.add(int(coco_id))
        for d in dets:
            results_coco.append({
                'image_id': int(coco_id),
                'category_id': int(d[5]),
                'bbox': [float(d[0]), float(d[1]), float(d[2]-d[0]), float(d[3]-d[1])],
                'score': float(d[4]),
            })

    metrics = evaluate_coco(results_coco, 's3_full', img_ids_used)

    total_miss, total_recovered, total_gt = 0, 0, 0
    for key in image_keys:
        gt_boxes = gt_per_image.get(key, [])
        miss, rec = compute_miss_recovery(edge_dets_map[key], all_final[key], gt_boxes)
        total_miss += miss
        total_recovered += rec
        total_gt += len(gt_boxes)

    metrics['edge_miss_total'] = total_miss
    metrics['recovered_total'] = total_recovered
    metrics['miss_recovery_recall'] = total_recovered / max(total_miss, 1)
    metrics['total_gt'] = total_gt
    metrics['avg_payload_kb'] = float(np.mean(all_payloads)) / 1024
    metrics['experiment'] = 'S3_full_image_upper_bound'

    print(f"\n  AP50={metrics['AP50']:.4f}, mAP={metrics['mAP50_95']:.4f}, "
          f"AP_S={metrics['AP_S']:.4f}")
    print(f"  Payload (JPEG95)={metrics['avg_payload_kb']:.0f} KB")
    print(f"  Miss recovery={metrics['miss_recovery_recall']:.4f} "
          f"({total_recovered}/{total_miss})")

    return metrics

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="Supplementary experiments v2")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--s1", action="store_true")
    parser.add_argument("--s2", action="store_true")
    parser.add_argument("--s3", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    max_images = None if args.full else (args.max_images or 20)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    images, image_keys, gt_coco, gt_per_image, img_id_map = load_dataset(max_images)

    # Load YOLO
    print(f"\n[INIT] Loading YOLO: {YOLO_MODEL_PATH}")
    yolo = YOLO(YOLO_MODEL_PATH)
    print("  -> OK")

    # Run experiments
    if args.s1 or args.all:
        t0 = time.time()
        results = run_s1(images, image_keys, gt_coco, gt_per_image, img_id_map,
                         yolo, target_kb_list=[87.0, 311.0])
        for r in results:
            r['elapsed_seconds'] = (time.time() - t0) / len(results)
        with open(os.path.join(args.output_dir, 'S1_jpeg_matched_rate.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  [SAVED] S1_jpeg_matched_rate.json")

    if args.s2 or args.all:
        t0 = time.time()
        result = run_s2(images, image_keys, gt_coco, gt_per_image, img_id_map,
                        yolo, target_kb=311.0)
        result['elapsed_seconds'] = time.time() - t0
        with open(os.path.join(args.output_dir, 'S2_support_matched_rate.json'), 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n  [SAVED] S2_support_matched_rate.json")

    if args.s3 or args.all:
        t0 = time.time()
        result = run_s3(images, image_keys, gt_coco, gt_per_image, img_id_map, yolo)
        result['elapsed_seconds'] = time.time() - t0
        with open(os.path.join(args.output_dir, 'S3_full_image_upper_bound.json'), 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n  [SAVED] S3_full_image_upper_bound.json")

    print(f"\n{'='*70}")
    print("Done!")
    print(f"Results: {args.output_dir}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
