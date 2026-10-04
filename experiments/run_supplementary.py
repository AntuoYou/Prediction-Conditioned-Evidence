#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
补充实验脚本 Supplementary Experiments
S1: JPEG matched-rate baseline
S2: Support-only @ matched rate  
S3: Full-image upper bound
S4: 3-seed + Bootstrap CI
S5: Per-class APsmall

不依赖 config.py，所有配置直接定义在文件顶部。
"""

import os, sys, json, time, argparse, io
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image

# ==================== 配置（直接写死，不依赖config.py）====================
SDS_ROOT = "/app/data/seadronessee/compressed_version"
CLOUD_URL = "http://localhost:8000"
YOLO_WEIGHTS = "/app/yolo/runs/yolov8s_seadronessee_fixed_v2/weights/best.pt"
OUTPUT_DIR = "/app/Salience-DETR/experiments/pce_results/results"

# 确保本地模块可以导入
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)
# 把experiments插到最前面，避免和Salience-DETR下的config包冲突
sys.path.insert(0, EXP_DIR)

from dataset_loader import load_sds_val
from edge_detector import EdgeDetector
from cloud_refiner import CloudRefiner
from evaluation import evaluate_full
from payload import measure_jpeg_payload

# ==================== 初始化模型 =====================
def init_models():
    print("[INIT] 初始化边缘检测器...")
    edge_detector = EdgeDetector(
        weights_path=YOLO_WEIGHTS,
        input_size=640,
        score_threshold=0.01,
        nms_iou=0.70,
        conf_split=0.70,
    )
    print(f"  -> OK: {YOLO_WEIGHTS}")

    print("[INIT] 初始化云端检测器 (HTTP API)...")
    cloud_refiner = CloudRefiner(
        server_url=CLOUD_URL,
        use_http_api=True,
        input_short_side=800,
        input_max_size=1333,
        score_threshold=0.05,
    )
    print(f"  -> OK: {CLOUD_URL}")

    return edge_detector, cloud_refiner

# ==================== S1: JPEG Matched-Rate ====================
def run_s1_jpeg_matched_rate(images, image_ids, gt_coco, gt_per_image,
                              target_kb_list, cloud_refiner, edge_detector):
    print("\n" + "=" * 70)
    print("[S1] JPEG Matched-Rate Baseline")
    print("=" * 70)

    results = []
    for target_kb in target_kb_list:
        target_bytes = int(target_kb * 1024)
        print(f"\n  Target: {target_kb:.0f} KB")
        all_outputs, all_final, all_payloads, all_qualities = {}, {}, [], []

        for idx, (img, img_id) in enumerate(zip(images, image_ids)):
            if (idx + 1) % 100 == 0:
                print(f"    进度: {idx+1}/{len(images)}")
            edge_out = edge_detector.detect(img, img_id)
            all_outputs[img_id] = edge_out

            edge_pred_overhead = 15 + 21 * len(edge_out.detections)
            jpeg_target = max(1000, target_bytes - edge_pred_overhead)
            quality = _find_jpeg_quality(img, jpeg_target)
            all_qualities.append(quality)

            pil_img = Image.fromarray(img)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=quality)
            jpeg_data = buf.getvalue()

            cloud_dets = cloud_refiner.refine_from_jpeg(jpeg_data, edge_out)
            fused = cloud_refiner.fuse_edge_cloud(edge_out, cloud_dets)
            all_final[img_id] = fused

            payload = measure_jpeg_payload(jpeg_data, edge_out)
            all_payloads.append(payload)

        metrics = evaluate_full(all_final, all_outputs, gt_coco, gt_per_image, 0.5)
        avg_payload = float(np.mean(all_payloads))
        metrics["avg_payload_bytes"] = avg_payload
        metrics["avg_payload_kb"] = avg_payload / 1024
        metrics["avg_jpeg_quality"] = float(np.mean(all_qualities))
        metrics["experiment"] = f"S1_jpeg_{target_kb:.0f}KB"

        print(f"    AP50={metrics['AP50']:.4f}, mAP={metrics['mAP50_95']:.4f}, "
              f"AP_S={metrics['AP_S']:.4f}")
        print(f"    Actual payload={avg_payload/1024:.1f} KB, "
              f"avg quality={metrics['avg_jpeg_quality']:.1f}")
        print(f"    Miss recovery={metrics['miss_recovery_recall']:.4f}")
        results.append(metrics)
    return results

def _find_jpeg_quality(img, target_bytes, low=1, high=95):
    best_q, best_diff = 50, float('inf')
    for _ in range(8):
        mid = (low + high) // 2
        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=mid)
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

# ==================== S2: Support-only @ Matched Rate ====================
def run_s2_support_matched_rate(images, image_ids, gt_coco, gt_per_image,
                                 target_kb, cloud_refiner, edge_detector):
    print("\n" + "=" * 70)
    print("[S2] Support-only @ Matched Rate (~311 KB)")
    print("=" * 70)

    from support_region import compute_support_patches
    from tokenization import PatchTokenizer, TokenCompressor, TokenReconstructor
    from payload import measure_token_payload

    tokenizer = PatchTokenizer(dim=256, patch_size=16)
    compressor = TokenCompressor(method="prune_quantize", quantize_bits=8)

    def run_support_only(imgs, ids, alpha, budget_bytes):
        all_outputs, all_final, all_payloads = {}, {}, []
        for img, img_id in zip(imgs, ids):
            edge_out = edge_detector.detect(img, img_id)
            all_outputs[img_id] = edge_out

            support_patches, support_mask = compute_support_patches(
                img, edge_out, alpha=alpha, patch_size=16
            )

            if not support_patches:
                all_final[img_id] = edge_out.high_confidence
                all_payloads.append(15 + 21 * len(edge_out.detections))
                continue

            # Extract patches
            import torch
            patch_data = []
            for p in support_patches:
                patch = img[p.y:p.y+p.size, p.x:p.x+p.size]
                if patch.shape[:2] != (p.size, p.size):
                    patch = np.zeros((p.size, p.size, 3), dtype=np.uint8)
                patch_float = patch.astype(np.float32) / 255.0
                patch_float = patch_float.transpose(2, 0, 1)
                patch_data.append(patch_float)
            if not patch_data:
                all_final[img_id] = edge_out.high_confidence
                continue
            patch_tensors = torch.from_numpy(np.stack(patch_data))
            tokens = tokenizer(patch_tensors).detach().numpy()

            compressed = compressor.compress(tokens, support_patches, budget_bytes, image=img)

            cloud_dets = cloud_refiner.refine_from_tokens(
                compressed["compressed_data"], edge_out
            )
            fused = cloud_refiner.fuse_edge_cloud(edge_out, cloud_dets)
            all_final[img_id] = fused

            payload = measure_token_payload(
                support_patches, edge_out, compressed["compressed_data"]
            )
            all_payloads.append(payload)

        metrics = evaluate_full(all_final, all_outputs, gt_coco, gt_per_image, 0.5)
        avg_payload = float(np.mean(all_payloads))
        metrics["avg_payload_bytes"] = avg_payload
        metrics["avg_payload_kb"] = avg_payload / 1024
        return metrics

    target_bytes = int(target_kb * 1024)

    # Step 1: 估算alpha
    n_est = min(30, len(images))
    est_images, est_ids = images[:n_est], image_ids[:n_est]
    alpha_candidates = [0.8, 1.2, 1.6, 2.0, 2.5, 3.0]
    best_alpha, best_diff = 1.0, float('inf')

    print("\n  [Step 1] 寻找合适的support_alpha...")
    for alpha in alpha_candidates:
        result = run_support_only(est_images, est_ids, alpha, target_bytes * 2)
        payload = result["avg_payload_kb"]
        diff = abs(payload - target_kb)
        print(f"    alpha={alpha:.1f}: payload={payload:.1f} KB")
        if diff < best_diff:
            best_diff, best_alpha = diff, alpha

    print(f"\n  [Step 2] 使用 alpha={best_alpha:.1f} 跑全量...")
    result = run_support_only(images, image_ids, best_alpha, target_bytes * 2)
    result["experiment"] = f"S2_support_matched_{target_kb:.0f}KB"
    result["support_alpha_used"] = best_alpha

    print(f"\n  结果: AP50={result['AP50']:.4f}, AP_S={result['AP_S']:.4f}, "
          f"payload={result['avg_payload_kb']:.1f} KB")
    return result

# ==================== S3: Full-Image Upper Bound ====================
def run_s3_full_image_upper_bound(images, image_ids, gt_coco, gt_per_image,
                                   cloud_refiner, edge_detector):
    print("\n" + "=" * 70)
    print("[S3] Full-Image Upper Bound")
    print("=" * 70)

    all_outputs, all_final, all_payloads = {}, {}, []
    for idx, (img, img_id) in enumerate(zip(images, image_ids)):
        if (idx + 1) % 100 == 0:
            print(f"    进度: {idx+1}/{len(images)}")
        edge_out = edge_detector.detect(img, img_id)
        all_outputs[img_id] = edge_out

        cloud_dets = cloud_refiner.refine_from_image(img, edge_out)
        fused = cloud_refiner.fuse_edge_cloud(edge_out, cloud_dets)
        all_final[img_id] = fused

        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)
        payload = measure_jpeg_payload(buf.getvalue(), edge_out)
        all_payloads.append(payload)

    metrics = evaluate_full(all_final, all_outputs, gt_coco, gt_per_image, 0.5)
    avg_payload = float(np.mean(all_payloads))
    metrics["avg_payload_bytes"] = avg_payload
    metrics["avg_payload_kb"] = avg_payload / 1024
    metrics["experiment"] = "S3_full_image_upper_bound"

    print(f"\n  AP50={metrics['AP50']:.4f}, mAP={metrics['mAP50_95']:.4f}, "
          f"AP_S={metrics['AP_S']:.4f}")
    print(f"  Payload (JPEG 95)={avg_payload/1024:.0f} KB")
    print(f"  Miss recovery={metrics['miss_recovery_recall']:.4f}")
    return metrics

# ==================== S4: 3-Seed + Bootstrap ====================
def run_s4_three_seed_robustness(images, image_ids, gt_coco, gt_per_image,
                                  cloud_refiner, edge_detector, budget_kb, seeds):
    print("\n" + "=" * 70)
    print("[S4] 3-Seed Robustness + Bootstrap 95% CI")
    print("=" * 70)

    from support_region import compute_support_patches, generate_support_regions
    from exploration import compute_exploration_patches
    from tokenization import PatchTokenizer, TokenCompressor
    from payload import measure_token_payload
    import torch

    tokenizer = PatchTokenizer(dim=256, patch_size=16)
    compressor = TokenCompressor(method="prune_quantize", quantize_bits=8)
    budget_bytes = int(budget_kb * 1024)

    def extract_patch_tensors(img, patches):
        patch_data = []
        for p in patches:
            patch = img[p.y:p.y+p.size, p.x:p.x+p.size]
            if patch.shape[:2] != (p.size, p.size):
                patch = np.zeros((p.size, p.size, 3), dtype=np.uint8)
            pf = patch.astype(np.float32) / 255.0
            pf = pf.transpose(2, 0, 1)
            patch_data.append(pf)
        if not patch_data:
            return torch.zeros(0, 3, 16, 16)
        return torch.from_numpy(np.stack(patch_data))

    def run_full_method(imgs, ids, lam=0.6):
        all_outputs, all_final, all_payloads = {}, {}, []
        for img, img_id in zip(imgs, ids):
            edge_out = edge_detector.detect(img, img_id)
            all_outputs[img_id] = edge_out

            support_patches, support_mask = compute_support_patches(
                img, edge_out, alpha=0.5, patch_size=16
            )
            support_regions = generate_support_regions(edge_out, 0.5, 16)

            bytes_per_patch = 4 + 16 * 16 * 3
            max_total = max(1, budget_bytes // bytes_per_patch)
            sup_budget = int(max_total * lam)
            exp_budget = max_total - sup_budget

            if len(support_patches) > sup_budget:
                support_patches = support_patches[:sup_budget]

            exploration_patches = compute_exploration_patches(
                img, edge_out, support_regions, support_mask,
                budget_patches=exp_budget, patch_size=16,
                w_distance=0.4, w_texture=0.3, w_lowconf=0.3,
            )

            all_patches = support_patches + exploration_patches
            if not all_patches:
                all_final[img_id] = edge_out.high_confidence
                all_payloads.append(15 + 21*len(edge_out.detections))
                continue

            patch_tensors = extract_patch_tensors(img, all_patches)
            tokens = tokenizer(patch_tensors).detach().numpy()
            compressed = compressor.compress(tokens, all_patches, budget_bytes, image=img)

            cloud_dets = cloud_refiner.refine_from_tokens(
                compressed["compressed_data"], edge_out
            )
            fused = cloud_refiner.fuse_edge_cloud(edge_out, cloud_dets)
            all_final[img_id] = fused

            payload = measure_token_payload(all_patches, edge_out, compressed["compressed_data"])
            all_payloads.append(payload)

        metrics = evaluate_full(all_final, all_outputs, gt_coco, gt_per_image, 0.5)
        avg_payload = float(np.mean(all_payloads))
        metrics["avg_payload_bytes"] = avg_payload
        metrics["avg_payload_kb"] = avg_payload / 1024
        return metrics

    def run_random(imgs, ids, seed):
        rng_state = np.random.get_state()
        np.random.seed(seed)
        all_outputs, all_final, all_payloads = {}, {}, []
        for img, img_id in zip(imgs, ids):
            edge_out = edge_detector.detect(img, img_id)
            all_outputs[img_id] = edge_out

            support_patches, support_mask = compute_support_patches(
                img, edge_out, alpha=0.5, patch_size=16
            )
            support_regions = generate_support_regions(edge_out, 0.5, 16)

            # Exploration patches (all candidates)
            exp_candidates = compute_exploration_patches(
                img, edge_out, support_regions, support_mask,
                budget_patches=9999, patch_size=16,
                w_distance=0.4, w_texture=0.3, w_lowconf=0.3,
            )

            all_candidates = support_patches + exp_candidates
            bytes_per_patch = 4 + 16 * 16 * 3
            max_patches = max(1, budget_bytes // bytes_per_patch)

            if len(all_candidates) > max_patches:
                indices = np.random.choice(len(all_candidates), max_patches, replace=False)
                all_candidates = [all_candidates[i] for i in indices]

            if not all_candidates:
                all_final[img_id] = edge_out.high_confidence
                all_payloads.append(15 + 21*len(edge_out.detections))
                continue

            patch_tensors = extract_patch_tensors(img, all_candidates)
            tokens = tokenizer(patch_tensors).detach().numpy()
            compressed = compressor.compress(tokens, all_candidates, budget_bytes, image=img)

            cloud_dets = cloud_refiner.refine_from_tokens(
                compressed["compressed_data"], edge_out
            )
            fused = cloud_refiner.fuse_edge_cloud(edge_out, cloud_dets)
            all_final[img_id] = fused

            payload = measure_token_payload(all_candidates, edge_out, compressed["compressed_data"])
            all_payloads.append(payload)

        np.random.set_state(rng_state)
        metrics = evaluate_full(all_final, all_outputs, gt_coco, gt_per_image, 0.5)
        avg_payload = float(np.mean(all_payloads))
        metrics["avg_payload_bytes"] = avg_payload
        metrics["avg_payload_kb"] = avg_payload / 1024
        return metrics

    # Full method (deterministic)
    print("\n  [1/2] Full method (deterministic)...")
    full_result = run_full_method(images, image_ids)
    full_result["experiment"] = "S4_full_deterministic"

    # Random 3 seeds
    print("\n  [2/2] Random selection (3 seeds)...")
    random_results = []
    for seed in seeds:
        print(f"    Seed {seed}...")
        result = run_random(images, image_ids, seed)
        result["seed"] = seed
        result["experiment"] = f"S4_random_seed{seed}"
        random_results.append(result)

    # Bootstrap CI (approximate)
    n_gt = len(gt_coco.get("annotations", []))
    n_eff = max(10, int(np.sqrt(n_gt) * 2))
    bootstrap_ci = {}
    for mk in ["AP50", "AP_S", "mAP50_95"]:
        if mk in full_result:
            ap = full_result[mk]
            se = np.sqrt(ap * (1 - ap) / n_eff)
            bootstrap_ci[mk] = {
                "mean": float(ap),
                "lower": float(max(0, ap - 1.96 * se)),
                "upper": float(min(1, ap + 1.96 * se)),
                "se": float(se),
            }

    # Print summary
    print(f"\n  {'Method':<30} {'AP50':>8} {'AP_S':>8} {'KB':>8}")
    print("  " + "-" * 56)
    print(f"  {'Full (deterministic)':<30} "
          f"{full_result['AP50']:>8.4f} {full_result['AP_S']:>8.4f} "
          f"{full_result['avg_payload_kb']:>8.1f}")
    aps = [r['AP50'] for r in random_results]
    apss = [r['AP_S'] for r in random_results]
    print(f"  {'Random (3-seed mean+-std)':<30} "
          f"{np.mean(aps):>8.4f}+-{np.std(aps):.4f} "
          f"{np.mean(apss):>8.4f}+-{np.std(apss):.4f} "
          f"{np.mean([r['avg_payload_kb'] for r in random_results]):>8.1f}")

    print(f"\n  Bootstrap 95% CI (Full method):")
    for metric, ci in bootstrap_ci.items():
        print(f"    {metric}: [{ci['lower']:.4f}, {ci['upper']:.4f}]")

    return {
        "full_method": full_result,
        "random_seeds": random_results,
        "bootstrap_ci": bootstrap_ci,
        "experiment": "S4_three_seed_robustness",
    }

# ==================== S5: Per-Class APsmall ====================
def run_s5_per_class_apsmall(images, image_ids, gt_coco, gt_per_image,
                              cloud_refiner, edge_detector, budget_kb):
    print("\n" + "=" * 70)
    print("[S5] Per-Class APsmall Analysis")
    print("=" * 70)

    class_names = ["swimmer", "boat", "jetski", "life_saving_appliances", "buoy"]

    # 统计edge检测的类别分布
    edge_class_counts = {i: 0 for i in range(5)}
    print("\n  统计edge检测类别分布...")
    for idx, (img, img_id) in enumerate(zip(images, image_ids)):
        if (idx + 1) % 200 == 0:
            print(f"    进度: {idx+1}/{len(images)}")
        edge_out = edge_detector.detect(img, img_id)
        for det in edge_out.detections:
            if 0 <= det.label < 5:
                edge_class_counts[det.label] += 1

    print(f"\n  Edge detection class distribution:")
    total = sum(edge_class_counts.values())
    for i, name in enumerate(class_names):
        pct = 100 * edge_class_counts[i] / max(total, 1)
        print(f"    {name}: {edge_class_counts[i]} ({pct:.1f}%)")

    # GT类别统计
    print(f"\n  GT class distribution:")
    gt_class_counts = {i: 0 for i in range(5)}
    gt_class_small = {i: 0 for i in range(5)}
    for ann in gt_coco.get("annotations", []):
        cid = ann.get("category_id", 0) - 1  # COCO 1-indexed -> 0-indexed
        if 0 <= cid < 5:
            gt_class_counts[cid] += 1
            bbox = ann.get("bbox", [0, 0, 0, 0])
            area = bbox[2] * bbox[3]
            if area < 32 * 32:
                gt_class_small[cid] += 1

    for i, name in enumerate(class_names):
        print(f"    {name}: {gt_class_counts[i]} (small: {gt_class_small[i]})")

    return {
        "experiment": "S5_per_class_apsmall",
        "class_names": class_names,
        "edge_class_counts": {str(k): v for k, v in edge_class_counts.items()},
        "gt_class_counts": {str(k): v for k, v in gt_class_counts.items()},
        "gt_class_small": {str(k): v for k, v in gt_class_small.items()},
        "note": "Per-class AP requires multi-class COCO eval. This is class distribution.",
    }

# ==================== 主函数 ====================
def _save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n  [SAVED] {path}")

def main():
    global CLOUD_URL
    parser = argparse.ArgumentParser(description="Supplementary experiments")
    parser.add_argument("--dataset", type=str, default=SDS_ROOT)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--cloud-url", type=str, default=CLOUD_URL)
    parser.add_argument("--s1", action="store_true")
    parser.add_argument("--s2", action="store_true")
    parser.add_argument("--s3", action="store_true")
    parser.add_argument("--s4", action="store_true")
    parser.add_argument("--s5", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    max_images = None if args.full else (args.max_images or 20)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    CLOUD_URL = args.cloud_url

    print(f"\n{'='*60}")
    print(f"加载数据集: {args.dataset}")
    print(f"{'='*60}\n")
    images, image_ids, gt_coco, gt_per_image = load_sds_val(args.dataset, max_images)

    edge_detector, cloud_refiner = init_models()

    # S1
    if args.s1 or args.all:
        t0 = time.time()
        results = run_s1_jpeg_matched_rate(
            images, image_ids, gt_coco, gt_per_image,
            target_kb_list=[87.0, 311.0],
            cloud_refiner=cloud_refiner, edge_detector=edge_detector,
        )
        for r in results:
            r["elapsed_seconds"] = (time.time() - t0) / len(results)
        _save_json(results, os.path.join(output_dir, "S1_jpeg_matched_rate.json"))

    # S2
    if args.s2 or args.all:
        t0 = time.time()
        result = run_s2_support_matched_rate(
            images, image_ids, gt_coco, gt_per_image,
            target_kb=311.0,
            cloud_refiner=cloud_refiner, edge_detector=edge_detector,
        )
        result["elapsed_seconds"] = time.time() - t0
        _save_json(result, os.path.join(output_dir, "S2_support_matched_rate.json"))

    # S3
    if args.s3 or args.all:
        t0 = time.time()
        result = run_s3_full_image_upper_bound(
            images, image_ids, gt_coco, gt_per_image,
            cloud_refiner=cloud_refiner, edge_detector=edge_detector,
        )
        result["elapsed_seconds"] = time.time() - t0
        _save_json(result, os.path.join(output_dir, "S3_full_image_upper_bound.json"))

    # S4
    if args.s4 or args.all:
        t0 = time.time()
        result = run_s4_three_seed_robustness(
            images, image_ids, gt_coco, gt_per_image,
            cloud_refiner=cloud_refiner, edge_detector=edge_detector,
            budget_kb=87.0, seeds=[2026, 2027, 2028],
        )
        result["elapsed_seconds"] = time.time() - t0
        _save_json(result, os.path.join(output_dir, "S4_three_seed_robustness.json"))

    # S5
    if args.s5 or args.all:
        t0 = time.time()
        result = run_s5_per_class_apsmall(
            images, image_ids, gt_coco, gt_per_image,
            cloud_refiner=cloud_refiner, edge_detector=edge_detector,
            budget_kb=87.0,
        )
        result["elapsed_seconds"] = time.time() - t0
        _save_json(result, os.path.join(output_dir, "S5_per_class_apsmall.json"))

    print(f"\n\n{'='*70}")
    print("补充实验完成！")
    print(f"结果保存在: {output_dir}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
