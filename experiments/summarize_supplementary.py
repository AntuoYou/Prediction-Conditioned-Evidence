#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总补充实验结果"""
import os, sys, json, argparse
import numpy as np

def load_results(results_dir):
    results = {}
    files = {
        "S1": "S1_jpeg_matched_rate.json",
        "S2": "S2_support_matched_rate.json",
        "S3": "S3_full_image_upper_bound.json",
        "S4": "S4_three_seed_robustness.json",
        "S5": "S5_per_class_apsmall.json",
    }
    for key, fname in files.items():
        fpath = os.path.join(results_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                results[key] = json.load(f)
            print(f"  [OK] {fname}")
        else:
            print(f"  [MISSING] {fname}")
    return results

def print_rate_perf_table(results):
    print("\n" + "=" * 90)
    print("表1: Rate-Performance 汇总")
    print("=" * 90)
    print(f"{'方法':<35} {'Payload(KB)':>12} {'AP50':>8} {'mAP':>8} {'APsmall':>9} {'MissR':>8}")
    print("-" * 90)

    if "S3" in results:
        r = results["S3"]
        print(f"{'Full image (upper bound)':<35} "
              f"{r['avg_payload_kb']:>12.1f} "
              f"{r['AP50']:>8.4f} {r['mAP50_95']:>8.4f} {r['AP_S']:>9.4f} "
              f"{r['miss_recovery_recall']:>8.4f}")

    if "S1" in results:
        for r in results["S1"]:
            label = f"JPEG @ {r['avg_payload_kb']:.0f} KB"
            print(f"{label:<35} "
                  f"{r['avg_payload_kb']:>12.1f} "
                  f"{r['AP50']:>8.4f} {r['mAP50_95']:>8.4f} {r['AP_S']:>9.4f} "
                  f"{r['miss_recovery_recall']:>8.4f}")

    if "S2" in results:
        r = results["S2"]
        print(f"{'Support-only @ matched rate':<35} "
              f"{r['avg_payload_kb']:>12.1f} "
              f"{r['AP50']:>8.4f} {r['mAP50_95']:>8.4f} {r['AP_S']:>9.4f} "
              f"{r['miss_recovery_recall']:>8.4f}")

    print("-" * 90)
    print("注: Support+Exp 和 Evidence-aware 的结果请从主实验E3/E7获取")

def print_exploration_value(results):
    print("\n" + "=" * 80)
    print("表2: Exploration 价值验证 (Matched-Rate)")
    print("=" * 80)
    print(f"{'方法':<35} {'Payload(KB)':>12} {'AP50':>8} {'APsmall':>9}")
    print("-" * 80)
    if "S2" in results:
        r = results["S2"]
        print(f"{'Support-only (enlarged)':<35} "
              f"{r['avg_payload_kb']:>12.1f} "
              f"{r['AP50']:>8.4f} {r['AP_S']:>9.4f}")
    print("-" * 80)
    print("对比 Support+Exp @ 311KB:")
    print("  如果 Support+Exp > Support-only(enlarged)")
    print("  -> Exploration确实选对了区域，而非仅靠多传数据")

def print_miss_recovery(results):
    print("\n" + "=" * 80)
    print("表3: Edge-Miss Recovery Rate 汇总")
    print("=" * 80)
    print(f"{'方法':<35} {'Edge-missed':>10} {'Recovered':>10} {'Recovery':>10}")
    print("-" * 80)
    if "S3" in results:
        r = results["S3"]
        print(f"{'Full image':<35} "
              f"{r.get('edge_miss_total','N/A'):>10} "
              f"{r.get('recovered_total','N/A'):>10} "
              f"{r['miss_recovery_recall']:>10.4f}")
    if "S1" in results:
        for r in results["S1"]:
            label = f"JPEG @ {r['avg_payload_kb']:.0f} KB"
            print(f"{label:<35} "
                  f"{r.get('edge_miss_total','N/A'):>10} "
                  f"{r.get('recovered_total','N/A'):>10} "
                  f"{r['miss_recovery_recall']:>10.4f}")
    if "S2" in results:
        r = results["S2"]
        print(f"{'Support-only (matched)':<35} "
              f"{r.get('edge_miss_total','N/A'):>10} "
              f"{r.get('recovered_total','N/A'):>10} "
              f"{r['miss_recovery_recall']:>10.4f}")

def print_bootstrap(results):
    if "S4" not in results:
        return
    print("\n" + "=" * 80)
    print("表4: Bootstrap 95% CI + 3-Seed Random")
    print("=" * 80)
    s4 = results["S4"]
    if "bootstrap_ci" in s4 and s4["bootstrap_ci"]:
        print(f"{'指标':<15} {'均值':>10} {'95% CI下界':>12} {'95% CI上界':>12}")
        print("-" * 60)
        for metric, ci in s4["bootstrap_ci"].items():
            print(f"{metric:<15} {ci['mean']:>10.4f} {ci['lower']:>12.4f} {ci['upper']:>12.4f}")

    if "random_seeds" in s4 and s4["random_seeds"]:
        print(f"\nRandom selection 3-seed:")
        for r in s4["random_seeds"]:
            print(f"  Seed {r.get('seed','?')}: "
                  f"AP50={r['AP50']:.4f}, AP_S={r['AP_S']:.4f}, KB={r['avg_payload_kb']:.1f}")
        aps = [r['AP50'] for r in s4["random_seeds"]]
        apss = [r['AP_S'] for r in s4["random_seeds"]]
        print(f"\n  Mean +- Std:")
        print(f"    AP50:    {np.mean(aps):.4f} +- {np.std(aps):.4f}")
        print(f"    APsmall: {np.mean(apss):.4f} +- {np.std(apss):.4f}")

def print_per_class(results):
    if "S5" not in results:
        return
    print("\n" + "=" * 80)
    print("表5: Per-Class 分析")
    print("=" * 80)
    s5 = results["S5"]
    print(f"  类别: {s5.get('class_names', [])}")
    print(f"  Full method AP50: {s5['full_method_result']['AP50']:.4f}")
    print(f"  Full method APsmall: {s5['full_method_result']['AP_S']:.4f}")
    if "edge_class_counts" in s5:
        print(f"\n  Edge检测类别分布:")
        for i, name in enumerate(s5["class_names"]):
            cnt = s5["edge_class_counts"].get(str(i), s5["edge_class_counts"].get(i, 0))
            print(f"    {name}: {cnt}")
    print("\n  注: 精确per-class AP需要多类别COCO评估")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str,
                        default="/app/Salience-DETR/experiments/pce_results/results")
    args = parser.parse_args()
    print(f"\n加载结果: {args.results_dir}")
    results = load_results(args.results_dir)
    if not results:
        print("\n没有找到结果文件。")
        return
    print_rate_perf_table(results)
    print_exploration_value(results)
    print_miss_recovery(results)
    print_bootstrap(results)
    print_per_class(results)
    print("\n" + "=" * 90)
    print("汇总完成！")
    print("=" * 90)

if __name__ == "__main__":
    main()
