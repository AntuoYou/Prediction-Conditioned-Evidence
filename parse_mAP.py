#!/usr/bin/env python3
"""
精确的mAP提取脚本
只匹配包含特定关键词的行，精确提取 IoU=0.50:0.95 的AP值
"""

import sys
import re
from collections import defaultdict


def main():
    # 从文件或标准输入读取
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    else:
        content = sys.stdin.read()

    # 用于存储结果 - 按评估轮次存储
    evaluation_data = []
    current_eval = {"epoch": None, "map_all": None, "map_small": None}

    # 先尝试匹配epoch
    epoch_pattern = re.compile(r'epoch\s*(\d+)', re.IGNORECASE)
    lines = content.split('\n')

    for i, line in enumerate(lines, 1):
        line = line.strip()

        # 检查是否包含epoch
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            current_eval["epoch"] = int(epoch_match.group(1))

        # 检查是否包含 "Accumulating evaluation results..." - 表示新的评估开始
        if "Accumulating evaluation results" in line:
            # 保存前一个评估结果（如果有数据）
            if current_eval["map_all"] is not None or current_eval["map_small"] is not None:
                evaluation_data.append(current_eval.copy())
                current_eval = {"epoch": current_eval.get("epoch"), "map_all": None, "map_small": None}

        # 检查是否是 IoU=0.50:0.95 的AP值
        if 'Average Precision' in line and 'IoU=0.50:0.95' in line:
            # 提取等号后面的数字
            parts = line.split('=')
            if len(parts) > 1:
                # 取最后一部分，并提取其中的数字
                last_part = parts[-1].strip()
                # 查找数字（支持整数和小数）
                match = re.search(r'([0-9]+\.[0-9]+|[0-9]+)', last_part)

                if match:
                    value = float(match.group(1))

                    # 使用正则匹配area类型，支持任意空格
                    if re.search(r'area\s*=\s*all', line, re.IGNORECASE):
                        current_eval["map_all"] = value
                    elif re.search(r'area\s*=\s*small', line, re.IGNORECASE):
                        current_eval["map_small"] = value

    # 添加最后一个评估结果
    if current_eval["map_all"] is not None or current_eval["map_small"] is not None:
        evaluation_data.append(current_eval.copy())

    # 输出统计信息
    print("\n" + "=" * 60)
    print("COCO评估结果汇总 (IoU=0.50:0.95)")
    print("=" * 60)

    # 如果没有找到epoch信息，使用评估序号
    if all(eval_data["epoch"] is None for eval_data in evaluation_data):
        for i, eval_data in enumerate(evaluation_data):
            eval_data["epoch"] = i + 1

    # 生成表格
    print(f"{'评估轮次':<10} | {'整体mAP':<10} | {'小目标mAP':<10}")
    print("-" * 40)

    for eval_data in evaluation_data:
        epoch = eval_data.get("epoch", "N/A")
        map_all = eval_data.get("map_all", "N/A")
        map_small = eval_data.get("map_small", "N/A")

        map_all_str = f"{map_all:.4f}" if isinstance(map_all, float) else "N/A"
        map_small_str = f"{map_small:.4f}" if isinstance(map_small, float) else "N/A"

        print(f"{epoch:<10} | {map_all_str:<10} | {map_small_str:<10}")

    print("=" * 60)

    # 计算统计信息
    all_maps = [d["map_all"] for d in evaluation_data if d["map_all"] is not None]
    small_maps = [d["map_small"] for d in evaluation_data if d["map_small"] is not None]

    print(f"\n📊 统计信息:")
    print(f"  总评估轮次: {len(evaluation_data)}")
    print(f"  有整体mAP的轮次: {len(all_maps)}")
    print(f"  有小目标mAP的轮次: {len(small_maps)}")

    if all_maps:
        best_all = max(all_maps)
        best_all_epoch = [d["epoch"] for d in evaluation_data if d.get("map_all") == best_all]
        print(f"  最佳整体mAP: {best_all:.4f} (评估轮次: {best_all_epoch})")

    if small_maps:
        best_small = max(small_maps)
        best_small_epoch = [d["epoch"] for d in evaluation_data if d.get("map_small") == best_small]
        print(f"  最佳小目标mAP: {best_small:.4f} (评估轮次: {best_small_epoch})")

    print("=" * 60)


if __name__ == "__main__":
    main()