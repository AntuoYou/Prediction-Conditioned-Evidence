#!/usr/bin/env python3
"""
极其简单的mAP提取脚本
使用正则匹配area模式，提取最后的小数
"""

import sys
import re


def main():
    # 从文件或标准输入读取
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    else:
        content = sys.stdin.read()

    # 用于存储结果
    all_maps = []
    small_maps = []

    # 按行处理
    for i, line in enumerate(content.split('\n'), 1):
        line = line.strip()

        # 检查是否包含目标行
        if 'Average Precision' in line:
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
                        all_maps.append(value)
                        print(f"第{i}行: 找到整体mAP: {value}")
                    elif re.search(r'area\s*=\s*small', line, re.IGNORECASE):
                        small_maps.append(value)
                        print(f"第{i}行: 找到小目标mAP: {value}")

    # 输出统计信息
    print("\n" + "=" * 60)
    print("统计结果:")
    print("=" * 60)
    print(f"整体mAP数量: {len(all_maps)}")
    print(f"小目标mAP数量: {len(small_maps)}")

    # 如果两个列表长度相同，输出对应表格
    if len(all_maps) == len(small_maps):
        print("\n对应关系表格:")
        print(f"{'序号':<6} | {'整体mAP':<10} | {'小目标mAP':<10}")
        print("-" * 40)
        for i in range(len(all_maps)):
            print(f"{i + 1:<6} | {all_maps[i]:<10.4f} | {small_maps[i]:<10.4f}")
    else:
        print("\n⚠️ 注意: 整体mAP和小目标mAP数量不匹配")
        if all_maps:
            print("整体mAP列表:", [f"{x:.4f}" for x in all_maps])
        if small_maps:
            print("小目标mAP列表:", [f"{x:.4f}" for x in small_maps])

    print("=" * 60)


if __name__ == "__main__":
    main()