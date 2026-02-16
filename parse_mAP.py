#!/usr/bin/env python3
"""
解析SEADRONSEE数据集DETR模型训练日志中的mAP指标
专为华为NPU训练日志设计，只提取整体mAP和小目标mAP
"""

import re
import sys
from collections import defaultdict
from pathlib import Path


class SimpleMAPParser:
    def __init__(self):
        # 非常宽松的正则表达式，只匹配关键模式
        self.epoch_pattern = re.compile(r'epoch\s*(\d+)', re.IGNORECASE)

        # 整体mAP：只要包含"Average Precision"和"area= all"即可
        self.map_all_pattern = re.compile(
            r'Average\s*Precision.*?area=\s*all.*?=\s*([0-9]+\.[0-9]+|[0-9]+)',
            re.IGNORECASE | re.DOTALL
        )

        # 小目标mAP：只要包含"Average Precision"和"area= small"即可
        self.map_small_pattern = re.compile(
            r'Average\s*Precision.*?area=\s*small.*?=\s*([0-9]+\.[0-9]+|[0-9]+)',
            re.IGNORECASE | re.DOTALL
        )

        # 备用模式：直接匹配数字，如果上述模式失败
        self.fallback_pattern = re.compile(r'=\s*([0-9]+\.[0-9]+|[0-9]+)')

    def extract_value(self, pattern, line, fallback=False):
        """从行中提取数值"""
        match = pattern.search(line)
        if match:
            try:
                return float(match.group(1))
            except:
                return None

        # 如果主要模式失败且允许备用模式
        if fallback:
            # 提取最后一个等号后的数字
            parts = line.split('=')
            if len(parts) > 1:
                last_part = parts[-1].strip()
                num_match = re.search(r'([0-9]+\.[0-9]+|[0-9]+)', last_part)
                if num_match:
                    try:
                        return float(num_match.group(1))
                    except:
                        return None
        return None

    def parse_log(self, log_content):
        """解析日志内容"""
        results = []
        current_epoch = None
        current_data = {}

        lines = log_content.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # 尝试匹配epoch
            epoch_match = self.epoch_pattern.search(line)
            if epoch_match:
                # 保存之前的数据
                if current_epoch is not None and current_data:
                    results.append((current_epoch, current_data.copy()))

                current_epoch = int(epoch_match.group(1))
                current_data = {'epoch': current_epoch}
                continue

            # 如果还没有找到epoch，尝试检查前几行
            if current_epoch is None and i > 0:
                for j in range(max(0, i - 2), i):
                    epoch_match = self.epoch_pattern.search(lines[j].strip())
                    if epoch_match:
                        current_epoch = int(epoch_match.group(1))
                        current_data = {'epoch': current_epoch}
                        break

            # 提取整体mAP
            if 'area= all' in line or 'area=all' in line or 'area = all' in line:
                map_all = self.extract_value(self.map_all_pattern, line, fallback=True)
                if map_all is not None:
                    current_data['map_all'] = map_all

            # 提取小目标mAP
            if 'area= small' in line or 'area=small' in line or 'area = small' in line:
                map_small = self.extract_value(self.map_small_pattern, line, fallback=True)
                if map_small is not None:
                    current_data['map_small'] = map_small

        # 添加最后一个epoch的数据
        if current_epoch is not None and current_data and len(current_data) > 1:
            results.append((current_epoch, current_data))

        return results


def create_table(results):
    """创建并输出表格"""
    if not results:
        print("未找到有效的mAP数据")
        return None, None, None

    # 准备数据
    epochs = []
    map_all_values = []
    map_small_values = []

    for epoch, data in results:
        epochs.append(epoch)
        map_all_values.append(data.get('map_all', float('nan')))
        map_small_values.append(data.get('map_small', float('nan')))

    return epochs, map_all_values, map_small_values


def print_results_table(epochs, map_all_values, map_small_values):
    """打印结果表格"""
    if not epochs:
        print("没有数据可显示")
        return None, None

    print("\n" + "=" * 60)
    print("SEADRONSEE数据集 DETR模型训练结果")
    print("华为NPU - 只显示整体mAP和小目标mAP")
    print("=" * 60)

    # 表头
    print(f"{'Epoch':<8} | {'整体mAP':<12} | {'小目标mAP':<12}")
    print("-" * 40)

    # 查找最佳整体mAP
    best_map = -1.0
    best_epoch = -1
    best_small_map = -1.0
    best_small_epoch = -1

    # 打印每行数据
    for i, epoch in enumerate(epochs):
        map_all = map_all_values[i]
        map_small = map_small_values[i]

        # 格式化显示
        map_all_str = f"{map_all:.4f}" if not float('nan') == map_all else "N/A"
        map_small_str = f"{map_small:.4f}" if not float('nan') == map_small else "N/A"

        print(f"{epoch:<8} | {map_all_str:<12} | {map_small_str:<12}")

        # 更新最佳值
        if not float('nan') == map_all and map_all > best_map:
            best_map = map_all
            best_epoch = epoch

        if not float('nan') == map_small and map_small > best_small_map:
            best_small_map = map_small
            best_small_epoch = epoch

    print("=" * 60)

    return best_epoch, best_map


def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        # 从文件读取
        log_file = Path(sys.argv[1])
        if not log_file.exists():
            print(f"错误: 文件 {log_file} 不存在")
            sys.exit(1)

        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()
        except Exception as e:
            print(f"读取文件时出错: {e}")
            sys.exit(1)

        print(f"正在解析日志文件: {log_file}")
    else:
        # 从标准输入读取
        print("请输入日志内容（Ctrl+D结束输入）:")
        log_content = sys.stdin.read()

    # 解析日志
    parser = SimpleMAPParser()
    results = parser.parse_log(log_content)

    # 创建表格
    epochs, map_all_values, map_small_values = create_table(results)

    # 打印结果
    if epochs:
        best_epoch, best_map = print_results_table(epochs, map_all_values, map_small_values)

        # 输出统计信息
        print(f"\n📊 统计信息:")
        print(f"  共找到 {len(epochs)} 个epoch的评估结果")

        if best_epoch != -1:
            print(f"\n🏆 最佳整体mAP: {best_map:.4f} (Epoch {best_epoch})")

            # 显示最佳epoch的小目标mAP
            for i, epoch in enumerate(epochs):
                if epoch == best_epoch:
                    small_map = map_small_values[i]
                    if not float('nan') == small_map:
                        print(f"  对应的小目标mAP: {small_map:.4f}")
                    break

        # 计算平均值（忽略NaN）
        valid_all_maps = [v for v in map_all_values if not float('nan') == v]
        valid_small_maps = [v for v in map_small_values if not float('nan') == v]

        if valid_all_maps:
            avg_all = sum(valid_all_maps) / len(valid_all_maps)
            print(f"  整体mAP平均值: {avg_all:.4f}")

        if valid_small_maps:
            avg_small = sum(valid_small_maps) / len(valid_small_maps)
            print(f"  小目标mAP平均值: {avg_small:.4f}")
    else:
        print("\n⚠️ 未找到任何mAP数据")
        print("\n提示: 请检查日志文件中是否包含以下格式的行:")
        print("  - 包含 'Average Precision' 和 'area= all'")
        print("  - 包含 'Average Precision' 和 'area= small'")
        print("  - 每行最后应有类似 '= 0.435' 的数值")

        # 显示日志样本帮助调试
        print("\n📋 日志样本:")
        lines = log_content.split('\n')
        sample_lines = [line for line in lines if 'Average Precision' in line][:5]
        for i, line in enumerate(sample_lines[:5]):
            print(f"  {i + 1}: {line[:100]}...")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()