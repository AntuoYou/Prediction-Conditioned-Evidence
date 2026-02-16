#!/usr/bin/env python3
"""
解析SEADRONSEE数据集DETR模型训练日志中的mAP指标
专为华为NPU训练日志设计，提取整体mAP和小目标mAP
"""

import re
import sys
from collections import defaultdict
from pathlib import Path


class SimpleMAPParser:
    def __init__(self):
        # 尝试多种epoch匹配模式
        self.epoch_patterns = [
            re.compile(r'epoch\s*(\d+)', re.IGNORECASE),
            re.compile(r'Epoch\s*(\d+)', re.IGNORECASE),
            re.compile(r'EPOCH\s*(\d+)', re.IGNORECASE),
            re.compile(r'\|\s*(\d+)\s*epoch', re.IGNORECASE),
        ]

        # 匹配整体mAP: IoU=0.50:0.95 | area=all
        self.map_all_pattern = re.compile(
            r'Average Precision.*?IoU=0\.50:0\.95.*?area=\s*all.*?=\s*([0-9]+\.[0-9]+)',
            re.IGNORECASE
        )

        # 匹配小目标mAP: IoU=0.50:0.95 | area=small
        self.map_small_pattern = re.compile(
            r'Average Precision.*?IoU=0\.50:0\.95.*?area=\s*small.*?=\s*([0-9]+\.[0-9]+)',
            re.IGNORECASE
        )

        # 简单的数值提取模式
        self.simple_value_pattern = re.compile(r'=\s*([0-9]+\.[0-9]+)')

    def find_epoch(self, line):
        """尝试用多种模式匹配epoch"""
        for pattern in self.epoch_patterns:
            match = pattern.search(line)
            if match:
                try:
                    return int(match.group(1))
                except:
                    continue
        return None

    def extract_value(self, line, pattern_type="all"):
        """从行中提取数值"""
        if pattern_type == "all":
            pattern = self.map_all_pattern
        elif pattern_type == "small":
            pattern = self.map_small_pattern
        else:
            pattern = self.simple_value_pattern

        match = pattern.search(line)
        if match:
            try:
                return float(match.group(1))
            except:
                return None
        return None

    def parse_log(self, log_content):
        """解析日志内容"""
        results = defaultdict(dict)
        current_epoch = None
        lines = log_content.split('\n')

        # 先尝试找出所有可能的epoch
        for i, line in enumerate(lines):
            line = line.strip()

            # 查找epoch
            epoch = self.find_epoch(line)
            if epoch is not None:
                current_epoch = epoch
                results[current_epoch] = {}

            # 如果还没有找到epoch，但遇到了mAP相关行，假设这是第一个epoch
            if current_epoch is None and 'Average Precision' in line:
                # 向前查找epoch
                found_epoch = False
                for j in range(max(0, i - 10), i):
                    epoch = self.find_epoch(lines[j])
                    if epoch is not None:
                        current_epoch = epoch
                        results[current_epoch] = {}
                        found_epoch = True
                        break

                # 如果还没找到，创建一个虚拟epoch
                if not found_epoch:
                    current_epoch = 0
                    results[current_epoch] = {}

            # 提取整体mAP
            if current_epoch is not None and 'area= all' in line and 'IoU=0.50:0.95' in line:
                map_all = self.extract_value(line, "all")
                if map_all is not None:
                    results[current_epoch]['map_all'] = map_all

            # 提取小目标mAP
            if current_epoch is not None and 'area= small' in line and 'IoU=0.50:0.95' in line:
                map_small = self.extract_value(line, "small")
                if map_small is not None:
                    results[current_epoch]['map_small'] = map_small

            # 备选方案：更宽松的匹配
            if current_epoch is not None and 'Average Precision' in line and '=' in line:
                # 检查是否是整体mAP
                if 'area=all' in line.replace(' ', '').lower():
                    if 'map_all' not in results[current_epoch]:
                        value = self.extract_value(line, "all")
                        if value is not None:
                            results[current_epoch]['map_all'] = value

                # 检查是否是小目标mAP
                if 'area=small' in line.replace(' ', '').lower():
                    if 'map_small' not in results[current_epoch]:
                        value = self.extract_value(line, "small")
                        if value is not None:
                            results[current_epoch]['map_small'] = value

        # 转换为排序后的列表
        sorted_results = []
        for epoch in sorted(results.keys()):
            if results[epoch]:  # 只包含有数据的epoch
                sorted_results.append((epoch, results[epoch]))

        return sorted_results


def print_debug_info(log_content, results):
    """打印调试信息"""
    print("\n🔍 调试信息:")
    print("-" * 60)
    print(f"找到 {len(results)} 个epoch的数据")

    # 显示前几行日志
    print("\n日志前30行:")
    print("-" * 60)
    lines = log_content.split('\n')[:30]
    for i, line in enumerate(lines):
        print(f"{i + 1:3d}: {line}")

    print("-" * 60)

    # 显示所有包含Average Precision的行
    print("\n所有包含'Average Precision'的行:")
    print("-" * 60)
    ap_lines = [line for line in log_content.split('\n') if 'Average Precision' in line]
    for i, line in enumerate(ap_lines[:20]):  # 只显示前20行
        print(f"{i + 1:3d}: {line}")

    print("-" * 60)


def create_simple_table(results):
    """创建简化表格"""
    if not results:
        return None, None

    print("\n" + "=" * 60)
    print("SEADRONSEE数据集 DETR模型mAP结果")
    print("=" * 60)
    print(f"{'Epoch':<10} | {'整体mAP':<10} | {'小目标mAP':<10}")
    print("-" * 40)

    best_map = -1.0
    best_epoch = -1
    best_small_map = -1.0
    best_small_epoch = -1

    for epoch, data in results:
        map_all = data.get('map_all', 'N/A')
        map_small = data.get('map_small', 'N/A')

        # 格式化显示
        map_all_str = f"{map_all:.4f}" if isinstance(map_all, float) else str(map_all)
        map_small_str = f"{map_small:.4f}" if isinstance(map_small, float) else str(map_small)

        print(f"{epoch:<10} | {map_all_str:<10} | {map_small_str:<10}")

        # 更新最佳值
        if isinstance(map_all, float) and map_all > best_map:
            best_map = map_all
            best_epoch = epoch

        if isinstance(map_small, float) and map_small > best_small_map:
            best_small_map = map_small
            best_small_epoch = epoch

    print("=" * 60)

    return best_epoch, best_map


def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
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
        print("请输入日志内容（Ctrl+D结束输入）:")
        log_content = sys.stdin.read()

    # 解析日志
    parser = SimpleMAPParser()
    results = parser.parse_log(log_content)

    if results:
        best_epoch, best_map = create_simple_table(results)

        # 输出统计信息
        print(f"\n📊 统计信息:")
        print(f"  共找到 {len(results)} 个epoch的评估结果")

        # 计算平均值
        all_maps = [data.get('map_all') for _, data in results if isinstance(data.get('map_all'), float)]
        small_maps = [data.get('map_small') for _, data in results if isinstance(data.get('map_small'), float)]

        if all_maps:
            avg_all = sum(all_maps) / len(all_maps)
            print(f"  整体mAP平均值: {avg_all:.4f}")

        if small_maps:
            avg_small = sum(small_maps) / len(small_maps)
            print(f"  小目标mAP平均值: {avg_small:.4f}")

        if best_epoch != -1:
            print(f"\n🏆 最佳整体mAP: {best_map:.4f} (Epoch {best_epoch})")

            # 显示最佳epoch的小目标mAP
            for epoch, data in results:
                if epoch == best_epoch:
                    small_map = data.get('map_small')
                    if isinstance(small_map, float):
                        print(f"  对应的小目标mAP: {small_map:.4f}")
                    break
    else:
        print("未找到任何mAP数据")
        print_debug_info(log_content, results)

        # 提供手动解析选项
        print("\n💡 手动解析尝试:")
        print("正在搜索包含关键字的行...")

        lines = log_content.split('\n')
        all_count = 0
        small_count = 0

        for i, line in enumerate(lines):
            if 'Average Precision' in line:
                # 检查是否包含整体mAP
                if 'area=all' in line.replace(' ', '').lower() and 'IoU=0.50:0.95' in line:
                    all_count += 1
                    # 提取数值
                    parts = line.split('=')
                    if len(parts) > 1:
                        value = parts[-1].strip()
                        print(f"  找到整体mAP行 {all_count}: {line[:80]}... -> 值: {value}")

                # 检查是否包含小目标mAP
                elif 'area=small' in line.replace(' ', '').lower() and 'IoU=0.50:0.95' in line:
                    small_count += 1
                    # 提取数值
                    parts = line.split('=')
                    if len(parts) > 1:
                        value = parts[-1].strip()
                        print(f"  找到小目标mAP行 {small_count}: {line[:80]}... -> 值: {value}")

        print(f"\n  总计找到: {all_count} 个整体mAP, {small_count} 个小目标mAP")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()