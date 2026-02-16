#!/usr/bin/env python3
"""
解析SEADRONSEE数据集DETR模型训练日志中的mAP指标（华为NPU兼容版）
支持灵活的日志格式，输出表格化结果并标记最佳mAP
"""

import re
import sys
from pathlib import Path


class FlexibleCOCOLogParser:
    def __init__(self):
        # 更灵活的正则表达式，允许各种空格和格式变化
        self.patterns = {
            # 更灵活的epoch匹配，支持各种格式
            'epoch': [
                re.compile(r'epoch\s*(\d+)', re.IGNORECASE),
                re.compile(r'Epoch\s*(\d+)', re.IGNORECASE),
                re.compile(r'EPOCH\s*(\d+)', re.IGNORECASE),
                re.compile(r'\|\s*(\d+)\s*epoch', re.IGNORECASE),
                re.compile(r'epoch:?\s*(\d+)', re.IGNORECASE),
            ],
            # mAP主要指标 - 非常灵活的正则表达式
            'ap_all': re.compile(
                r'Average\s*Precision.*?\(?AP\)?.*?'
                r'(?:IoU|iou|IOU)[=\s:]*0\.50:0\.95.*?'
                r'(?:area|AREA).*?(?:all|ALL).*?'
                r'(?:maxDets|maxDet|maxDetections).*?100.*?'
                r'[:=\s]*\s*([0-9]+\.[0-9]+|[0-9]+)',
                re.IGNORECASE
            ),
            # AP50
            'ap_50': re.compile(
                r'Average\s*Precision.*?\(?AP\)?.*?'
                r'(?:IoU|iou|IOU)[=\s:]*0\.50\b.*?'
                r'(?:area|AREA).*?(?:all|ALL).*?'
                r'(?:maxDets|maxDet|maxDetections).*?100.*?'
                r'[:=\s]*\s*([0-9]+\.[0-9]+|[0-9]+)',
                re.IGNORECASE
            ),
            # AP75
            'ap_75': re.compile(
                r'Average\s*Precision.*?\(?AP\)?.*?'
                r'(?:IoU|iou|IOU)[=\s:]*0\.75\b.*?'
                r'(?:area|AREA).*?(?:all|ALL).*?'
                r'(?:maxDets|maxDet|maxDetections).*?100.*?'
                r'[:=\s]*\s*([0-9]+\.[0-9]+|[0-9]+)',
                re.IGNORECASE
            ),
            # AP Small
            'ap_small': re.compile(
                r'Average\s*Precision.*?\(?AP\)?.*?'
                r'(?:IoU|iou|IOU)[=\s:]*0\.50:0\.95.*?'
                r'(?:area|AREA).*?(?:small|SMALL).*?'
                r'(?:maxDets|maxDet|maxDetections).*?100.*?'
                r'[:=\s]*\s*([0-9]+\.[0-9]+|[0-9]+)',
                re.IGNORECASE
            ),
            # AP Medium
            'ap_medium': re.compile(
                r'Average\s*Precision.*?\(?AP\)?.*?'
                r'(?:IoU|iou|IOU)[=\s:]*0\.50:0\.95.*?'
                r'(?:area|AREA).*?(?:medium|MEDIUM).*?'
                r'(?:maxDets|maxDet|maxDetections).*?100.*?'
                r'[:=\s]*\s*([0-9]+\.[0-9]+|[0-9]+)',
                re.IGNORECASE
            ),
            # AP Large
            'ap_large': re.compile(
                r'Average\s*Precision.*?\(?AP\)?.*?'
                r'(?:IoU|iou|IOU)[=\s:]*0\.50:0\.95.*?'
                r'(?:area|AREA).*?(?:large|LARGE).*?'
                r'(?:maxDets|maxDet|maxDetections).*?100.*?'
                r'[:=\s]*\s*([0-9]+\.[0-9]+|[0-9]+)',
                re.IGNORECASE
            ),
            # 简化的mAP匹配（如果上面的匹配失败）
            'simple_map': re.compile(
                r'(?:mAP|MAP|map|AP).*?[=\s:]*\s*([0-9]+\.[0-9]+|[0-9]+)',
                re.IGNORECASE
            ),
        }

    def find_epoch(self, line):
        """尝试用多种模式匹配epoch"""
        for pattern in self.patterns['epoch']:
            match = pattern.search(line)
            if match:
                return int(match.group(1))
        return None

    def extract_value(self, pattern, line):
        """从单行中提取数值"""
        match = pattern.search(line)
        if match:
            try:
                return float(match.group(1))
            except:
                return None
        return None

    def parse_log(self, log_content):
        """解析日志内容，提取各epoch的mAP指标"""
        results = []
        current_epoch = None
        current_data = {}
        lines = log_content.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # 尝试检测epoch
            epoch = self.find_epoch(line)
            if epoch is not None:
                # 保存之前的epoch数据（如果有）
                if current_epoch is not None and current_data:
                    results.append((current_epoch, current_data.copy()))
                current_epoch = epoch
                current_data = {'epoch': epoch}
                continue

            # 如果还没有找到epoch，但遇到了数值，可能这是第一个epoch
            if current_epoch is None and any(char.isdigit() and '.' in line for char in line):
                # 尝试查找是否有隐藏的epoch信息
                for j in range(max(0, i - 3), i):
                    epoch = self.find_epoch(lines[j])
                    if epoch is not None:
                        current_epoch = epoch
                        current_data = {'epoch': epoch}
                        break
                if current_epoch is None:
                    # 假设这是第一个epoch
                    current_epoch = 0
                    current_data = {'epoch': 0}

            # 尝试提取各种mAP指标
            if current_epoch is not None:
                for key in ['ap_all', 'ap_50', 'ap_75', 'ap_small', 'ap_medium', 'ap_large']:
                    if key in self.patterns:
                        value = self.extract_value(self.patterns[key], line)
                        if value is not None:
                            current_data[key] = value

                # 如果ap_all没找到，尝试简单匹配
                if 'ap_all' not in current_data:
                    value = self.extract_value(self.patterns['simple_map'], line)
                    if value is not None:
                        current_data['ap_all'] = value

        # 添加最后一个epoch的数据
        if current_epoch is not None and current_data:
            results.append((current_epoch, current_data))

        # 如果没有明确的epoch信息，但找到了数据，将其视为单个epoch
        if not results and any('ap_all' in data for data in [current_data] if current_data):
            results.append((0, current_data))

        return results


def print_debug_info(log_content, results):
    """打印调试信息以帮助理解解析过程"""
    print("\n🔍 调试信息:")
    print("-" * 60)

    # 显示找到的epoch数量
    print(f"找到 {len(results)} 个epoch的数据")

    # 显示每个epoch找到的指标
    for epoch, data in results:
        print(f"\nEpoch {epoch}:")
        for key, value in data.items():
            if key != 'epoch':
                print(f"  {key}: {value}")

    # 显示日志样本
    print(f"\n📋 日志样本（前20行）:")
    print("-" * 60)
    lines = log_content.split('\n')[:20]
    for i, line in enumerate(lines):
        print(f"{i + 1:3d}: {line}")

    print("-" * 60)


def format_table(results):
    """格式化输出表格"""
    if not results:
        print("⚠️ 未找到有效的评估结果")
        return None, None

    # 表头
    headers = ["Epoch", "mAP", "AP50", "AP75", "AP-S", "AP-M", "AP-L"]

    # 准备表格数据
    table_data = []
    best_mAP = -1.0
    best_epoch = -1

    for epoch, data in results:
        # 获取主要指标（如果存在）
        mAP = data.get('ap_all', 0.0)
        row = [
            str(epoch),
            f"{mAP:.3f}" if 'ap_all' in data else "N/A",
            f"{data.get('ap_50', 0.0):.3f}" if 'ap_50' in data else "N/A",
            f"{data.get('ap_75', 0.0):.3f}" if 'ap_75' in data else "N/A",
            f"{data.get('ap_small', 0.0):.3f}" if 'ap_small' in data else "N/A",
            f"{data.get('ap_medium', 0.0):.3f}" if 'ap_medium' in data else "N/A",
            f"{data.get('ap_large', 0.0):.3f}" if 'ap_large' in data else "N/A",
        ]
        table_data.append(row)

        # 更新最佳mAP
        if mAP > best_mAP and 'ap_all' in data:
            best_mAP = mAP
            best_epoch = epoch

    # 输出表格
    print("\n" + "=" * 80)
    print("SEADRONSEE数据集 DETR模型训练结果汇总（华为NPU）")
    print("=" * 80)

    # 简单制表符格式的表格
    col_widths = [10, 8, 8, 8, 8, 8, 8]

    # 打印表头
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)
    print(header_line)
    print(separator)

    # 打印数据行
    for row in table_data:
        row_line = " | ".join(val.ljust(w) for val, w in zip(row, col_widths))
        print(row_line)

    print("=" * 80)

    return best_epoch, best_mAP


def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        # 从文件读取
        log_file = Path(sys.argv[1])
        if not log_file.exists():
            print(f"❌ 错误: 文件 {log_file} 不存在")
            sys.exit(1)

        try:
            log_content = log_file.read_text(encoding='utf-8', errors='ignore')
        except:
            print(f"❌ 错误: 无法读取文件 {log_file}")
            sys.exit(1)

        print(f"📁 正在解析日志文件: {log_file}")
    else:
        # 从标准输入读取
        print("📝 请输入日志内容（Ctrl+D结束输入）:")
        log_content = sys.stdin.read()

    # 检查日志内容
    if not log_content or len(log_content.strip()) == 0:
        print("❌ 错误: 日志内容为空")
        sys.exit(1)

    # 解析日志
    parser = FlexibleCOCOLogParser()
    results = parser.parse_log(log_content)

    # 输出结果
    best_epoch, best_mAP = format_table(results)

    # 输出最佳结果
    if best_epoch != -1:
        print(f"\n🏆 最佳模型: Epoch {best_epoch}")
        print(f"   最佳mAP: {best_mAP:.3f} @ IoU=0.50:0.95")

        # 输出该epoch的详细指标
        for epoch, data in results:
            if epoch == best_epoch:
                print(f"\n   Epoch {best_epoch} 详细指标:")
                if 'ap_50' in data:
                    print(f"   AP@0.50: {data.get('ap_50', 0.0):.3f}")
                if 'ap_75' in data:
                    print(f"   AP@0.75: {data.get('ap_75', 0.0):.3f}")
                if 'ap_small' in data:
                    print(f"   AP-Small: {data.get('ap_small', 0.0):.3f}")
                if 'ap_medium' in data:
                    print(f"   AP-Medium: {data.get('ap_medium', 0.0):.3f}")
                if 'ap_large' in data:
                    print(f"   AP-Large: {data.get('ap_large', 0.0):.3f}")
                break
    else:
        print("\n⚠️ 未找到有效的mAP指标")
        print_debug_info(log_content, results)

        # 提供更多帮助信息
        print(f"\n💡 调试提示:")
        print("1. 检查日志文件是否包含COCO评估输出")
        print("2. 可以尝试在日志中搜索以下关键词:")
        print("   - 'Average Precision'")
        print("   - 'AP @[ IoU='")
        print("   - 'mAP'")
        print("   - 'epoch'")
        print("3. 如果日志格式特殊，可以分享部分日志内容以便进一步调整解析器")

    print("=" * 80)


if __name__ == "__main__":
    main()