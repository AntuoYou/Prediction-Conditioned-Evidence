#!/usr/bin/env python3
"""
解析SEADRONSEE数据集DETR模型训练日志中的mAP指标
支持从文件或标准输入读取日志，输出表格化结果并标记最佳mAP
"""

import re
import sys
from collections import defaultdict
from pathlib import Path


class COCOLogParser:
    def __init__(self):
        # 定义正则表达式匹配模式
        self.patterns = {
            'epoch': re.compile(r'.*epoch\s+(\d+).*', re.IGNORECASE),
            'ap_all': re.compile(r'Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95.*=\s*([\d\.]+)'),
            'ap_50': re.compile(r'Average Precision\s+\(AP\) @\[ IoU=0\.50\s*\].*=\s*([\d\.]+)'),
            'ap_75': re.compile(r'Average Precision\s+\(AP\) @\[ IoU=0\.75\s*\].*=\s*([\d\.]+)'),
            'ap_small': re.compile(r'Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95.*area=\s*small.*=\s*([\d\.]+)'),
            'ap_medium': re.compile(r'Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95.*area=\s*medium.*=\s*([\d\.]+)'),
            'ap_large': re.compile(r'Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95.*area=\s*large.*=\s*([\d\.]+)'),
        }

    def parse_log(self, log_content):
        """解析日志内容，提取各epoch的mAP指标"""
        results = []
        current_epoch = None
        current_data = {}

        for line in log_content.split('\n'):
            line = line.strip()
            if not line:
                continue

            # 检测epoch行
            epoch_match = self.patterns['epoch'].match(line)
            if epoch_match:
                if current_epoch is not None and current_data:
                    results.append((current_epoch, current_data))
                current_epoch = int(epoch_match.group(1))
                current_data = {}
                continue

            # 匹配各项指标
            for key, pattern in self.patterns.items():
                if key == 'epoch':
                    continue
                match = pattern.search(line)
                if match:
                    current_data[key] = float(match.group(1))

        # 添加最后一个epoch的数据
        if current_epoch is not None and current_data:
            results.append((current_epoch, current_data))

        return results


def format_table(results):
    """格式化输出表格"""
    if not results:
        print("未找到有效的评估结果")
        return

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
    print("SEADRONSEE数据集 DETR模型训练结果汇总")
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

    # 输出最佳结果
    if best_epoch != -1:
        print(f"\n🏆 最佳模型: Epoch {best_epoch}")
        print(f"   最佳mAP: {best_mAP:.3f} @ IoU=0.50:0.95")

        # 输出该epoch的详细指标
        for epoch, data in results:
            if epoch == best_epoch:
                print(f"\n   Epoch {best_epoch} 详细指标:")
                print(f"   AP@0.50: {data.get('ap_50', 0.0):.3f}")
                print(f"   AP@0.75: {data.get('ap_75', 0.0):.3f}")
                print(f"   AP-Small: {data.get('ap_small', 0.0):.3f}")
                print(f"   AP-Medium: {data.get('ap_medium', 0.0):.3f}")
                print(f"   AP-Large: {data.get('ap_large', 0.0):.3f}")
                break
    else:
        print("\n⚠️  未找到有效的mAP指标")

    print("=" * 80)


def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        # 从文件读取
        log_file = Path(sys.argv[1])
        if not log_file.exists():
            print(f"错误: 文件 {log_file} 不存在")
            sys.exit(1)

        log_content = log_file.read_text(encoding='utf-8')
        print(f"正在解析日志文件: {log_file}")
    else:
        # 从标准输入读取
        print("请输入日志内容（Ctrl+D结束输入）:")
        log_content = sys.stdin.read()

    # 解析日志
    parser = COCOLogParser()
    results = parser.parse_log(log_content)

    # 输出结果
    if results:
        format_table(results)
    else:
        print("未在日志中找到任何epoch的评估结果")
        print("\n提示: 请确保日志中包含以下格式的行:")
        print("  - 'epoch X' 标识每个epoch")
        print("  - 'Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = X.XXX'")


if __name__ == "__main__":
    main()