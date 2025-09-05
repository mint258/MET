#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
freeze_layer_new.py: 自动从日志文件提取最佳 Validation R² 并绘制图表（改进版）
使用 finetune_vs_direct.py 中的日志解析逻辑，以避免数据读取不全的问题。
用法:
    python freeze_layer_new.py \
        --log_dir ./logs \
        --output freeze_layer.png
"""
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

# 全局样式配置
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "lines.markersize": 8,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 600,
})

# 借鉴自 finetune_vs_direct.py 的反向行迭代提取方法
R2_PATTERN = re.compile(r"validation R2 (\d+\.\d+)")

def extract_best_r2(log_path: Path) -> float:
    """
    返回日志中最后一次出现的 validation R² 值，若未匹配到则返回 nan。
    采用反向遍历行以确保取到最后一条记录。
    """
    try:
        with log_path.open('r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        for line in reversed(lines):
            m = R2_PATTERN.search(line)
            if m:
                return float(m.group(1))
    except Exception as e:
        print(f"Warning: failed to read {log_path}: {e}")
    return np.nan


def main():
    parser = argparse.ArgumentParser(
        description='自动提取日志并绘图（改进版）'
    )
    parser.add_argument('--log_dir', type=str, default='.', help='日志文件所在目录')
    parser.add_argument('--output', type=str, default='freeze_layer.png', help='输出图像文件名')
    args = parser.parse_args()

    # 定义各层参数及标签
    trainable_parameters = [7016705, 2108675, 893955, 861059]
    tick_labels = ["Embedding", "EGNN",
                   "Linear",  "Transformer"]
    bg_colors = {
        "Embedding": "#f6f6bc",
        "EGNN": "#afcf78",
        "Linear": "#eab375",
        "Transformer": "#c1bed6"
    }
    line1_color = "#8aafc9"
    line2_color = "#ea8e83"
    background_color = "white"
    font_color = "black"

    # 扫描日志文件目录，仅匹配 .log 文件
    log_dir = Path(args.log_dir)
    files = sorted(log_dir.glob('freeze_layer_*_*.log'))
    if not files:
        print(f"未在目录 {args.log_dir} 中找到任何 freeze_layer_*_*.log 文件")
        return

    # 提取 (seed, layer) -> R2
    data = {}
    seeds = set()
    for fp in files:
        fn = fp.name
        m = re.match(r'freeze_layer_(\d+)_(\d+)\.log$', fn)
        if not m:
            continue
        layer = int(m.group(1))
        seed = int(m.group(2))
        val = extract_best_r2(fp)
        data[(seed, layer)] = val
        seeds.add(seed)

    if not data:
        print("未能提取到任何 R² 数据")
        return
    seeds = sorted(seeds)

    # 构建每个 seed 对应的各层 R² 列表，并计算均值与标准差
    val_matrix = []
    for seed in seeds:
        arr = [data.get((seed, l), np.nan) for l in range(len(trainable_parameters))]
        val_matrix.append(arr)
        print(f"val_{seed} = {arr}")
    val_matrix = np.vstack(val_matrix)
    val_R2_mean = np.nanmean(val_matrix, axis=0)
    val_R2_std  = np.nanstd(val_matrix, axis=0)

    # 开始绘图
    fig, ax1 = plt.subplots(figsize=(10, 6), facecolor=background_color)
    plt.tight_layout()
    plt.xticks(rotation=45, ha='right')

    # 左轴：Trainable Parameters
    x = list(range(len(trainable_parameters), 0, -1))
    ax1.set_xlabel('Trainable Layers', color=font_color)
    ax1.set_ylabel('Trainable Parameters', color=line1_color)
    line1, = ax1.plot(
        x, trainable_parameters,
        color=line1_color, marker='o', linestyle='-', label='Parameters'
    )
    ax1.tick_params(axis='y', colors=line1_color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tick_labels, color=font_color)
    ax1.grid(False)

    # 不同层背景色
    for xi, label in zip(x, tick_labels):
        ax1.axvspan(xi - 0.5, xi + 0.5,
                    facecolor=bg_colors.get(label, '#FFFFFF'), alpha=0.3)

    # 右轴：Validation R²
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation R²', color=line2_color)
    eb = ax2.errorbar(
        x, val_R2_mean, yerr=val_R2_std,
        fmt='s--', color=line2_color,
        ecolor=line2_color, capsize=3, label='Validation R2'
    )
    line2 = eb[0]
    ax2.tick_params(axis='y', colors=line2_color)
    ax2.set_ylim(0, 1)
    ax2.grid(False)

    # 合并图例
    # ax1.legend(handles=[line1, line2], loc='upper left', fontsize=8, frameon=True)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(args.output, dpi=300, facecolor=background_color)
    plt.close(fig)
    print(f"已生成并保存图像：{args.output}")

if __name__ == '__main__':
    main()
