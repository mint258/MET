#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_vs_direct.py
=====================

从形如
    finetune_data_<dataset>_<seed>.log
    direct_data_<dataset>_<seed>.log
的日志文件中，自动提取最后一次
    "Saved best model with validation R2 <value> ..."
里的 R²，按照 <dataset> 分组，
计算各数据规模下『微调 / 从零训练』的
平均值与标准差并绘图。

Usage
-----
python finetune_vs_direct.py \
    --log-dir  ./logs \
    --out      fine_vs_direct.png
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'

R2_PATTERN = re.compile(r"validation R2 (\d+\.\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract R² from finetune / direct logs and plot."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("."),
        help="Directory containing *.log files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("finetune_vs_direct_r2.png"),
        help="Output PNG filename.",
    )
    return parser.parse_args()


def extract_r2(file_path: Path) -> float | None:
    """Return the LAST occurrence of R² in the log file."""
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in reversed(f.readlines()):
                m = R2_PATTERN.search(line)
                if m:
                    return float(m.group(1))
    except Exception as exc:  # pragma: no cover
        logging.warning("Failed to read %s: %s", file_path, exc)
    return None


def collect_stats(
    log_dir: Path,
) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
    """Scan directory and build {dataset_size: [r2,...]} dicts."""
    finetune: Dict[int, List[float]] = defaultdict(list)
    direct: Dict[int, List[float]] = defaultdict(list)

    for f in log_dir.glob("*.log"):
        name = f.name
        if name.startswith("finetune_data_"):
            m = re.match(r"finetune_data_(\d+)_(\d+)\.log", name)
            if not m:
                continue
            size = int(m.group(1))
            val = extract_r2(f)
            if val is not None:
                finetune[size].append(val)
        elif name.startswith("direct_data_"):
            m = re.match(r"direct_data_(\d+)_(\d+)\.log", name)
            if not m:
                continue
            size = int(m.group(1))
            val = extract_r2(f)
            if val is not None:
                direct[size].append(val)

    return finetune, direct


def compute_mean_std(
    data: Dict[int, List[float]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted sizes, means, stds."""
    if not data:
        return np.array([]), np.array([]), np.array([])
    sizes = np.array(sorted(data.keys()))
    means = np.array([np.mean(data[s]) for s in sizes])
    stds = np.array([np.std(data[s], ddof=1) if len(data[s]) > 1 else 0.0 for s in sizes])
    return sizes, means, stds


def set_nature_style() -> None:
    plt.rcParams.update(
        {
        "font.family":      "Arial",
        "font.size":        14,
        "axes.edgecolor":   "black",
        "axes.linewidth":   1.0,
        "grid.linestyle":   "",
        "grid.color":       "grey",
        "grid.linewidth":   0.5,
        "xtick.labelsize":  14,
        "ytick.labelsize":  14,
        "lines.markersize": 8,
        "lines.linewidth":  2.0,
        }
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    finetune_stats, direct_stats = collect_stats(args.log_dir)
    if not finetune_stats and not direct_stats:
        logging.error("未在 %s 中找到符合命名规则的日志文件。", args.log_dir)
        sys.exit(1)

    s_f, mean_f, std_f = compute_mean_std(finetune_stats)
    s_d, mean_d, std_d = compute_mean_std(direct_stats)

    # 交叉检查数据规模一致性
    all_sizes = sorted(set(s_f) | set(s_d))
    logging.info("数据规模: %s", all_sizes)

    if not np.array_equal(s_f, s_d):
        logging.warning("finetune 与 direct 的数据规模不完全一致，图中将缺失缺口。")

    # 绘图
    set_nature_style()
    background = "white"
    plt.figure(figsize=(6.5, 6), facecolor=background)

    if s_f.size:
        plt.errorbar(
            np.log10(s_f),
            mean_f,
            yerr=std_f,
            fmt="o-",
            capsize=4,
            label="Finetuned Model",
            color="#8aafc9",
        )
    if s_d.size:
        plt.errorbar(
            np.log10(s_d),
            mean_d,
            yerr=std_d,
            fmt="o-",
            capsize=4,
            label="Train from scratch Model",
            color="#ea8e83",
        )

    plt.xlabel("Training set size")
    plt.ylabel("R²")

    # 使用对数刻度但显式显示原始数值
    x_ticks = np.log10(all_sizes)
    plt.xticks(x_ticks, labels=[f"{x:,}" for x in all_sizes])
    plt.grid(True, which="both")
    # 自定义图例：保持 marker 的颜色与尺寸一致，但不显示误差棒/连线
    from matplotlib.lines import Line2D as _Line2D
    from matplotlib.container import ErrorbarContainer as _ErrC
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    proxies = []
    for h in handles:
        # 解包 ErrorbarContainer，取其中的 Line2D 以读取样式
        line = None
        if isinstance(h, _ErrC):
            if hasattr(h, 'lines') and h.lines:
                line = h.lines[0]
            else:
                try:
                    line = h[0]
                except Exception:
                    line = None
        else:
            line = h
        # 兜底
        if line is None:
            line = h
        # 读取 marker/尺寸/颜色
        marker = getattr(line, 'get_marker', lambda: 'o')() or 'o'
        ms = getattr(line, 'get_markersize', lambda: 6)() or 6
        base = getattr(line, 'get_color', lambda: 'black')() or 'black'
        def _col(name, fallback):
            v = getattr(line, name, lambda: fallback)()
            if isinstance(v, str) and v.lower() == 'auto':
                return fallback
            return v
        mfc = _col('get_markerfacecolor', base) or base
        mec = _col('get_markeredgecolor', base) or base
        proxies.append(_Line2D([], [], linestyle='None', marker=marker, markersize=ms,
                               markerfacecolor=mfc, markeredgecolor=mec))
    ax.legend(handles=proxies, labels=labels, frameon=False)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, facecolor=background)
    logging.info("已保存图像至 %s", args.out)


if __name__ == "__main__":
    main()
