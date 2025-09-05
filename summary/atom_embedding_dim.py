#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_atom_embedding_dim.py
==========================

▪ 自动汇总 dipole{N}/atom_dim{dim}_{seed} 文件中的
  “Saved best model with validation R2 …” 数值；
▪ 计算平均值 / 标准差；
▪ 绘制 Charge Val R² 与三组 Subsequent-task Val R² 曲线（mean±std）；
▪ Nature 风格输出 PNG。

用法:
    python auto_atom_embedding_dim.py            # 默认当前目录
    python auto_atom_embedding_dim.py --root ~/path/to/exp_root
"""

import re
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib as mpl

# ───────── CLI ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path("."), help="实验根目录, 其下包含 dipole1000/5000/100000")
parser.add_argument("--path", type=Path, default="atom_embedding_dim_auto.svg", help="output path")
args = parser.parse_args()
root: Path = args.root.expanduser().resolve()

# ───────── 配置 ────────────────────────────────────────────────────────────────
DATASETS = {
    "1000":  dict(folder="dipole1000",  linestyle="--", color="#ea8e83", marker="s"),
    "5000":  dict(folder="dipole5000",  linestyle="-.", color="#ea8e83", marker="p"),
    "100000":dict(folder="dipole100000",linestyle=":" , color="#ea8e83", marker="^"),
}
ATOM_DIMS = [8, 16, 32, 64, 128, 256]

# 左轴固定的 Charge Val R²（如需自动化可改写同样的统计逻辑）
charge_val_R2 = np.array([0.9985, 0.9991, 0.9989, 0.9992, 0.9992, 0.9991])

# ───────── 正则表达式 & 统计 ──────────────────────────────────────────────────
pat_r2 = re.compile(r"Saved best model with validation R2 ([0-9]*\.?[0-9]+)")

summary_mean = {}  # dataset_size -> np.array(len=6)
summary_std  = {}

for size, cfg in DATASETS.items():
    means, stds = [], []
    dpath = root / cfg["folder"]
    if not dpath.is_dir():
        raise FileNotFoundError(f"目录 {dpath} 不存在")

    for dim in ATOM_DIMS:
        r2_vals = []
        for file in dpath.glob(f"atom_dim{dim}_*"):
            if not file.is_file():
                continue
            with file.open("r") as fh:
                # 倒序搜索速度快
                for line in reversed(fh.readlines()):
                    m = pat_r2.search(line)
                    if m:
                        r2_vals.append(float(m.group(1)))
                        break
        if not r2_vals:
            print(f"[WARN] {cfg['folder']} 维度 {dim} 未找到 R²，填 nan")
            means.append(np.nan)
            stds.append(np.nan)
        else:
            r2_arr = np.asarray(r2_vals)
            means.append(r2_arr.mean())
            stds.append(r2_arr.std(ddof=1))
    summary_mean[size] = np.array(means)
    summary_std[size]  = np.array(stds)

# ───────── 绘图 (Nature 风格) ─────────────────────────────────────────────────
plt.rcParams.update({
     "font.family":      "Arial",
     "font.size":        14,
     "axes.edgecolor":   "black",
     "axes.linewidth":   1.0,
     "grid.color":       "grey",
     "grid.linestyle":   "",
     "grid.linewidth":   0.5,
     "lines.markersize": 8,
     "lines.linewidth":  2.0,
 })
mpl.rcParams['svg.fonttype'] = 'none'

fig, ax1 = plt.subplots(figsize=(7, 6), facecolor="white")

# ─── 对数刻度，让相同倍数（8→16→32→64）在画布上等距离分布 ────────────────
ax1.set_xscale('log')
# 强制主刻度就在这些原始维度上
ax1.set_xticks(ATOM_DIMS)
# 保证标签就地输出原始数值（而不是 10^x 之类的）
ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax1.xaxis.set_minor_formatter(mticker.NullFormatter())

# 左轴: Charge Val R²
ax1.plot(ATOM_DIMS, charge_val_R2,
         marker="o", color="#8aafc9", label="Charge Val R²")
ax1.set_xlabel("Atom Embedding Dimension")
ax1.set_ylabel("Charge R²", color='#8aafc9')
ax1.set_xticks(ATOM_DIMS)
ax1.set_ylim(0.99, 1.0)
ax1.tick_params(axis="y", colors='#8aafc9')
ax1.grid(True, which="both")

# 右轴: Subsequent-task Val R² (三条曲线)
ax2 = ax1.twinx()
ax2.set_ylabel("Subsequent Task Val R²", color='#ea8e83')
ax2.set_ylim(0, 1.0)
ax2.spines['left'].set_color('#8aafc9')
ax2.spines['right'].set_color('#ea8e83')
ax2.tick_params(axis="y", colors='#ea8e83')

for size, cfg in DATASETS.items():
    mean = summary_mean[size]
    std  = summary_std[size]
    label = f"{size} molecules"
    ax2.errorbar(ATOM_DIMS, mean, yerr=std,
                 linestyle=cfg["linestyle"],
                 color=cfg["color"], capsize=3, label=label, marker=cfg["marker"], markerfacecolor="none", markeredgecolor="#ea8e83", markeredgewidth=1.5)

# 图例移到图内左下角，四行一列
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2,
           l1 + l2,
           loc='lower right',
           frameon=False,
           ncol=1)

plt.tight_layout()
out_path = args.path
plt.savefig(out_path, dpi=300, facecolor="white", bbox_inches='tight')
print(f"[✓] 图已保存至 {out_path}")
