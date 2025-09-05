#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
combine_figures.py
------------------
把 (a) finetune_vs_direct、(b) atom_embedding_dim、(c) property_predict
三张单独生成的 PNG/PDF 合并成一张 2×2 Mosaic：

 ┌───────────────┬───────────────┐
 │      (a)      │      (b)      │
 ├───────────────┴───────────────┤
 │              (c)              │
 └───────────────────────────────┘

用法
----
python combine_figures.py \
    --fig-a finetune_vs_direct_nature.png \
    --fig-b atom_embedding_dim_nature.png \
    --fig-c compare_plot.png \
    --out  figure_combo.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image  as mpimg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Combine separate figures into a mosaic.")
    p.add_argument("--fig-a", type=Path, default=Path("finetune_vs_direct_r2.png"),
                   help="Path to finetune_vs_direct figure.")
    p.add_argument("--fig-b", type=Path, default=Path("dipole_compare_20000_new.png"),
                   help="Path to property_predict figure.")
    p.add_argument("--fig-c", type=Path, default=Path("atom_embedding_dim_auto.png"),
                   help="Path to atom_embedding_dim figure.")
    p.add_argument("--out",   type=Path, default=Path("figure_combo.png"),
                   help="Output combined figure (png / pdf / svg).")
    return p.parse_args()


def add_panel_label(ax, label: str) -> None:
    """在子图左上角添加面板字母"""
    ax.text(-0.05, 1.05, label,
            transform=ax.transAxes,
            fontsize=12, fontweight="bold",
            family="Arial", va="bottom", ha="left")


def main() -> None:
    args = parse_args()

    # ─── 读取图片 ───────────────────────────────────────
    img_a = mpimg.imread(args.fig_a)
    img_b = mpimg.imread(args.fig_b)
    img_c = mpimg.imread(args.fig_c)

    # ─── 创建 2×2 Mosaic: top 两列, bottom 跨两列 ───────
    fig = plt.figure(figsize=(7.2, 6.4), constrained_layout=True)  # 单栏≈3.5 in → 双栏≈7 in
    mosaic = """
        AB
        CC
    """
    ax_dict = fig.subplot_mosaic(mosaic, gridspec_kw={"height_ratios": [1, 1]})

    # ─── 放置三张图 ─────────────────────────────────────
    ax_dict["A"].imshow(img_a); ax_dict["A"].axis("off"); add_panel_label(ax_dict["A"], "a")
    ax_dict["B"].imshow(img_b); ax_dict["B"].axis("off"); add_panel_label(ax_dict["B"], "b")
    ax_dict["C"].imshow(img_c); ax_dict["C"].axis("off"); add_panel_label(ax_dict["C"], "c")

    # ─── 保存 ───────────────────────────────────────────
    out_path: Path = args.out.with_suffix(args.out.suffix or ".png")
    fig.savefig(out_path, dpi=300)
    print(f"✓ Combined figure saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
