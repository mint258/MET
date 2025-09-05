# -*- coding: utf-8 -*-
"""
Nature-style rcParams & 专用调色板
在所有绘图脚本的最前面 `import nature_mpl_style as nstyle`
即可一键生效
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ─── 1. 全局排版 ───────────────────────────────────────
mpl.rcParams.update({
    # 分辨率 & 字体
    'figure.dpi'      : 300,
    'font.family'     : 'Times New Roman',
    'font.size'       : 9,     # 正文等同大小
    'axes.labelsize'  : 10,
    'axes.titlesize'  : 10,
    'xtick.labelsize' : 9,
    'ytick.labelsize' : 9,
    'legend.fontsize' : 9,
    # 线宽/边框
    'axes.linewidth'  : 1.0,
    'lines.linewidth' : 1.2,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'xtick.direction' : 'out',
    'ytick.direction' : 'out',
    'savefig.transparent': True,
})

# ─── 2. 专用调色板（6 聚合物 + 2 通用） ────────────────
POLY_COLORS = {  # HEX 取自上条回答
    'F0F0':'#909090', 'F0F1':'#4C72B0', 'F0F2':'#56B4E9',
    'F1F1':'#D55E00', 'F1F2':'#73B55B', 'F2F2':'#CC79A7'
}
ACCENT1 = '#4C72B0'  # 主蓝
ACCENT2 = '#D55E00'  # 主橙

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=list(POLY_COLORS.values())
                                             + [ACCENT1, ACCENT2])

# ─── 3. Marker 统一放大 ───────────────────────────────
MARKER_KW = dict(ms=6, mec='white', mew=0.4)   # 6 pt ≈ 50 px

def hide_right_top(ax):
    """去掉右/上边框"""
    for sp in ['right', 'top']:
        ax.spines[sp].set_visible(False)

def panel_label(ax, label):
    """左上角面板字母：Arial Bold, 12 pt"""
    ax.text(-0.08, 1.02, label, transform=ax.transAxes,
            ha='left', va='bottom',
            fontsize=12, fontweight='bold',
            family='Arial')

# （该模块无需执行主程序）
