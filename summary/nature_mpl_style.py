# -*- coding: utf-8 -*-
"""
Nature-style rcParams & ר�õ�ɫ��
�����л�ͼ�ű�����ǰ�� `import nature_mpl_style as nstyle`
����һ����Ч
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ������ 1. ȫ���Ű� ������������������������������������������������������������������������������
mpl.rcParams.update({
    # �ֱ��� & ����
    'figure.dpi'      : 300,
    'font.family'     : 'Times New Roman',
    'font.size'       : 9,     # ���ĵ�ͬ��С
    'axes.labelsize'  : 10,
    'axes.titlesize'  : 10,
    'xtick.labelsize' : 9,
    'ytick.labelsize' : 9,
    'legend.fontsize' : 9,
    # �߿�/�߿�
    'axes.linewidth'  : 1.0,
    'lines.linewidth' : 1.2,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'xtick.direction' : 'out',
    'ytick.direction' : 'out',
    'savefig.transparent': True,
})

# ������ 2. ר�õ�ɫ�壨6 �ۺ��� + 2 ͨ�ã� ��������������������������������
POLY_COLORS = {  # HEX ȡ�������ش�
    'F0F0':'#909090', 'F0F1':'#4C72B0', 'F0F2':'#56B4E9',
    'F1F1':'#D55E00', 'F1F2':'#73B55B', 'F2F2':'#CC79A7'
}
ACCENT1 = '#4C72B0'  # ����
ACCENT2 = '#D55E00'  # ����

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=list(POLY_COLORS.values())
                                             + [ACCENT1, ACCENT2])

# ������ 3. Marker ͳһ�Ŵ� ��������������������������������������������������������������
MARKER_KW = dict(ms=6, mec='white', mew=0.4)   # 6 pt �� 50 px

def hide_right_top(ax):
    """ȥ����/�ϱ߿�"""
    for sp in ['right', 'top']:
        ax.spines[sp].set_visible(False)

def panel_label(ax, label):
    """���Ͻ������ĸ��Arial Bold, 12 pt"""
    ax.text(-0.08, 1.02, label, transform=ax.transAxes,
            ha='left', va='bottom',
            fontsize=12, fontweight='bold',
            family='Arial')

# ����ģ������ִ��������
