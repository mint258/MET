#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compose_4svg_panel.py
---------------------
手动通过命令行参数选择 4 张 SVG，拼成 2x2 面板，并导出 PDF。
为避免左侧（或任意边）被裁切，导出前自动给根 <svg> 画布四周加页边距。

依赖：svgutils, lxml, cairosvg
    pip install svgutils lxml cairosvg

示例：
    python compose_4svg_panel.py \
        --a fig1.svg --b fig2.svg --c fig3.svg --d fig4.svg \
        --out-pdf combine.pdf \
        --cell-w 360 --cell-h 270 --pad-x 30 --pad-y 30 --margin 20
"""
import argparse
from pathlib import Path
from typing import Tuple

from lxml import etree
from svgutils.transform import fromfile, SVGFigure
from cairosvg import svg2pdf

def _parse_len(s: str) -> float:
    """解析带单位的长度（px/pt/mm/cm/in），返回 float（单位按像素近似对待）。"""
    if s is None:
        return 0.0
    s = str(s).strip().lower()
    for u in ('px', 'pt', 'mm', 'cm', 'in'):
        if s.endswith(u):
            s = s[:-len(u)]
            break
    try:
        return float(s)
    except Exception:
        return 0.0

def _load_svg_get_size(svg_path: Path) -> Tuple[float, float]:
    """用 lxml 读取 SVG 根节点，返回 (width, height)。若缺失则从 viewBox 推导。"""
    parser = etree.XMLParser(remove_blank_text=False, recover=True)
    tree = etree.parse(str(svg_path), parser)
    root = tree.getroot()
    w = _parse_len(root.get('width'))
    h = _parse_len(root.get('height'))
    vb = root.get('viewBox')
    if (w <= 0 or h <= 0) and vb:
        vals = [v for v in vb.replace(',', ' ').split() if v.strip()]
        if len(vals) >= 4:
            try:
                w = float(vals[2])
                h = float(vals[3])
            except Exception:
                pass
    if w <= 0: w = 100.0
    if h <= 0: h = 100.0
    return w, h

def _expand_svg_canvas(svg_path: Path, margin: float = 20.0) -> None:
    """在不改变图形元素的前提下，扩大根 <svg> 的 viewBox 与 width/height，避免裁切。"""
    try:
        parser = etree.XMLParser(remove_blank_text=False, recover=True)
        tree = etree.parse(str(svg_path), parser)
        root = tree.getroot()
        if not root.tag.lower().endswith('svg'):
            return

        vb = root.get('viewBox')
        if vb is None:
            # 尝试由 width/height 推 viewBox
            w = _parse_len(root.get('width'))
            h = _parse_len(root.get('height'))
            if w <= 0 or h <= 0:
                return
            x0 = 0.0; y0 = 0.0
        else:
            vals = [v for v in vb.replace(',', ' ').split() if v.strip()]
            if len(vals) < 4:
                return
            x0, y0, w, h = [float(vals[i]) for i in range(4)]

        # 无论是否原先有 viewBox，都给四周加 margin
        if vb is None:
            w_new = w + 2*margin
            h_new = h + 2*margin
            root.set('viewBox', f"{-margin:g} {-margin:g} {w_new:g} {h_new:g}")
        else:
            x0_new = x0 - margin
            y0_new = y0 - margin
            w_new  = (w if vb is None else w) + 2*margin
            h_new  = (h if vb is None else h) + 2*margin
            root.set('viewBox', f"{x0_new:g} {y0_new:g} {w_new:g} {h_new:g}")

        # width/height 也各 + 2*margin（保留单位）
        def _bump(attr):
            val = root.get(attr)
            if not val:
                return
            s = str(val)
            unit = ''
            for u in ('px','pt','mm','cm','in'):
                if s.endswith(u):
                    unit = u
                    s = s[:-len(u)]
                    break
            try:
                old = float(s)
            except Exception:
                return
            root.set(attr, f"{old + 2*margin}{unit}")
        _bump('width'); _bump('height')

        tree.write(str(svg_path), encoding='utf-8', xml_declaration=True)
    except Exception:
        # 失败不影响后续流程
        pass

def compose_panel(a: Path, b: Path, c: Path, d: Path,
                  out_svg: Path,
                  cell_w: float, cell_h: float,
                  pad_x: float, pad_y: float) -> None:
    """将四张 SVG 放到 2x2 网格，按各自等比缩放以适配 cell。"""
    # 计算总画布大小
    total_w = pad_x*3 + cell_w*2
    total_h = pad_y*3 + cell_h*2
    fig = SVGFigure(f"{total_w}", f"{total_h}")

    # 每个单元的左下角坐标（注意 svgutils 以左上角为 (0,0)，y 向下）
    # 我们统一按左上角定位
    origins = [
        (pad_x, pad_y),                       # A: 左上
        (pad_x*2 + cell_w, pad_y),            # B: 右上
        (pad_x, pad_y*2 + cell_h),            # C: 左下
        (pad_x*2 + cell_w, pad_y*2 + cell_h)  # D: 右下
    ]

    for svg_path, (ox, oy) in zip([a, b, c, d], origins):
        # 读尺寸，决定缩放
        w, h = _load_svg_get_size(svg_path)
        sx = cell_w / w
        sy = cell_h / h
        s = min(sx, sy)  # 等比缩放以完全放入 cell
        # 居中放置（在 cell 内）
        used_w = w * s
        used_h = h * s
        x = ox + (cell_w - used_w) / 2.0
        y = oy + (cell_h - used_h) / 2.0

        # 加载并移动/缩放
        fig_i = fromfile(str(svg_path))
        root = fig_i.getroot()  # Element
        root.moveto(x, y, scale_x=s, scale_y=s)
        fig.append([root])

    fig.save(str(out_svg))

def main():
    p = argparse.ArgumentParser(description="选择 4 张 SVG 组成 2x2 面板并导出 PDF。")
    p.add_argument("--a", type=Path, required=True, help="左上 SVG")
    p.add_argument("--b", type=Path, required=True, help="右上 SVG")
    p.add_argument("--c", type=Path, required=True, help="左下 SVG")
    p.add_argument("--d", type=Path, required=True, help="右下 SVG")
    p.add_argument("--out-svg", type=Path, default=Path("combine.svg"), help="中间 SVG 输出")
    p.add_argument("--out-pdf", type=Path, default=Path("combine.pdf"), help="最终 PDF 输出")
    p.add_argument("--cell-w", type=float, default=360.0, help="每格宽")
    p.add_argument("--cell-h", type=float, default=270.0, help="每格高")
    p.add_argument("--pad-x", type=float, default=30.0, help="格与格之间以及外侧的水平间距")
    p.add_argument("--pad-y", type=float, default=30.0, help="格与格之间以及外侧的垂直间距")
    p.add_argument("--margin", type=float, default=20.0, help="导出 PDF 前为 SVG 画布四周添加的页边距（pt）")
    args = p.parse_args()

    compose_panel(args.a, args.b, args.c, args.d,
                  out_svg=args.out_svg,
                  cell_w=args.cell_w, cell_h=args.cell_h,
                  pad_x=args.pad_x, pad_y=args.pad_y)

    # 加边距，避免任何一侧被裁切
    _expand_svg_canvas(args.out_svg, margin=args.margin)

    # 导出 PDF
    svg2pdf(url=str(args.out_svg), write_to=str(args.out_pdf))

if __name__ == "__main__":
    main()
