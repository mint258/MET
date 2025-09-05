#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compose_panel.py (fixed)
- Avoid parsing 'transform' attribute strings; pass scale explicitly.
- Safe fallback when axes rect is missing: place graphic at (padx, pady) of its cell.
"""
import os
import sys
import subprocess
from pathlib import Path
from typing import Tuple, Optional

from lxml import etree
from svgutils.transform import fromfile, SVGFigure, TextElement
from cairosvg import svg2pdf

def _expand_svg_canvas(svg_path: Path, margin: float = 20.0) -> None:
    """在不改变图形元素的前提下，扩大根 <svg> 的 viewBox 与 width/height，
    等价于四周加页边距，避免左侧被裁切。"""
    try:
        parser = etree.XMLParser(remove_blank_text=False, recover=True)
        tree = etree.parse(str(svg_path), parser)
        root = tree.getroot()
        # 命名空间兼容
        if not root.tag.lower().endswith('svg'):
            return

        # 解析 viewBox
        vb = root.get('viewBox')
        if vb is None:
            # 如果没有 viewBox，则根据 width/height 推一个
            w_attr = root.get('width')
            h_attr = root.get('height')
            def _to_float(v):
                if v is None:
                    return None
                try:
                    return float(str(v).replace('px','').replace('pt',''))
                except Exception:
                    return None
            w = _to_float(w_attr)
            h = _to_float(h_attr)
            if w is None or h is None:
                return  # 无法安全处理则放弃修改
            vb_vals = [0.0, 0.0, float(w), float(h)]
        else:
            vb_vals = [float(x) for x in vb.replace(',', ' ').split() if x.strip()][:4]
            if len(vb_vals) != 4:
                return

        x0, y0, w, h = vb_vals
        # 在四周各加 margin
        x0_new = x0 - margin
        y0_new = y0 - margin
        w_new = w + 2*margin
        h_new = h + 2*margin

        root.set('viewBox', f"{x0_new:g} {y0_new:g} {w_new:g} {h_new:g}")

        # 同步尝试调整 width/height 数值（保留原单位）
        def _adjust_with_unit(attr_name):
            v = root.get(attr_name)
            if v is None:
                return
            s = str(v)
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
            root.set(attr_name, f"{old + 2*margin}{unit}")
        _adjust_with_unit('width')
        _adjust_with_unit('height')

        tree.write(str(svg_path), encoding='utf-8', xml_declaration=True)
    except Exception:
        # 安全失败：不影响后续流程
        pass


# ---------- User-provided commands (as per your message) ----------
CMD_A = [
    "python", "GNN_EGNN_comparison.py",
    "--checkpoint_a", "best_gnn_transformer_model_1.pth",
    "--checkpoint_b", "best_model_dim128_1.pth",
    "--model_a", "smiles2vec", "--model_b", "comenet_charge",
    "--test_data_root", "../../data/data_valid_test/",
    "--batch_size", "64",
    "--plot_path", "charge_compare.svg",
]
OUT_A = Path("charge_compare.svg")  # expected output

CMD_B = [
    "python", "finetune_vs_direct.py",
    "--log-dir", "finetune_direct_test/",
    "--out", "finetune_vs_direct_r2.svg",
]
OUT_B = Path("finetune_vs_direct_r2.svg")  # expected output

CMD_C = [
    "python", "property_predict_qm7.py",
    "--checkpoint_a", "best_qm7_layer_4_data500.pth",
    "--checkpoint_b", "best_qm7_layer_0_data_500.pth",
    "--test_data_root", "../../benchmark/data/qm7_divide/test/",
    "--target_property", "rot_A",
    "--plot_path", "qm7.svg",
]
OUT_C = Path("qm7.svg")  # expected output

CMD_D = [
    "python", "atom_embedding_dim.py",
    "--root", "atom_dim_test/"
]
ROOT_D = Path("atom_dim_test")

SVG_NS = "http://www.w3.org/2000/svg"
def _ns(tag: str) -> str:
    return "{%s}%s" % (SVG_NS, tag)

def run_cmd(cmd: list, cwd: Optional[Path] = None) -> None:
    print(">> Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)

def find_latest_svg_in(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    svgs = list(root.rglob("*.svg"))
    if not svgs:
        return None
    svgs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return svgs[0]

def parse_svg_axes_rect(svg_path: Path):
    """
    Heuristic: find the first axes group and read its background/clip rect.
    Returns (x, y, w, h) or (None, None, None, None) if not found.
    """
    try:
        tree = etree.parse(str(svg_path))
        root = tree.getroot()

        axes_groups = root.findall(".//%s" % _ns("g"))
        axes_group = None
        for g in axes_groups:
            gid = g.attrib.get("id", "")
            if gid.startswith("axes_"):
                axes_group = g
                break

        candidates = []
        search_root = axes_group if axes_group is not None else root
        for rect in search_root.findall(".//%s" % _ns("rect")):
            w = float(rect.attrib.get("width", "0") or "0")
            h = float(rect.attrib.get("height", "0") or "0")
            x = float(rect.attrib.get("x", "0") or "0")
            y = float(rect.attrib.get("y", "0") or "0")
            area = w * h
            if area > 0:
                candidates.append((area, x, y, w, h))

        if not candidates:
            for clip in root.findall(".//%s" % _ns("clipPath")):
                rect = clip.find(".//%s" % _ns("rect"))
                if rect is not None:
                    w = float(rect.attrib.get("width", "0") or "0")
                    h = float(rect.attrib.get("height", "0") or "0")
                    x = float(rect.attrib.get("x", "0") or "0")
                    y = float(rect.attrib.get("y", "0") or "0")
                    area = w * h
                    if area > 0:
                        candidates.append((area, x, y, w, h))

        if not candidates:
            return (None, None, None, None)

        candidates.sort(reverse=True, key=lambda t: t[0])
        _, x, y, w, h = candidates[0]
        return (x, y, w, h)
    except Exception as e:
        print(f"[WARN] Failed to parse axes rect from {svg_path}: {e}")
        return (None, None, None, None)

def compose_panel(a_path: Path, b_path: Path, c_path: Path, d_path: Path,
                  out_path: Path = Path("panel.svg"),
                  target_axes_size: Tuple[int, int] = (360, 270),
                  pad: Tuple[int, int] = (30, 30)) -> None:
    """
    Read four SVGs, scale each so its axes rectangle width/height matches `target_axes_size`,
    then place them into a 2x2 layout with labels a/b/c/d.
    """
    Wt, Ht = target_axes_size
    padx, pady = pad

    fA = fromfile(str(a_path)); rA = fA.getroot()
    fB = fromfile(str(b_path)); rB = fB.getroot()
    fC = fromfile(str(c_path)); rC = fC.getroot()
    fD = fromfile(str(d_path)); rD = fD.getroot()

    rectA = parse_svg_axes_rect(a_path)
    rectB = parse_svg_axes_rect(b_path)
    rectC = parse_svg_axes_rect(c_path)
    rectD = parse_svg_axes_rect(d_path)

    def scale_root(root_elem, rect):
        x, y, w, h = rect
        if None in rect or w == 0 or h == 0:
            return 1.0
        sx = Wt / w
        sy = Ht / h
        s = min(sx, sy)
        root_elem.scale(s)
        return s

    sA = scale_root(rA, rectA)
    sB = scale_root(rB, rectB)
    sC = scale_root(rC, rectC)
    sD = scale_root(rD, rectD)

    Wc = Wt + 2*padx
    Hc = Ht + 2*pady
    total_w = 2*Wc
    total_h = 2*Hc
    fig = SVGFigure(f"{total_w}px", f"{total_h}px")
    # Í¬Ê±Ð´Ò»¸ö viewBox£¬±£Ö¤ CairoSVG ÄÜÊ¶±ð×ø±êÏµ
    fig.root.attrib['viewBox'] = f"0 0 {total_w} {total_h}"

    positions = [
        (0, 0),       # a
        (Wc, 0),      # b
        (0, Hc),      # c
        (Wc, Hc),     # d
    ]

    def move_to_cell(root_elem, rect, cell_xy, scale):
        cx, cy = cell_xy
        if rect[0] is None:
            # Fallback: if we couldn't detect axes rect, just place the figure
            root_elem.moveto(cx + padx, cy + pady)
            return
        x, y, _, _ = rect
        current_x = scale * x
        current_y = scale * y
        tx = cx + padx - current_x
        ty = cy + pady - current_y
        root_elem.moveto(tx, ty)

    move_to_cell(rA, rectA, positions[0], sA)
    move_to_cell(rB, rectB, positions[1], sB)
    move_to_cell(rC, rectC, positions[2], sC)
    move_to_cell(rD, rectD, positions[3], sD)

    labels = ['a', 'b', 'c', 'd']
    label_pos = [
        (positions[0][0] + padx + 4, positions[0][1] + pady + 16),
        (positions[1][0] + padx + 4, positions[1][1] + pady + 16),
        (positions[2][0] + padx + 4, positions[2][1] + pady + 16),
        (positions[3][0] + padx + 4, positions[3][1] + pady + 16),
    ]
    text_elems = [
        TextElement(str(x), str(y), lab, size=14, weight="bold")
        for (x, y), lab in zip(label_pos, labels)
    ]

    fig.append([rA, rB, rC, rD] + text_elems)
    fig.save(str(out_path))
    print(f"Saved panel to {out_path}")

def main():
    # 1) Run plot commands
    run_cmd(CMD_A)
    run_cmd(CMD_B)
    run_cmd(CMD_C)
    run_cmd(CMD_D)

    # 2) Check outputs
    if not OUT_A.exists():
        print(f"[ERROR] Expected {OUT_A} not found."); sys.exit(1)
    if not OUT_B.exists():
        print(f("[ERROR] Expected {OUT_B} not found.")); sys.exit(1)
    if not OUT_C.exists():
        print(f"[ERROR] Expected {OUT_C} not found."); sys.exit(1)

    out_d = find_latest_svg_in(ROOT_D) or find_latest_svg_in(Path("."))
    if out_d is None:
        print("[ERROR] Could not find the SVG output from atom_embedding_dim.py."); sys.exit(1)
    print(f"Detected D SVG: {out_d}")

    compose_panel(OUT_A, OUT_B, OUT_C, out_d,
                  out_path=Path("combine.svg"),
                  target_axes_size=(360, 270),
                  pad=(30, 30))
    _expand_svg_canvas(Path("combine.svg"), margin=20.0)  # 扩大画布避免左侧裁切
    svg2pdf(url="combine.svg", write_to="combine.pdf")
if __name__ == "__main__":
    main()
