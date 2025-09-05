#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compose_embeddings_panel.py (d fixed to alignment_analysis.svg)
-------------------------------------------------------
Usage:
  python compose_embeddings_panel.py

What it does:
  - Runs:
      python embedding_plot_qm9.py --xyz_dir ../data/data1000 --model_path best_model_dim128_1.pth --device cuda --perplexity 12 --batch_size 50 --mw_plot --dipole_plot --binary_group_plots
      python embedding_plot.py     --xyz_dir new_xyz_structures/ --model_path best_model_dim128_1.pth --device cuda --perplexity 12 --batch_size 50
  - Picks SVGs as:
      (a) Dipole plot      → filename contains "dipole"
      (b) Carboxy binary   → filename contains "carbox"
      (c) Fluoro binary    → filename contains "fluor"
      (d) EXACTLY the file named "alignment_analysis.svg" (newest one if multiple)
  - Aligns the axes (plot frame) sizes across all four and composes a 2×2 panel labeled a/b/c/d.
  - Outputs panel.svg

Dependencies:
  pip install svgutils lxml
"""
import sys
import time
import subprocess
from pathlib import Path
from typing import Tuple, Optional, List
from lxml import etree
from svgutils.transform import fromfile, SVGFigure, TextElement
from cairosvg import svg2pdf

# --- Added helpers: optional per-SVG trim and safe canvas margin to avoid clipping ---
from lxml import etree

def _parse_len_unit(s: str):
    if s is None:
        return 0.0, ""
    s0 = str(s)
    unit = ""
    for u in ("px","pt","mm","cm","in"):
        if s0.endswith(u):
            unit = u; s0 = s0[:-len(u)]
            break
    try:
        return float(s0), unit
    except Exception:
        return 0.0, unit

def _trim_svg(in_path: Path, out_path: Path, trim_l: float, trim_t: float, trim_r: float, trim_b: float) -> Path:
    if all(v == 0 for v in (trim_l, trim_t, trim_r, trim_b)):
        return in_path
    parser = etree.XMLParser(remove_blank_text=False, recover=True)
    tree = etree.parse(str(in_path), parser); root = tree.getroot()
    if not root.tag.lower().endswith('svg'):
        return in_path
    # Get width/height
    w_val, w_unit = _parse_len_unit(root.get("width"))
    h_val, h_unit = _parse_len_unit(root.get("height"))
    vb = root.get("viewBox")
    if vb:
        parts = [p for p in vb.replace(",", " ").split() if p.strip()]
        if len(parts) >= 4:
            x0 = float(parts[0]); y0 = float(parts[1])
            w  = float(parts[2]); h  = float(parts[3])
        else:
            x0, y0, w, h = 0.0, 0.0, (w_val or 100.0), (h_val or 100.0)
    else:
        x0, y0 = 0.0, 0.0
        w = w_val or 100.0; h = h_val or 100.0

    # Apply crop but keep within bounds
    tl = max(0.0, trim_l); tt = max(0.0, trim_t)
    tr = max(0.0, trim_r); tb = max(0.0, trim_b)
    x0n = x0 + tl
    y0n = y0 + tt
    wn  = max(1.0, w - tl - tr)
    hn  = max(1.0, h - tt - tb)

    root.set("viewBox", f"{x0n:g} {y0n:g} {wn:g} {hn:g}")
    # Update width/height numerically to keep aspect consistent (preserve unit)
    if root.get("width") is not None:
        root.set("width", f"{wn}{w_unit}")
    if root.get("height") is not None:
        root.set("height", f"{hn}{h_unit}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_path), encoding="utf-8", xml_declaration=True)
    return out_path

def _expand_svg_canvas(svg_path: Path, margin: float = 20.0) -> None:
    try:
        parser = etree.XMLParser(remove_blank_text=False, recover=True)
        tree = etree.parse(str(svg_path), parser); root = tree.getroot()
        if not root.tag.lower().endswith('svg'):
            return
        vb = root.get('viewBox')
        w_val, w_unit = _parse_len_unit(root.get('width'))
        h_val, h_unit = _parse_len_unit(root.get('height'))
        if vb:
            parts = [p for p in vb.replace(',', ' ').split() if p.strip()]
            if len(parts) >= 4:
                x0 = float(parts[0]); y0 = float(parts[1])
                w  = float(parts[2]); h  = float(parts[3])
            else:
                x0, y0, w, h = 0.0, 0.0, (w_val or 100.0), (h_val or 100.0)
        else:
            x0, y0 = 0.0, 0.0
            w = w_val or 100.0; h = h_val or 100.0
        x0n = x0 - margin; y0n = y0 - margin
        wn = w + 2*margin;  hn = h + 2*margin
        root.set('viewBox', f"{x0n:g} {y0n:g} {wn:g} {hn:g}")
        if root.get('width') is not None:
            root.set('width', f"{(w_val or w) + 2*margin}{w_unit}")
        if root.get('height') is not None:
            root.set('height', f"{(h_val or h) + 2*margin}{h_unit}")
        tree.write(str(svg_path), encoding='utf-8', xml_declaration=True)
    except Exception:
        pass
# --- End helpers ---


# ---------- Commands ----------
CMD_QM9 = [
    "python", "embedding_plot_qm9.py",
    "--xyz_dir", "../../database/",
    "--model_path", "best_model_dim128_1.pth",
    "--device", "cuda",
    "--perplexity", "12",
    "--batch_size", "50",
    "--mw_plot",
    "--dipole_plot",
    "--binary_group_plots",
]

CMD_GENERAL = [
    "python", "embedding_plot.py",
    "--xyz_dir", "new_xyz_structures/",
    "--model_path", "best_model_dim128_1.pth",
    "--device", "cuda",
    "--perplexity", "12",
    "--batch_size", "50",
]

SVG_NS = "http://www.w3.org/2000/svg"
def _ns(tag: str) -> str:
    return "{%s}%s" % (SVG_NS, tag)

def run_cmd(cmd: list) -> None:
    print(">> Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def find_svgs_after(ts: float, root: Path = Path(".")) -> List[Path]:
    svgs = [p for p in root.rglob("*.svg") if p.stat().st_mtime >= ts]
    svgs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return svgs

def newest_alignment_svg(ts: float, roots: List[Path]) -> Optional[Path]:
    cand: List[Path] = []
    for r in roots:
        if not r.exists():
            continue
        for p in r.rglob("alignment_analysis.svg"):
            if p.stat().st_mtime >= ts:
                cand.append(p)
    if not cand:
        return None
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0]

def pick_by_keyword(candidates: List[Path], keyword: str) -> Optional[Path]:
    kw = keyword.lower()
    for p in candidates:
        if kw in p.name.lower():
            return p
    return None

def parse_svg_axes_rect(svg_path: Path):
    """Return (x, y, w, h) of the axes rectangle; (None,...) if not found."""
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

        def collect_rects(scope):
            cand = []
            for rect in scope.findall(".//%s" % _ns("rect")):
                try:
                    w = float(rect.attrib.get("width", "0") or "0")
                    h = float(rect.attrib.get("height", "0") or "0")
                    x = float(rect.attrib.get("x", "0") or "0")
                    y = float(rect.attrib.get("y", "0") or "0")
                except ValueError:
                    continue
                area = w * h
                if area > 0:
                    cand.append((area, x, y, w, h))
            return cand

        search_root = axes_group if axes_group is not None else root
        candidates = collect_rects(search_root)

        if not candidates:
            for clip in root.findall(".//%s" % _ns("clipPath")):
                rect = clip.find(".//%s" % _ns("rect"))
                if rect is not None:
                    try:
                        w = float(rect.attrib.get("width", "0") or "0")
                        h = float(rect.attrib.get("height", "0") or "0")
                        x = float(rect.attrib.get("x", "0") or "0")
                        y = float(rect.attrib.get("y", "0") or "0")
                    except ValueError:
                        continue
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
    # 同时写一个 viewBox，保证 CairoSVG 能识别坐标系
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
            root_elem.moveto(cx + padx, cy + pady)
            return
        x, y, _, _ = rect
        tx = cx + padx - scale * x
        ty = cy + pady - scale * y
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
    text_elems = [TextElement(str(x), str(y), lab, size=14, weight="bold")
                  for (x, y), lab in zip(label_pos, labels)]

    fig.append([rA, rB, rC, rD] + text_elems)
    fig.save(str(out_path))
    print(f"Saved panel to {out_path}")

def main():
    t0 = time.time()

    # 1) Run both plotting commands
    run_cmd(CMD_QM9)
    run_cmd(CMD_GENERAL)

    # 2) Find the three QM9 SVGs by keyword
    svgs = find_svgs_after(t0, Path("."))
    if not svgs:
        print("[ERROR] No SVGs found after running plotting commands.")
        sys.exit(1)

    a_path = pick_by_keyword(svgs, "dipole")
    b_path = pick_by_keyword(svgs, "carbox")
    c_path = pick_by_keyword(svgs, "fluor")

    missing = []
    if a_path is None: missing.append("dipole")
    if b_path is None: missing.append("carbox(y)")
    if c_path is None: missing.append("fluoro")
    if missing:
        print(f"[ERROR] Could not find expected qm9 SVG(s): {', '.join(missing)}")
        print("Found candidates:")
        for p in svgs:
            print("  -", p)
        sys.exit(1)

    # 3) Find (d) EXACTLY 'alignment_analysis.svg' (prefer newly created, search current dir and the provided xyz_dir)
    d_path = newest_alignment_svg(t0, [Path("."), Path("new_xyz_structures")])
    if d_path is None:
        print("[ERROR] Could not find 'alignment_analysis.svg' produced by embedding_plot.py.")
        sys.exit(1)

    print("Selected SVGs:")
    print(" a:", a_path)
    print(" b:", b_path)
    print(" c:", c_path)
    print(" d:", d_path)
    # Optional manual trims to equalize visual plot area (l,t,r,b) in px
    import os
    def _get_trim(name):
        v = os.environ.get(name)
        if not v:
            return (0.0,0.0,0.0,0.0)
        parts = [p.strip() for p in v.split(',')]
        if len(parts) != 4:
            return (0.0,0.0,0.0,0.0)
        try:
            return tuple(float(x) for x in parts)
        except Exception:
            return (0.0,0.0,0.0,0.0)
    trim_a = _get_trim('TRIM_A')
    trim_b = _get_trim('TRIM_B')
    trim_c = _get_trim('TRIM_C')
    trim_d = _get_trim('TRIM_D')
    # If any trim specified, create temporary cropped SVGs for composing
    tmp_dir = Path('.tmp_trimmed'); tmp_dir.mkdir(exist_ok=True)
    a_use = _trim_svg(a_path, tmp_dir/'a.svg', *trim_a)
    b_use = _trim_svg(b_path, tmp_dir/'b.svg', *trim_b)
    c_use = _trim_svg(c_path, tmp_dir/'c.svg', *trim_c)
    d_use = _trim_svg(d_path, tmp_dir/'d.svg', *trim_d)

    # 4) Compose final panel
    compose_panel(a_use, b_use, c_use, d_use,
                  out_path=Path("alignment.svg"),
                  target_axes_size=(360, 270),
                  pad=(30, 30))
        # Safe margin to avoid any side clipping (override via env MARGIN_PT)
    import os
    _expand_svg_canvas(Path("alignment.svg"), margin=float(os.environ.get('MARGIN_PT', '20')))
    svg2pdf(url="alignment.svg", write_to="alignment.pdf")
    
if __name__ == "__main__":
    main()
