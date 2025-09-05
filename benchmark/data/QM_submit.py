#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QM_submit.py  ‒  批量几何优化（MOPAC PM7）

功能
-----
1. 读取 *.xyz 文件（格式：第1行=原子数；第2行=注释；接着 N 行原子坐标；
   文件末尾可选一行 SMILES）。
2. 生成带 “PM7 XYZ” 关键字的 MOPAC 输入文件，执行几何优化。
3. 解析 MOPAC 生成的 <name>.xyz 文件，取**最后一帧**坐标作为优化结果。
4. 输出新的 *.opt.xyz：
   - 第1、2行完整照抄原文件；
   - 原子行只含元素和三维坐标（不写电荷/能量等）；
   - 若原文件末尾有 SMILES，则原样写回。

用法示例
---------
python QM_submit.py --xyz_dir ./input_xyz --output_dir ./opt_xyz
"""

import os
import glob
import argparse
import subprocess
import time

# ────────────────────────── I/O 工具 ──────────────────────────
def read_xyz(path):
    """读取单个 XYZ，返回各组成部分。"""
    with open(path, "r") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]  # 去空行
    n_atoms = int(lines[0])
    comment = lines[1]
    atom_lines = lines[2 : 2 + n_atoms]
    smiles = lines[-1] if len(lines) > 2 + n_atoms else ""

    atoms, coords = [], []
    for line in atom_lines:
        parts = line.split()
        atoms.append(parts[0])
        coords.append(tuple(map(float, parts[1:4])))
    return n_atoms, comment, atoms, coords, smiles


def create_mop_input(xyz_path, atoms, coords, out_dir):
    """生成 MOPAC 输入文件，首行关键字含 XYZ 以便输出坐标。"""
    base = os.path.splitext(os.path.basename(xyz_path))[0]
    inp_file = os.path.join(out_dir, f"{base}.mop")

    header = "PM7 XYZ\n\n"  # PM7 几何优化并输出 .xyz
    title = f"{base} - geometry optimisation\n"

    with open(inp_file, "w") as f:
        f.write(header)
        f.write(title)
        for (x, y, z), a in zip(coords, atoms):
            f.write(f"{a}   {x:.6f}   {y:.6f}   {z:.6f}\n")
    return inp_file, base


def run_mopac(base, cwd):
    """调用 MOPAC，阻塞直到结束。"""
    subprocess.run(["mopac", base], cwd=cwd, check=True)


def parse_final_coords(xyz_file, n_atoms):
    """
    解析 MOPAC 生成的 <name>.xyz，取最后一帧坐标。
    该文件包含若干帧，每帧格式仍是标准 XYZ。
    """
    with open(xyz_file, "r") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    frames = []
    i = 0
    while i < len(lines):
        nat = int(lines[i])
        frame_comment = lines[i + 1]  # 可忽略
        frame_atoms = lines[i + 2 : i + 2 + nat]
        frames.append(frame_atoms)
        i += 2 + nat

    final_atoms = frames[-1]
    coords = []
    for line in final_atoms:
        parts = line.split()
        coords.append(tuple(map(float, parts[1:4])))
    if len(coords) != n_atoms:
        raise ValueError("原子数与优化结果不一致，可能优化失败。")
    return coords


def write_opt_xyz(out_path, n_atoms, comment, atoms, coords, smiles):
    """按要求写出优化后 XYZ。"""
    with open(out_path, "w") as f:
        f.write(f"{n_atoms}\n")
        f.write(comment + "\n")
        for a, (x, y, z) in zip(atoms, coords):
            f.write(f"{a:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")
        if smiles:
            f.write(smiles + "\n")


# ────────────────────────── 主流程 ──────────────────────────
def main(xyz_dir, output_dir):
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)

    xyz_files = glob.glob(os.path.join(xyz_dir, "*.xyz"))
    print(f"在 {xyz_dir} 找到 {len(xyz_files)} 个 XYZ 文件。")

    ok, fail = 0, 0
    for xyz_path in xyz_files:
        try:
            n_atoms, comment, atoms, coords, smiles = read_xyz(xyz_path)
            mop_inp, base = create_mop_input(xyz_path, atoms, coords, output_dir)
            print('mopac input name:', mop_inp)
            run_mopac(base, output_dir)

            mop_xyz = os.path.join(output_dir, f"{base}.xyz")
            opt_coords = parse_final_coords(mop_xyz, n_atoms)

            out_xyz = os.path.join(output_dir, f"{base}.xyz")
            write_opt_xyz(out_xyz, n_atoms, comment, atoms, opt_coords, smiles)

            print(f"[OK ] {base} → {base}.opt.xyz")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {os.path.basename(xyz_path)}  原因: {e}")
            fail += 1

    print(f"完成：成功 {ok}，失败 {fail}，耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="批量使用 MOPAC PM7 对 XYZ 文件进行几何优化（保留原文件前两行，不写能量/电荷）。"
    )
    parser.add_argument("--xyz_dir", required=True, help="待优化 XYZ 目录")
    parser.add_argument("--output_dir", required=True, help="结果输出目录")
    args = parser.parse_args()

    main(args.xyz_dir, args.output_dir)
