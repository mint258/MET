#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qm7_to_freesolv.py — 将 QM7 数据集的 .xyz 文件 转换成 FreeSolv 样式的标准 XYZ

QM7 原始格式示例（qm7_0000.xyz）：
-417.96
C   0.97780  -0.00250  -0.00440
H   2.09530  -0.00240   0.00410


转换后（qm7_0000.xyz）：
5
qm7 0 -417.96
C   0.97780  -0.00250  -0.00440
H   2.09530  -0.00240   0.00410

"""

import os
import glob
import argparse

def convert_qm7_file(in_path, out_path):
    # 读取所有非空行
    with open(in_path, 'r') as fr:
        lines = [l.strip() for l in fr if l.strip()]
    # 第一行是能量
    energy = lines[0]
    # 后面每行都是原子坐标
    atom_lines = lines[1:]
    n_atoms = len(atom_lines)

    # 从文件名提取 dataset 和 索引
    # 比如 qm7_0000.xyz -> dataset='qm7', idx_int=0
    name = os.path.splitext(os.path.basename(in_path))[0]
    parts = name.split('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        dataset, idx_str = parts
        idx_int = int(idx_str)
    else:
        dataset = name
        idx_int = ''

    # 生成 FreeSolv 风格的注释行
    comment = f"{dataset} {idx_int} {energy}"

    # 写出标准 XYZ
    with open(out_path, 'w') as fw:
        fw.write(f"{n_atoms}\n")
        fw.write(comment + "\n")
        for line in atom_lines:
            fw.write(line + "\n")

def main(input_dir, output_dir, pattern="qm7_*.xyz"):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        print(f"⚠️ 未找到匹配 {pattern} 的文件，请检查 input_dir 路径")
        return

    for fn in files:
        base = os.path.basename(fn)
        out_path = os.path.join(output_dir, base)
        try:
            convert_qm7_file(fn, out_path)
            print(f"✔ 转换成功：{base}")
        except Exception as e:
            print(f"✖ 转换失败：{base}，原因：{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将 QM7 数据集的 xyz 文件 转换为 FreeSolv 风格的标准 XYZ"
    )
    parser.add_argument(
        "--input_dir", "-i", required=True,
        help="QM7 xyz 文件所在目录（默认匹配 qm7_*.xyz）"
    )
    parser.add_argument(
        "--output_dir", "-o", required=True,
        help="转换后文件存放目录"
    )
    parser.add_argument(
        "--pattern", "-p", default="qm7_*.xyz",
        help="文件名匹配模式，默认 qm7_*.xyz"
    )
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.pattern)
