# sdf_to_xyz_converter.py
# -*- coding: utf-8 -*-

import os
import subprocess
import argparse

def convert_sdf_to_xyz(sdf_path, output_dir, dataset_name):
    """
    使用 Open Babel 的命令行工具将 SDF 文件转换为多个 XYZ 文件。

    参数:
    - sdf_path (str): 输入的 SDF 文件路径。
    - output_dir (str): 输出的 XYZ 文件目录。
    - dataset_name (str): 数据集名称，用于文件前缀（如 'qm7' 或 'qm8'）。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 构建输出文件路径， 'qm7_.xyz' 会被 obabel 替换为 'qm7_1.xyz' 等
    output_file = os.path.join(output_dir, f"{dataset_name}_.xyz")

    # 构建 obabel 命令
    command = [
        'obabel',
        sdf_path,
        '-O', output_file,
        '-m'  # Split output into multiple files
    ]

    try:
        # 执行命令
        subprocess.run(command, check=True)
        print(f"成功将 {sdf_path} 转换为 XYZ 文件，保存在 {output_dir} 目录中。")
    except subprocess.CalledProcessError as e:
        print(f"转换失败: {e}")
    except FileNotFoundError:
        print("无法找到 obabel 命令，请确保 Open Babel 已正确安装并添加到系统 PATH 中。")

def main():
    parser = argparse.ArgumentParser(description="使用 Open Babel 将 SDF 文件转换为多个 XYZ 文件。")
    parser.add_argument('--qm7_sdf', type=str, required=True, help="QM7 数据集的输入 SDF 文件路径。")
    parser.add_argument('--qm8_sdf', type=str, required=True, help="QM8 数据集的输入 SDF 文件路径。")
    parser.add_argument('--freesolv_sdf', type=str, required=False, help="FreeSolv 数据集的输入 SDF 文件路径。")
    parser.add_argument('--output_dir', type=str, default='xyz_output', help="输出的 XYZ 文件根目录。")

    args = parser.parse_args()

    # 转换 QM7 数据集
    if args.qm7_sdf:
        qm7_output_dir = os.path.join(args.output_dir, 'qm7_xyz')
        print(f"正在转换 QM7 数据集: {args.qm7_sdf} 到 {qm7_output_dir}")
        convert_sdf_to_xyz(sdf_path=args.qm7_sdf, output_dir=qm7_output_dir, dataset_name='qm7')

    # 转换 QM8 数据集
    if args.qm8_sdf:
        qm8_output_dir = os.path.join(args.output_dir, 'qm8_xyz')
        print(f"正在转换 QM8 数据集: {args.qm8_sdf} 到 {qm8_output_dir}")
        convert_sdf_to_xyz(sdf_path=args.qm8_sdf, output_dir=qm8_output_dir, dataset_name='qm8')

    # 转换 FreeSolv 数据集
    if args.freesolv_sdf:
        freesolv_output_dir = os.path.join(args.output_dir, 'freesolv_xyz')
        print(f"正在转换 FreeSolv 数据集: {args.freesolv_sdf} 到 {freesolv_output_dir}")
        convert_sdf_to_xyz(sdf_path=args.freesolv_sdf, output_dir=freesolv_output_dir, dataset_name='freesolv')

if __name__ == "__main__":
    main()
