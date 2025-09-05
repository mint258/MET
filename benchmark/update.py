# update_qm7_xyz_labels.py
# -*- coding: utf-8 -*-

import os
import csv
import re
import argparse

def load_labels(csv_path):
    """
    从 CSV 文件中加载标签。
    
    参数:
    - csv_path (str): CSV 文件路径。
    
    返回:
    - dict: 文件名到标签的映射字典。
    """
    labels = {}
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            row = row[0].split(',')
            if len(row) != 2:
                print(f"警告: 无效的行格式: {row}")
                continue
            filename, label = row
            labels[filename] = label
    return labels

def extract_number(filename):
    """
    从文件名中提取数字部分。
    
    参数:
    - filename (str): 文件名，例如 'qm7_0000.xyz'
    
    返回:
    - str: 数字部分，例如 '0000'
    """
    match = re.match(r"qm7_(\d+)\.xyz", filename)
    if match:
        return match.group(1)
    else:
        print(f"警告: 文件名 {filename} 不符合预期格式。")
        return None

def update_xyz_file(xyz_path, label):
    """
    更新 XYZ 文件的第二行，添加数字部分和目标值。
    
    参数:
    - xyz_path (str): XYZ 文件路径。
    - label (str): 目标值。
    """
    filename = os.path.basename(xyz_path)
    number = extract_number(filename)
    if number is None:
        print(f"警告: 无法提取数字，跳过文件 {xyz_path}")
        return
    
    # 读取文件内容
    with open(xyz_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        print(f"警告: 文件 {xyz_path} 格式不正确，跳过。")
        return
    
    # 原始第二行内容，例如 'gdb7k_0000.xyz'
    original_second_line = lines[1].strip()
    
    # 构建新的第二行内容
    new_second_line = f"{original_second_line} {number} {label}"
    
    # 更新第二行
    lines[1] = new_second_line + '\n'
    
    # 写回文件
    with open(xyz_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"已更新文件: {xyz_path}")

def rename_xyz_file(xyz_path, output_dir):
    """
    重命名 XYZ 文件为 'qm7_*.xyz' 格式。
    
    参数:
    - xyz_path (str): XYZ 文件路径。
    - output_dir (str): 重命名后的文件保存目录。
    """
    filename = os.path.basename(xyz_path)
    match = re.match(r"qm7_(\d+)\.xyz", filename)
    if match:
        number = match.group(1)
        new_filename = f"qm7_{number}.xyz"
        new_path = os.path.join(output_dir, new_filename)
        os.rename(xyz_path, new_path)
        print(f"已重命名文件: {xyz_path} -> {new_path}")
    else:
        print(f"警告: 文件名 {filename} 不符合重命名规则，跳过。")

def main():
    parser = argparse.ArgumentParser(description="更新 QM7 XYZ 文件的目标标签，并重命名文件。")
    parser.add_argument('--xyz_dir', type=str, required=True, help="包含 QM7 XYZ 文件的目录路径。")
    parser.add_argument('--csv_file', type=str, required=True, help="包含标签的 CSV 文件路径。")
    parser.add_argument('--output_dir', type=str, required=False, default=None, help="重命名后的文件保存目录（可选）。如果不指定，将在原目录下重命名。")
    
    args = parser.parse_args()
    
    xyz_dir = args.xyz_dir
    csv_file = args.csv_file
    output_dir = args.output_dir if args.output_dir else xyz_dir
    
    # 检查目录和文件是否存在
    if not os.path.isdir(xyz_dir):
        print(f"错误: 指定的 XYZ 目录 {xyz_dir} 不存在。")
        return
    if not os.path.isfile(csv_file):
        print(f"错误: 指定的 CSV 文件 {csv_file} 不存在。")
        return
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"已创建输出目录: {output_dir}")
    
    # 加载标签
    labels = load_labels(csv_file)
    print(f"已加载 {len(labels)} 个标签。")
    
    # 遍历 XYZ 文件并更新
    for root, dirs, files in os.walk(xyz_dir):
        for file in files:
            if file.endswith('.xyz'):
                xyz_path = os.path.join(root, file)
                if file in labels:
                    label = labels[file]
                    update_xyz_file(xyz_path, label)
                    # 重命名文件
                    if output_dir != xyz_dir:
                        rename_xyz_file(xyz_path, output_dir)
                else:
                    print(f"警告: 文件 {file} 在 CSV 中没有对应的标签，跳过。")
    
    print("所有文件处理完成。")

if __name__ == "__main__":
    main()
