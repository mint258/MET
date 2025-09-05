# rename_xyz_files.py
# -*- coding: utf-8 -*-

import os
import re

def rename_xyz_files(xyz_dir):
    """
    遍历目标文件夹中的每一个 .xyz 文件，提取第二行的数字，去掉前面的数字，
    并重新命名文件为 'qm7_*.xyz' 格式。
    
    参数:
    - xyz_dir (str): XYZ 文件所在的目录路径。
    """
    # 遍历目录中的所有 .xyz 文件
    for root, dirs, files in os.walk(xyz_dir):
        for file in files:
            if file.endswith('.xyz'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        if len(lines) < 2:
                            print(f"警告: 文件 {file_path} 格式不正确，跳过。")
                            continue
                        
                        # 第二行的内容形如 gdb7k_????.xyz
                        second_line = lines[1].strip()
                        
                        # 使用正则表达式提取 'gdb7k_' 后面的数字
                        match = re.search(r"gdb7k_(\d+)\.xyz", second_line)
                        if match:
                            number = match.group(1)  # 提取数字部分
                            new_filename = f"qm7_{number}.xyz"
                            new_file_path = os.path.join(root, new_filename)
                            
                            # 重命名文件
                            os.rename(file_path, new_file_path)
                            print(f"已重命名: {file_path} -> {new_file_path}")
                        else:
                            print(f"警告: 文件 {file_path} 的第二行无法提取数字，跳过。")
                except Exception as e:
                    print(f"错误: 无法处理文件 {file_path}. 错误信息: {e}")

def main():
    # 输入目标文件夹路径
    xyz_dir = input("请输入目标文件夹路径（包含所有 xyz 文件）：").strip()
    
    if os.path.exists(xyz_dir) and os.path.isdir(xyz_dir):
        rename_xyz_files(xyz_dir)
    else:
        print("错误: 输入的路径无效，请检查路径是否正确。")

if __name__ == "__main__":
    main()
