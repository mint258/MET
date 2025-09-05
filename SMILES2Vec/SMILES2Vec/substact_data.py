# -*- coding: utf-8 -*-

import os
import json
import pysmiles
import networkx as nx
import re
import argparse

def extract_smiles_from_xyz(file_path):
    """
    从XYZ文件中提取第一个SMILES字符串。

    参数:
        file_path (str): XYZ文件的路径。

    返回:
        str 或 None: 提取到的SMILES字符串，或在失败时返回None。
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 3:
            print(f"文件 {file_path} 太短，无法包含SMILES。")
            return None
        
        try:
            num_atoms = int(lines[0].strip())
        except ValueError:
            print(f"文件 {file_path} 的第一行不是整数。")
            return None
        
        # 确保文件有足够的行数
        if len(lines) < 2 + num_atoms + 1:
            print(f"文件 {file_path} 没有足够的行来包含原子信息和SMILES。")
            return None
        
        # 读取原子行（此处不使用）
        atom_lines = lines[2:2 + num_atoms]
        
        # 读取包含SMILES的行
        smiles_line = lines[3 + num_atoms].strip()
        # 使用正则表达式按空白字符分割
        smiles_candidates = re.split(r'\s+', smiles_line)
        if len(smiles_candidates) == 0:
            print(f"文件 {file_path} 中未找到SMILES。")
            return None
        smiles = smiles_candidates[0]  # 取第一个SMILES字符串
        return smiles
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def convert_smiles_to_graph(smiles):
    """
    将SMILES字符串转换为图结构。

    参数:
        smiles (str): SMILES字符串。

    返回:
        dict 或 None: 包含节点和边信息的字典，或在解析失败时返回None。
    """
    try:
        # 使用pysmiles解析SMILES
        mol_graph = pysmiles.read_smiles(smiles)
    except Exception as e:
        print(f"无法解析SMILES: {smiles}, 错误: {e}")
        return None
    
    nodes = []
    # 为节点分配0-based索引
    node_mapping = {node: idx for idx, node in enumerate(mol_graph.nodes())}
    for node, data in mol_graph.nodes(data=True):
        node_info = {
            "element": data.get('element', 'Unknown'),
            "charge": data.get('charge', 0),
            "aromatic": data.get('aromatic', False),
            "hcount": data.get('hcount', 0),
        }
        # 处理立体化学信息（如果存在）
        stereo = data.get('stereo', None)
        if stereo is not None:
            node_info['stereo'] = stereo
        nodes.append(node_info)
    
    edges = []
    for u, v, data in mol_graph.edges(data=True):
        edge_info = {
            "source": node_mapping[u],
            "target": node_mapping[v],
            "order": data.get('order', 1)
        }
        edges.append(edge_info)
    
    return {"nodes": nodes, "edges": edges}

def process_xyz_files(folder_path, output_json_path):
    """
    处理文件夹下所有XYZ文件，提取SMILES，转换为图结构，并保存到JSON文件。

    参数:
        folder_path (str): 包含XYZ文件的文件夹路径。
        output_json_path (str): 输出JSON文件的路径。
    """
    molecules = []
    total_files = 0
    processed_files = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.xyz'):
                total_files += 1
                file_path = os.path.join(root, file)
                smiles = extract_smiles_from_xyz(file_path)
                if smiles:
                    graph = convert_smiles_to_graph(smiles)
                    if graph:
                        molecules.append(graph)
                        processed_files += 1
    # 保存到JSON
    with open(output_json_path, 'w') as f:
        json.dump(molecules, f, indent=4)
    print(f"处理了 {processed_files} 个XYZ文件，共计 {total_files} 个文件。")
    print(f"已将 {len(molecules)} 个分子保存到 '{output_json_path}'。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从XYZ文件中提取SMILES，转换为图结构，并保存为JSON。')
    parser.add_argument('--input_folder', type=str, required=True, help='包含XYZ文件的文件夹路径。')
    parser.add_argument('--output_json', type=str, default='molecules.json', help='输出JSON文件的路径。')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_folder):
        print(f"输入文件夹 '{args.input_folder}' 不存在或不是一个目录。")
        exit(1)
    
    process_xyz_files(args.input_folder, args.output_json)
