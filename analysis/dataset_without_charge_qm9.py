# dataset_without_charge.py
# -*- coding: utf-8 -*-

import os
from torch_geometric.data import Dataset, Data
import torch
import periodictable
from rdkit import Chem
from torch_cluster import radius_graph

class MoleculeDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, cutoff=2.5):
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        # 获取所有的 .xyz 文件路径
        self.files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.xyz')]
        
        # 定义所有的标量性质名称，确保顺序与文件中的顺序一致
        self.all_properties = [
            "rot_A", "rot_B", "rot_C", "dipole", "polarizability",
            "HOMO_energy", "LUMO_energy", "gap", "R2", "zpve",
            "U0", "U298", "H298", "G298", "Cv"
            # 如有更多性质，请继续添加
        ]
        
        # 创建性质到索引的映射
        self.property_to_index = {prop: idx for idx, prop in enumerate(self.all_properties)}
        
        # 设置半径阈值
        self.cutoff = cutoff

    def len(self):
        return len(self.files)
    
    def get(self, idx):
        file_path = self.files[idx]
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # 第一行：原子数量
        atom_num = int(lines[0].strip())
        
        # 第二行：标量性质
        # 假设第二行的第一个元素是原子数量，其余为标量性质
        scalar_props = list(map(float, [part.replace('*^', 'e') for part in lines[1].strip().split()[2:]]))
        
        # 第3行到第 n_a+2 行：原子信息
        atom_lines = lines[2:2 + atom_num]
        atom_types = []
        positions = []
        charges = []
        for line in atom_lines:
            parts = line.strip().split()
            atom_types.append(parts[0])
            pos = [float(part.replace('*^', 'e')) for part in parts[1:4]]
            positions.append(pos)
            if len(parts) >= 5:
                charge = float(parts[4].replace('*^', 'e'))
            else:
                charge = 0.0  # 或者其他默认值
            charges.append(charge)

        # 将原子类型转换为索引
        element_dict = {element.symbol: idx for idx, element in enumerate(periodictable.elements)}
        z = torch.tensor([element_dict.get(atom, 0) for atom in atom_types], dtype=torch.long)

        pos = torch.tensor(positions, dtype=torch.float)
        charge = torch.tensor(charges, dtype=torch.float).view(-1, 1)  # 确保形状为 [num_nodes, 1]
        
        # 构建边（edge_index）使用 radius_graph
        edge_index = radius_graph(pos, r=self.cutoff, loop=False)  # [2, num_edges]
        
        # 创建 Data 对象
        data = Data(x=z.view(-1, 1), pos=pos, edge_index=edge_index, y=charge)

        # 添加文件名属性
        data.filename = os.path.basename(file_path)

        # 添加标量性质
        # 确保标量性质的数量与 all_properties 一致
        if len(scalar_props) != len(self.all_properties):
            raise ValueError(f"文件 {file_path} 的标量性质数量与预定义的性质列表不一致。")
        
        scalar_props = torch.tensor(scalar_props, dtype=torch.float)
        data.scalar_props = scalar_props  # 形状为 [num_properties]
        
        # 解析手性信息
        # 假设最后一行包含两个 InChI 字符串，第二个为手性 InChI
        last_line = lines[-1].strip().split()
        if len(last_line) >= 2:
            chiral_inchi = last_line[1]
            data.chiral_inchi = chiral_inchi
        
        return data
