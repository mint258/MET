# dataset_without_charge.py
# -*- coding: utf-8 -*-

import os
from torch_geometric.data import Dataset, Data
import torch
import periodictable
from torch_cluster import radius_graph

class MoleculeDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, cutoff=2.5):
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        # 只读取 .xyz 文件
        self.files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.xyz')]
        self.cutoff = cutoff

    def len(self):
        return len(self.files)
    
    def get(self, idx):
        file_path = self.files[idx]
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # 第一行为原子数量
        try:
            atom_num = int(lines[0].strip())
        except Exception as e:
            raise ValueError(f"无法解析文件 {file_path} 第一行的原子数量: {e}")

        # 第二行通常是注释（忽略即可）
        # 第三行起，每行包含：原子符号 x y z [charge]（若有charge，则忽略，统一赋值为0）
        atom_types = []
        positions = []
        for line in lines[2:2 + atom_num]:
            parts = line.strip().split()
            if len(parts) < 4:
                raise ValueError(f"文件 {file_path} 中某行信息不足: {line}")
            atom_types.append(parts[0])
            try:
                pos = [float(part.replace('*^', 'e')) for part in parts[1:4]]
            except Exception as e:
                raise ValueError(f"解析文件 {file_path} 时无法将坐标转换为浮点数，行内容：{line} 错误：{e}")
            positions.append(pos)

        chiral_inchi = lines[-1].strip().split()
        
        # 将原子符号转换为索引
        element_dict = {element.symbol: idx for idx, element in enumerate(periodictable.elements)}
        z = torch.tensor([element_dict.get(atom, 0) for atom in atom_types], dtype=torch.long)
        pos = torch.tensor(positions, dtype=torch.float)
        # 预测时不需要真实电荷，使用0占位
        charge = torch.zeros((atom_num, 1), dtype=torch.float)
        
        # 使用 radius_graph 构建边（edge_index）
        edge_index = radius_graph(pos, r=self.cutoff, loop=False)
        data = Data(x=z.view(-1, 1), pos=pos, edge_index=edge_index, y=charge, chiral_inchi=chiral_inchi[0])
        data.filename = os.path.basename(file_path)
        # print(data.chiral_inchi)
        return data
