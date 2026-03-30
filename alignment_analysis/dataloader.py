# dataset_without_charge.py
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from torch_geometric.data import Dataset, Data
import torch
import periodictable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from graph_compat import radius_graph

class MoleculeDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, cutoff=2.5):
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        self.files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.xyz')]
        self.cutoff = cutoff

    def len(self):
        return len(self.files)
    
    def get(self, idx):
        file_path = self.files[idx]
        with open(file_path, 'r') as f:
            lines = f.readlines()

        try:
            atom_num = int(lines[0].strip())
        except Exception as e:
            raise ValueError(f"Could not parse the atom count in the first line of {file_path}: {e}")

        atom_types = []
        positions = []
        for line in lines[2:2 + atom_num]:
            parts = line.strip().split()
            if len(parts) < 4:
                raise ValueError(f"Encountered an incomplete atom line in {file_path}: {line}")
            atom_types.append(parts[0])
            try:
                pos = [float(part.replace('*^', 'e')) for part in parts[1:4]]
            except Exception as e:
                raise ValueError(f"Could not convert coordinates to floats in {file_path}. Line: {line}. Error: {e}")
            positions.append(pos)

        chiral_inchi = lines[-1].strip().split()
        
        element_dict = {element.symbol: idx for idx, element in enumerate(periodictable.elements)}
        z = torch.tensor([element_dict.get(atom, 0) for atom in atom_types], dtype=torch.long)
        pos = torch.tensor(positions, dtype=torch.float)
        charge = torch.zeros((atom_num, 1), dtype=torch.float)
        
        edge_index = radius_graph(pos, r=self.cutoff, loop=False)
        data = Data(x=z.view(-1, 1), pos=pos, edge_index=edge_index, y=charge, chiral_inchi=chiral_inchi[0])
        data.filename = os.path.basename(file_path)
        # print(data.chiral_inchi)
        return data
