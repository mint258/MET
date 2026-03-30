# dataset_without_charge.py
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from torch_geometric.data import Dataset, Data
import torch
import periodictable
from rdkit import Chem

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from graph_compat import radius_graph

class MoleculeDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, cutoff=2.5):
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        self.files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.xyz')]
        
        self.all_properties = [
            "rot_A", "rot_B", "rot_C", "dipole", "polarizability",
            "HOMO_energy", "LUMO_energy", "gap", "R2", "zpve",
            "U0", "U298", "H298", "G298", "Cv"
        ]
        
        self.property_to_index = {prop: idx for idx, prop in enumerate(self.all_properties)}
        
        self.cutoff = cutoff

    def len(self):
        return len(self.files)
    
    def get(self, idx):
        file_path = self.files[idx]
        with open(file_path, 'r') as f:
            lines = f.readlines()

        atom_num = int(lines[0].strip())
        
        scalar_props = list(map(float, [part.replace('*^', 'e') for part in lines[1].strip().split()[2:]]))
        
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
                charge = 0.0
            charges.append(charge)

        element_dict = {element.symbol: idx for idx, element in enumerate(periodictable.elements)}
        z = torch.tensor([element_dict.get(atom, 0) for atom in atom_types], dtype=torch.long)

        pos = torch.tensor(positions, dtype=torch.float)
        charge = torch.tensor(charges, dtype=torch.float).view(-1, 1)
        
        edge_index = radius_graph(pos, r=self.cutoff, loop=False)  # [2, num_edges]
        
        data = Data(x=z.view(-1, 1), pos=pos, edge_index=edge_index, y=charge)

        data.filename = os.path.basename(file_path)

        if len(scalar_props) != len(self.all_properties):
            raise ValueError(
                f"The number of scalar properties in {file_path} does not match the predefined property list."
            )
        
        scalar_props = torch.tensor(scalar_props, dtype=torch.float)
        data.scalar_props = scalar_props
        
        last_line = lines[-1].strip().split()
        if len(last_line) >= 2:
            chiral_inchi = last_line[1]
            data.chiral_inchi = chiral_inchi
        
        return data
