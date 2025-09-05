# -*- coding: utf-8 -*-

import pysmiles
import networkx as nx
from matplotlib import pyplot as plt

# 例如一个 SMILES 字符串
smiles = 'CC1CC(C)OCCO1'  
mol = pysmiles.read_smiles(smiles)
mol_with_H = pysmiles.read_smiles(smiles, explicit_hydrogen=True)
print(mol_with_H.nodes(data='element'))
# 将 SMILES 转换为 NetworkX 图
mol_graph = pysmiles.read_smiles(smiles)

# 输出节点和边的信息
print("Nodes (Atoms):", mol_graph.nodes(data=True))
print("Edges (Bonds):", mol_graph.edges(data=True))

# 可视化分子图
nx.draw(mol_graph, with_labels=True)
plt.savefig('test.png')
