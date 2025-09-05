#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate 50 position-isomers on an n-octane backbone (C1-C8) for
five functional groups, while
1) eliminating mirror duplicates via  i ↔ 9-i symmetry,
2) covering 1- to 6-substituted cases in升序 until 50 patterns collected.
"""

from itertools import combinations
from rdkit import Chem

# ---------- 1. 生成 50 组“取代位集合”并去掉镜像 ----------
TARGET_N = 50
subs_patterns = []          # list[ tuple(sorted positions) ]
seen = set()                # store frozenset + mirror to skip duplicates

def is_mirror_duplicate(pos_set):
    mirror = frozenset(9 - i for i in pos_set)  # 9-i
    key    = frozenset(pos_set)
    return key in seen or mirror in seen

for k in range(1, 7):  # 1-…6-取代
    for comb in combinations(range(1, 9), k):  # positions 1-8
        if is_mirror_duplicate(comb):
            continue
        subs_patterns.append(comb)
        seen.add(frozenset(comb))
        if len(subs_patterns) >= TARGET_N:
            break
    if len(subs_patterns) >= TARGET_N:
        break

# ---------- 2. 工具函数：根据位集合 + 官能团生成 SMILES ----------
def make_octane(group, pos_set):
    """
    group : e.g. 'F'  '[N+](=O)[O-]'  'C#N'  'C(=O)O'  'O'
    pos_set : iterable of ints, 1-8 被取代的位置
    """
    atoms = []
    for idx in range(1, 9):             # 线性正辛烷骨架
        if idx in pos_set:
            atoms.append(f"C({group})")
        else:
            atoms.append("C")
    smi_raw = "".join(atoms)
    # 让 RDKit 重新 canonicalize，避免末端 "C(F)F" 这类多余括号
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi_raw))

# ---------- 3. 生成各官能团列表 ----------
groups = {
    "Fluoro" : "F",
    "Nitro"  : "[N+](=O)[O-]",
    "Cyano"  : "C#N",
    "Carboxy": "C(=O)O",
    "Hydroxy": "O"
}

result = {name: [make_octane(g, p) for p in subs_patterns]
          for name, g in groups.items()}

# ---------- 4. 打印 / 或写文件 ----------
for name, smi_list in result.items():
    print(f"\n{name} ({len(smi_list)}):")
    for smi in smi_list:
        print(f'\"{smi}\",')
