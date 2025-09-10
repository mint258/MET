# dataset_without_charge.py
# -*- coding: utf-8 -*-

import os
import json
import math
from typing import List, Dict, Tuple, Optional

import torch
from torch_geometric.data import Dataset, Data
from torch_cluster import radius_graph

import periodictable
from rdkit import Chem
from rdkit.Chem import AllChem

import pandas as pd
import re


# -------------------- small utils --------------------

def _normalize_num_token(tok: str) -> str:
    """把 '1.23*^4' 这类写法转为 '1.23e4'，顺便剥离多余逗号分号。"""
    tok = tok.strip().strip(",;")
    tok = tok.replace("*^", "e").replace("^", "e")
    return tok


def _parse_xyz_comment_numeric_seq(line: str) -> List[float]:
    """
    从 .xyz 第二行注释“按位置”抽取数值序列：
      - 以空白/逗号/分号拆分为 token（不丢位置）
      - 依次尝试：
          1) 直接把 token 转为 float
          2) 若失败且包含 '=' 或 ':'，取右侧子串再转为 float
          3) 若仍失败，则该位置记为 NaN
    返回长度 = token 数的列表（数值或 NaN），以保证按位置索引。
    """
    s = line.strip()
    if not s:
        return []

    tokens = re.split(r"[\s,;]+", s)
    seq: List[float] = []

    for t in tokens:
        if not t:
            continue

        raw = t.strip()
        # 若是 k=v / k:v，优先取右侧
        if "=" in raw:
            _, rhs = raw.split("=", 1)
            cand = rhs
        elif ":" in raw:
            _, rhs = raw.split(":", 1)
            cand = rhs
        else:
            cand = raw

        # 规范化科学计数法 & 去除成对括号/引号
        cand = _normalize_num_token(cand)
        cand = cand.strip().strip('[]{}()"\'')  # 容错去壳

        # 逐步尝试转为 float
        try:
            val = float(cand)
        except Exception:
            # 如果像 "1921}" 这种尾随符号，再做一次温和清理
            cand2 = re.sub(r"[^\dEe+\-\.]+$", "", cand)
            try:
                val = float(cand2)
            except Exception:
                val = float("nan")

        seq.append(val)

    return seq

def _elements_symbol_to_index() -> Dict[str, int]:
    """
    维持与旧版兼容的“元素符号 -> 索引”映射（按 periodictable.elements 的枚举顺序），
    注意这不是原子序数 Z。
    """
    return {el.symbol: idx for idx, el in enumerate(periodictable.elements)}


def _build_graph_from_smiles(smiles: str, try_uff: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    用 RDKit 从 SMILES 生成 3D 构象，返回 (z, pos)
    z: [N] long (沿用旧版“元素索引”而非 Z)
    pos: [N, 3] float
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    # 生成 3D 坐标
    params = AllChem.ETKDGv3()
    params.randomSeed = 2024
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol, params)
        if status != 0:
            raise RuntimeError(f"RDKit ETKDG failed for: {smiles}")
    if try_uff:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
    conf = mol.GetConformer()
    N = mol.GetNumAtoms()
    pos = torch.zeros((N, 3), dtype=torch.float)
    z = torch.zeros((N,), dtype=torch.long)
    elem_map = _elements_symbol_to_index()
    for i, atom in enumerate(mol.GetAtoms()):
        sym = atom.GetSymbol()
        z[i] = elem_map.get(sym, 0)
        c = conf.GetAtomPosition(i)
        pos[i, 0] = c.x
        pos[i, 1] = c.y
        pos[i, 2] = c.z
    return z, pos


# -------------------- unified dataset --------------------

class MoleculeDataset(Dataset):
    """
    统一数据集（接口保持不变）：
      - 同一目录下同时支持 *.xyz 与 *.csv
      - 训练监督的“标量属性”不再依赖旧的 property 列表，而是：
          扫描目录后确定本数据集的“位置维度 P”（position_dim）：
            * XYZ：仅当第二行注释为“纯数字序列”时，长度计入位置维度
            * CSV：把除 smiles 外的数值列按列顺序视作位置 0..K-1（每行可能不同 K）
          最终 P 为全体样本的最大长度
      - all_properties = ["P0", "P1", ..., f"P{P-1}"] ；property_to_index 相应建立
      - 每个样本返回 Data(...) 并附带：
          data.scalar_props: torch.float[P]（缺失填 NaN）
          data.scalar_mask : torch.bool[P]（非 NaN 为 True）
          data.filename    : str
    """

    def __init__(self, root, transform=None, pre_transform=None, cutoff=2.5):
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        self.root = root
        self.cutoff = cutoff

        # 收集 xyz / csv
        self.xyz_files: List[str] = sorted(
            [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(".xyz")]
        )
        self.csv_files: List[str] = sorted(
            [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(".csv")]
        )

        # 构建样本索引表：
        #   xyz -> ("xyz", path)
        #   csv -> ("csv", (csv_path, row_idx))
        self.items: List[Tuple[str, object]] = []
        for p in self.xyz_files:
            self.items.append(("xyz", p))

        # 惰性缓存 CSV 表 & 它们的“数值列顺序”
        self._csv_tables: Dict[str, pd.DataFrame] = {}
        self._csv_numcols: Dict[str, List[str]] = {}
        for csv_path in self.csv_files:
            df = pd.read_csv(csv_path)
            self._csv_tables[csv_path] = df
            # 识别数值列（排除 smiles / id 等）
            reserved = {"smiles", "smile", "SMILES", "SMILE", "id", "ID", "name", "Name", "filename", "Filename"}
            numcols = [c for c in df.columns
                       if (c not in reserved) and pd.api.types.is_numeric_dtype(df[c])]
            self._csv_numcols[csv_path] = numcols
            for i in range(len(df)):
                self.items.append(("csv", (csv_path, i)))

        # -------- 扫描确定“位置维度 P” --------
        position_dim = 0

        # xyz: 第二行注释为“纯数字序列”的长度
        for p in self.xyz_files:
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    lines = fh.readlines()
                if len(lines) >= 2:
                    seq = _parse_xyz_comment_numeric_seq(lines[1])
                    if len(seq) > position_dim:
                        position_dim = len(seq)
            except Exception:
                continue

        # csv: “数值列”个数
        for csv_path, df in self._csv_tables.items():
            k = len(self._csv_numcols[csv_path])
            if k > position_dim:
                position_dim = k

        self.position_dim: int = position_dim
        self.all_properties: List[str] = [f"P{i}" for i in range(self.position_dim)]
        self.property_to_index: Dict[str, int] = {p: i for i, p in enumerate(self.all_properties)}

        # files 字段（保持兼容，可打印样本定位）
        self.files: List[str] = []
        for kind, meta in self.items:
            if kind == "xyz":
                self.files.append(meta)
            else:
                csv_path, i = meta
                self.files.append(f"{csv_path}#{i}")

        # 元素映射（符号 -> 索引）
        self._elem_map = _elements_symbol_to_index()

    # ---- torch_geometric required ----
    def len(self):
        return len(self.items)

    def get(self, idx):
        kind, meta = self.items[idx]
        if kind == "xyz":
            return self._get_from_xyz(meta)
        else:
            csv_path, row_idx = meta
            return self._get_from_csv(csv_path, row_idx)

    # ---------------- internal helpers ----------------

    def _get_from_xyz(self, file_path: str) -> Data:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]

        if len(lines) < 3:
            raise ValueError(f"Bad xyz file (too few lines): {file_path}")

        # 第 1 行：原子数
        try:
            atom_num = int(str(lines[0]).strip())
        except Exception:
            raise ValueError(f"First line must be integer atom count: {file_path}")

        # 第 2 行：尝试解析为“纯数字序列”
        seq = _parse_xyz_comment_numeric_seq(lines[1])

        # 第 3..(2+N) 行：元素 + 坐标 (+ 可选电荷)
        atom_lines = lines[2:2 + atom_num]
        if len(atom_lines) != atom_num:
            raise ValueError(f"Atom count mismatch in {file_path}: header N={atom_num}, but got {len(atom_lines)} lines")

        atom_types: List[str] = []
        positions: List[List[float]] = []
        charges: List[float] = []
        for L in atom_lines:
            parts = L.strip().split()
            if len(parts) < 4:
                raise ValueError(f"Bad atom line: {L}")
            sym = parts[0]
            xyz = [_normalize_num_token(x) for x in parts[1:4]]
            pos = [float(x) for x in xyz]
            atom_types.append(sym)
            positions.append(pos)
            if len(parts) >= 5:
                try:
                    chg = float(_normalize_num_token(parts[4]))
                except Exception:
                    chg = 0.0
            else:
                chg = 0.0
            charges.append(chg)

        z = torch.tensor([self._elem_map.get(sym, 0) for sym in atom_types], dtype=torch.long)
        pos = torch.tensor(positions, dtype=torch.float)
        charge = torch.tensor(charges, dtype=torch.float).view(-1, 1)

        # 半径图
        edge_index = radius_graph(pos, r=self.cutoff, loop=False)

        data = Data(x=z.view(-1, 1), pos=pos, edge_index=edge_index, y=charge)
        data.filename = os.path.basename(file_path)

        # 对齐位置向量 P：序列填前 len(seq)，其余置 NaN
        P = self.position_dim
        vals = torch.full((P,), float("nan"), dtype=torch.float) if P > 0 else torch.empty((0,), dtype=torch.float)
        if seq and P > 0:
            n = min(P, len(seq))
            vals[:n] = torch.tensor(seq[:n], dtype=torch.float)
        mask = ~torch.isnan(vals) if P > 0 else torch.empty((0,), dtype=torch.bool)

        data.scalar_props = vals
        data.scalar_mask = mask

        # 可选：最后一行可能含有 InChI 信息（保持兼容）
        try:
            last_parts = str(lines[-1]).strip().split()
            if len(last_parts) >= 2:
                data.chiral_inchi = last_parts[1]
        except Exception:
            pass

        return data

    def _get_from_csv(self, csv_path: str, row_idx: int) -> Data:
        df = self._csv_tables[csv_path]
        row = df.iloc[row_idx]

        # 寻找 smiles 列
        smiles = None
        for key in ("smiles", "SMILES", "smile", "SMILE"):
            if key in df.columns:
                val = row[key]
                smiles = str(val) if (not pd.isna(val)) else None
                break
        if not smiles:
            raise ValueError(f"CSV must contain a 'smiles' column: {csv_path}")

        # 生成 3D
        z, pos = _build_graph_from_smiles(smiles)

        # 半径图
        edge_index = radius_graph(pos, r=self.cutoff, loop=False)

        # y：保持接口（逐原子电荷），CSV 无此信息，填 0
        y = torch.zeros((z.numel(), 1), dtype=torch.float)

        data = Data(x=z.view(-1, 1), pos=pos, edge_index=edge_index, y=y)
        data.filename = f"{os.path.basename(csv_path)}#{row_idx}"

        # CSV 数值列 -> 位置序列（按列顺序）
        numcols = self._csv_numcols[csv_path]
        seq = []
        for c in numcols:
            v = row[c]
            if pd.isna(v):
                seq.append(float("nan"))
            else:
                try:
                    seq.append(float(v))
                except Exception:
                    seq.append(float("nan"))

        P = self.position_dim
        vals = torch.full((P,), float("nan"), dtype=torch.float) if P > 0 else torch.empty((0,), dtype=torch.float)
        if P > 0 and len(seq) > 0:
            n = min(P, len(seq))
            vals[:n] = torch.tensor(seq[:n], dtype=torch.float)
        mask = ~torch.isnan(vals) if P > 0 else torch.empty((0,), dtype=torch.bool)

        data.scalar_props = vals
        data.scalar_mask = mask

        return data
