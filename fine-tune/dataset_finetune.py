# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import periodictable
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from graph_compat import radius_graph
from met_utils import QM7_PROPERTY_NAMES, QM9_PROPERTY_NAMES, collect_files


def _normalize_num_token(token: str) -> str:
    token = token.strip().strip(",;")
    token = token.replace("*^", "e").replace("^", "e")
    return token


def _safe_float(token: str) -> float:
    try:
        return float(_normalize_num_token(token))
    except Exception:
        cleaned = re.sub(r"[^\dEe+\-\.]+$", "", _normalize_num_token(token))
        return float(cleaned) if cleaned else float("nan")


def _elements_symbol_to_index() -> Dict[str, int]:
    return {element.symbol: idx for idx, element in enumerate(periodictable.elements)}


def _build_conformers_from_smiles(
    smiles: str,
    elem_map: Dict[str, int],
    num_conformers: int,
    conformer_seed: int,
    use_uff_optimization: bool,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = conformer_seed
    params.pruneRmsThresh = 0.1
    conformer_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=max(1, num_conformers), params=params))
    if not conformer_ids:
        params.useRandomCoords = True
        conformer_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=max(1, num_conformers), params=params))
    if not conformer_ids:
        raise RuntimeError(f"RDKit ETKDG failed for: {smiles}")

    if use_uff_optimization:
        try:
            AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=200)
        except Exception:
            pass

    conformers = []
    for conformer_id in conformer_ids:
        conf = mol.GetConformer(conformer_id)
        num_atoms = mol.GetNumAtoms()
        pos = torch.zeros((num_atoms, 3), dtype=torch.float)
        z = torch.zeros((num_atoms,), dtype=torch.long)
        for atom_idx, atom in enumerate(mol.GetAtoms()):
            z[atom_idx] = elem_map.get(atom.GetSymbol(), 0)
            coord = conf.GetAtomPosition(atom_idx)
            pos[atom_idx] = torch.tensor([coord.x, coord.y, coord.z], dtype=torch.float)
        conformers.append((z, pos))
    return conformers


def _parse_known_xyz_properties(comment_line: str) -> Tuple[List[str], List[float], str]:
    tokens = comment_line.strip().split()
    if not tokens:
        return [], [], "generic"

    prefix = tokens[0].lower()
    if prefix == "gdb" and len(tokens) >= 17:
        values = [_safe_float(tok) for tok in tokens[2:17]]
        return list(QM9_PROPERTY_NAMES), values, "qm9"

    if prefix == "qm7" and len(tokens) >= 3:
        values = [_safe_float(tokens[-1])]
        return list(QM7_PROPERTY_NAMES), values, "qm7"

    values = [_safe_float(tok) for tok in tokens]
    names = [f"P{i}" for i in range(len(values))]
    return names, values, "generic"


class MoleculeDataset(Dataset):
    def __init__(
        self,
        root: Optional[str] = None,
        transform=None,
        pre_transform=None,
        cutoff: float = 2.5,
        manifest_path: Optional[str] = None,
        num_conformers: int = 1,
        conformer_seed: int = 2024,
        use_uff_optimization: bool = True,
    ):
        super().__init__(root or manifest_path or ".", transform, pre_transform)
        self.root = root
        self.cutoff = cutoff
        self.manifest_path = manifest_path
        self.num_conformers = max(1, num_conformers)
        self.conformer_seed = conformer_seed
        self.use_uff_optimization = use_uff_optimization
        self._elem_map = _elements_symbol_to_index()

        self.xyz_files = collect_files(root=root, manifest_path=manifest_path, suffixes=(".xyz",)) if self._has_xyz_input() else []
        self.csv_files = collect_files(root=root, manifest_path=manifest_path, suffixes=(".csv",)) if self._has_csv_input() else []

        self.items: List[Tuple[str, object]] = [("xyz", file_path) for file_path in self.xyz_files]
        self._csv_tables: Dict[Path, pd.DataFrame] = {}
        self._csv_numcols: Dict[Path, List[str]] = {}

        for csv_path in self.csv_files:
            df = pd.read_csv(csv_path)
            self._csv_tables[csv_path] = df
            reserved = {"smiles", "smile", "SMILES", "SMILE", "id", "ID", "name", "Name", "filename", "Filename"}
            numcols = [col for col in df.columns if col not in reserved and pd.api.types.is_numeric_dtype(df[col])]
            self._csv_numcols[csv_path] = numcols
            for row_idx in range(len(df)):
                self.items.append(("csv", (csv_path, row_idx)))

        self.files = []
        property_order: List[str] = []
        dataset_kinds = set()
        for item_type, meta in self.items:
            if item_type == "xyz":
                names, _, dataset_kind = self._peek_xyz_properties(meta)
                self.files.append(str(meta))
            else:
                csv_path, row_idx = meta
                names = list(self._csv_numcols[csv_path])
                dataset_kind = "csv"
                self.files.append(f"{csv_path}#{row_idx}")

            for name in names:
                if name not in property_order:
                    property_order.append(name)
            dataset_kinds.add(dataset_kind)

        self.dataset_kind = dataset_kinds.pop() if len(dataset_kinds) == 1 else "mixed"
        self.all_properties = property_order
        self.property_to_index = {name: idx for idx, name in enumerate(self.all_properties)}
        for idx, name in enumerate(self.all_properties):
            self.property_to_index[f"P{idx}"] = idx
            self.property_to_index[f"P{idx + 2}"] = idx
        if self.dataset_kind == "qm7" and "atomization_energy" in self.property_to_index:
            energy_idx = self.property_to_index["atomization_energy"]
            self.property_to_index["rot_A"] = energy_idx

    def _has_csv_input(self) -> bool:
        if self.manifest_path is None:
            if self.root is None:
                return False
            root_path = Path(self.root)
            if root_path.is_file():
                return root_path.suffix.lower() == ".csv"
            return any(name.lower().endswith(".csv") for name in os.listdir(root_path))
        manifest = Path(self.manifest_path)
        for line in manifest.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and stripped.lower().endswith(".csv"):
                return True
        return False

    def _has_xyz_input(self) -> bool:
        if self.manifest_path is None:
            if self.root is None:
                return False
            root_path = Path(self.root)
            if root_path.is_file():
                return root_path.suffix.lower() == ".xyz"
            return any(name.lower().endswith(".xyz") for name in os.listdir(root_path))
        manifest = Path(self.manifest_path)
        for line in manifest.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and stripped.lower().endswith(".xyz"):
                return True
        return False

    def _peek_xyz_properties(self, file_path: Path) -> Tuple[List[str], List[float], str]:
        with file_path.open("r", encoding="utf-8") as f:
            _ = f.readline()
            comment = f.readline().rstrip("\n")
        return _parse_known_xyz_properties(comment)

    def len(self):
        return len(self.items)

    def resolve_property_name(self, property_name: str) -> str:
        if property_name not in self.property_to_index:
            raise KeyError(f"Unknown property '{property_name}'. Available names: {sorted(self.property_to_index)}")
        return self.all_properties[self.property_to_index[property_name]]

    def get(self, idx):
        item_type, meta = self.items[idx]
        if item_type == "xyz":
            return self._get_from_xyz(meta)
        csv_path, row_idx = meta
        return self._get_from_csv(csv_path, row_idx)

    def _vectorize_scalar_properties(self, prop_map: Dict[str, float]) -> Tuple[torch.Tensor, torch.Tensor]:
        values = torch.full((len(self.all_properties),), float("nan"), dtype=torch.float)
        for name, value in prop_map.items():
            values[self.property_to_index[name]] = float(value)
        mask = ~torch.isnan(values)
        return values, mask

    def _make_data_object(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        y: torch.Tensor,
        filename: str,
        scalar_props: torch.Tensor,
        scalar_mask: torch.Tensor,
        chiral_inchi: Optional[str] = None,
    ) -> Data:
        edge_index = radius_graph(pos, r=self.cutoff, loop=False)
        data = Data(x=z.view(-1, 1), pos=pos, edge_index=edge_index, y=y)
        data.filename = filename
        data.scalar_props = scalar_props.clone()
        data.scalar_mask = scalar_mask.clone()
        if chiral_inchi:
            data.chiral_inchi = chiral_inchi
        return data

    def _wrap_sample(self, conformers: List[Data], scalar_props: torch.Tensor, scalar_mask: torch.Tensor, filename: str) -> Dict:
        return {
            "conformers": conformers,
            "scalar_props": scalar_props,
            "scalar_mask": scalar_mask,
            "filename": filename,
        }

    def _get_from_xyz(self, file_path: Path) -> Dict:
        with file_path.open("r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f]

        if len(lines) < 3:
            raise ValueError(f"Bad xyz file (too few lines): {file_path}")

        atom_num = int(lines[0].strip())
        prop_names, prop_values, _ = _parse_known_xyz_properties(lines[1])
        prop_map = {name: value for name, value in zip(prop_names, prop_values)}
        scalar_props, scalar_mask = self._vectorize_scalar_properties(prop_map)

        atom_types: List[str] = []
        positions: List[List[float]] = []
        charges: List[float] = []
        for atom_line in lines[2:2 + atom_num]:
            parts = atom_line.strip().split()
            if len(parts) < 4:
                raise ValueError(f"Bad atom line in {file_path}: {atom_line}")
            atom_types.append(parts[0])
            positions.append([_safe_float(part) for part in parts[1:4]])
            charges.append(_safe_float(parts[4]) if len(parts) >= 5 else 0.0)

        z = torch.tensor([self._elem_map.get(symbol, 0) for symbol in atom_types], dtype=torch.long)
        pos = torch.tensor(positions, dtype=torch.float)
        y = torch.tensor(charges, dtype=torch.float).view(-1, 1)
        chiral_inchi = None
        last_parts = lines[-1].strip().split()
        if len(last_parts) >= 2:
            chiral_inchi = last_parts[1]

        conformer = self._make_data_object(z, pos, y, file_path.name, scalar_props, scalar_mask, chiral_inchi=chiral_inchi)
        return self._wrap_sample([conformer], scalar_props, scalar_mask, file_path.name)

    def _get_from_csv(self, csv_path: Path, row_idx: int) -> Dict:
        df = self._csv_tables[csv_path]
        row = df.iloc[row_idx]

        smiles = None
        for column in ("smiles", "SMILES", "smile", "SMILE"):
            if column in df.columns and not pd.isna(row[column]):
                smiles = str(row[column])
                break
        if not smiles:
            raise ValueError(f"CSV must contain a non-empty SMILES column: {csv_path}")

        prop_map = {}
        for column in self._csv_numcols[csv_path]:
            prop_map[column] = float(row[column]) if not pd.isna(row[column]) else float("nan")
        scalar_props, scalar_mask = self._vectorize_scalar_properties(prop_map)

        conformers = []
        y = None
        for conformer_idx, (z, pos) in enumerate(
            _build_conformers_from_smiles(
                smiles,
                elem_map=self._elem_map,
                num_conformers=self.num_conformers,
                conformer_seed=self.conformer_seed,
                use_uff_optimization=self.use_uff_optimization,
            )
        ):
            if y is None:
                y = torch.zeros((z.numel(), 1), dtype=torch.float)
            conformers.append(
                self._make_data_object(
                    z,
                    pos,
                    y,
                    f"{csv_path.name}#{row_idx}_conf{conformer_idx}",
                    scalar_props,
                    scalar_mask,
                )
            )

        return self._wrap_sample(conformers, scalar_props, scalar_mask, f"{csv_path.name}#{row_idx}")
