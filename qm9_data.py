from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, List

import lmdb
import periodictable
import torch
from torch_geometric.data import Data

from graph_compat import radius_graph
from met_utils import QM9_PROPERTY_NAMES


def safe_float(token: str) -> float:
    return float(str(token).replace("*^", "e"))


def element_symbol_to_index() -> Dict[str, int]:
    return {element.symbol: idx for idx, element in enumerate(periodictable.elements)}


def parse_qm9_xyz_file(file_path: str | Path) -> Dict:
    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    atom_num = int(lines[0].strip())
    scalar_props = [safe_float(part) for part in lines[1].strip().split()[2:17]]

    atom_types: List[str] = []
    positions: List[List[float]] = []
    charges: List[float] = []
    for line in lines[2:2 + atom_num]:
        parts = line.strip().split()
        atom_types.append(parts[0])
        positions.append([safe_float(part) for part in parts[1:4]])
        charges.append(safe_float(parts[4]) if len(parts) >= 5 else 0.0)

    element_dict = element_symbol_to_index()
    z = [element_dict.get(atom_type, 0) for atom_type in atom_types]
    chiral_inchi = None
    last_line = lines[-1].strip().split()
    if len(last_line) >= 2:
        chiral_inchi = last_line[1]

    return {
        "filename": file_path.name,
        "z": z,
        "pos": positions,
        "y": charges,
        "scalar_props": scalar_props,
        "chiral_inchi": chiral_inchi,
    }


def qm9_record_to_data(record: Dict, cutoff: float) -> Data:
    z = torch.tensor(record["z"], dtype=torch.long)
    pos = torch.tensor(record["pos"], dtype=torch.float)
    y = torch.tensor(record["y"], dtype=torch.float).view(-1, 1)
    edge_index = radius_graph(pos, r=cutoff, loop=False)

    data = Data(x=z.view(-1, 1), pos=pos, edge_index=edge_index, y=y)
    data.filename = record["filename"]
    data.scalar_props = torch.tensor(record["scalar_props"], dtype=torch.float)
    data.scalar_mask = torch.ones_like(data.scalar_props, dtype=torch.bool)
    if record.get("chiral_inchi"):
        data.chiral_inchi = record["chiral_inchi"]
    return data


def is_lmdb_path(path: str | Path | None) -> bool:
    if path is None:
        return False
    path = Path(path)
    return path.suffix.lower() == ".lmdb" or (path.is_dir() and (path / "data.mdb").exists())


def lmdb_open_kwargs(path: str | Path) -> Dict:
    path = Path(path)
    return {"subdir": path.is_dir()}


def write_qm9_lmdb(
    xyz_files: Iterable[Path],
    output_path: str | Path,
    map_size: int = 64 * 1024**2,
    commit_interval: int = 1000,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(
        str(output_path),
        map_size=map_size,
        subdir=False,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
    )

    xyz_files = list(xyz_files)
    with env.begin(write=True) as txn:
        txn.put(b"__meta__/length", str(len(xyz_files)).encode("utf-8"))
        txn.put(b"__meta__/property_names", pickle.dumps(list(QM9_PROPERTY_NAMES), protocol=4))

    txn = env.begin(write=True)
    idx = 0
    try:
        while idx < len(xyz_files):
            xyz_path = xyz_files[idx]
            record = parse_qm9_xyz_file(xyz_path)
            key = f"{idx:08d}".encode("ascii")
            value = pickle.dumps(record, protocol=4)
            try:
                txn.put(key, value)
                idx += 1
                if idx % max(1, commit_interval) == 0:
                    txn.commit()
                    txn = env.begin(write=True)
            except lmdb.MapFullError:
                txn.abort()
                current_map_size = env.info()["map_size"]
                grown_map_size = max(current_map_size * 2, current_map_size + 64 * 1024**2)
                env.set_mapsize(grown_map_size)
                txn = env.begin(write=True)
        txn.commit()
    except Exception:
        txn.abort()
        raise

    env.sync()
    env.close()
    return output_path


def load_qm9_lmdb_record(env: lmdb.Environment, index: int) -> Dict:
    with env.begin(write=False) as txn:
        payload = txn.get(f"{index:08d}".encode("ascii"))
    if payload is None:
        raise IndexError(f"Missing LMDB record at index {index}.")
    return pickle.loads(payload)
