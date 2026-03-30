from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, List

import lmdb

from met_utils import QM7_PROPERTY_NAMES
from qm9_data import element_symbol_to_index, safe_float


def parse_qm7_xyz_file(file_path: str | Path) -> Dict:
    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    atom_num = int(lines[0].strip())
    scalar_props = [safe_float(lines[1].strip().split()[-1])]

    atom_types: List[str] = []
    positions: List[List[float]] = []
    for line in lines[2:2 + atom_num]:
        parts = line.strip().split()
        atom_types.append(parts[0])
        positions.append([safe_float(part) for part in parts[1:4]])

    element_dict = element_symbol_to_index()
    z = [element_dict.get(atom_type, 0) for atom_type in atom_types]

    return {
        "filename": file_path.name,
        "z": z,
        "pos": positions,
        "y": [0.0] * atom_num,
        "scalar_props": scalar_props,
        "chiral_inchi": None,
    }


def write_qm7_lmdb(
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
        txn.put(b"__meta__/property_names", pickle.dumps(list(QM7_PROPERTY_NAMES), protocol=4))

    txn = env.begin(write=True)
    idx = 0
    try:
        while idx < len(xyz_files):
            xyz_path = xyz_files[idx]
            record = parse_qm7_xyz_file(xyz_path)
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
