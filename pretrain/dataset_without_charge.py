# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

import lmdb
from torch_geometric.data import Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from met_utils import QM9_PROPERTY_NAMES, collect_files
from qm9_data import is_lmdb_path, lmdb_open_kwargs, load_qm9_lmdb_record, qm9_record_to_data


class MoleculeDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, cutoff=2.5, manifest_path=None):
        super().__init__(root or manifest_path or ".", transform, pre_transform)
        self.cutoff = cutoff
        self.all_properties = list(QM9_PROPERTY_NAMES)
        self.property_to_index = {prop: idx for idx, prop in enumerate(self.all_properties)}
        self._env = None
        self._source_type = "xyz"

        if is_lmdb_path(root):
            self._source_type = "lmdb"
            self._lmdb_path = Path(root)
            self._env = lmdb.open(
                str(self._lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                **lmdb_open_kwargs(self._lmdb_path),
            )
            with self._env.begin(write=False) as txn:
                self._length = int(txn.get(b"__meta__/length").decode("utf-8"))
            self.files = [f"{self._lmdb_path}#{idx}" for idx in range(self._length)]
        else:
            self.files = collect_files(root=root, manifest_path=manifest_path, suffixes=(".xyz",))

    def len(self):
        return len(self.files)

    def get(self, idx):
        if self._source_type == "lmdb":
            record = load_qm9_lmdb_record(self._env, idx)
            return qm9_record_to_data(record, cutoff=self.cutoff)

        from qm9_data import parse_qm9_xyz_file

        record = parse_qm9_xyz_file(self.files[idx])
        return qm9_record_to_data(record, cutoff=self.cutoff)

    def __del__(self):
        if self._env is not None:
            self._env.close()
