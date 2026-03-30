#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from met_utils import collect_files
from qm9_data import write_qm9_lmdb


def main():
    parser = argparse.ArgumentParser(description="Convert a QM9 xyz directory or manifest to LMDB.")
    parser.add_argument("--input_root", default=None, help="Directory containing QM9 xyz files.")
    parser.add_argument("--input_manifest", default=None, help="Optional manifest listing QM9 xyz files.")
    parser.add_argument("--output_path", required=True, help="Output LMDB directory path.")
    parser.add_argument("--map_size_gb", type=float, default=0.064, help="Initial LMDB map size in GB. The builder auto-expands if needed.")
    args = parser.parse_args()

    xyz_files = collect_files(root=args.input_root, manifest_path=args.input_manifest, suffixes=(".xyz",))
    output_path = write_qm9_lmdb(
        xyz_files,
        output_path=args.output_path,
        map_size=int(args.map_size_gb * 1024**3),
    )
    print(f"Wrote {len(xyz_files)} QM9 records to {output_path}")


if __name__ == "__main__":
    main()
