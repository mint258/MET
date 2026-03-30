#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
from pathlib import Path


def write_manifest(path: Path, files):
    path.parent.mkdir(parents=True, exist_ok=True)
    rel_lines = [str(Path(file_path).resolve().relative_to(path.parent.parent)) for file_path in files]
    path.write_text("\n".join(rel_lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Create deterministic subset manifests for MET reproduction.")
    parser.add_argument("--output_dir", default="manifests")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()

    rng = random.Random(args.seed)

    qm7_test_names = {path.name for path in (repo_root / "data/QM7/test_database").glob("*.xyz")}
    qm7_pool = sorted(
        path for path in (repo_root / "data/QM7/full_database").glob("*.xyz") if path.name not in qm7_test_names
    )
    qm7_shuffled = qm7_pool[:]
    rng.shuffle(qm7_shuffled)
    for size in [100, 300, 500, 800, 1000]:
        write_manifest(output_dir / f"qm7_train_{size}.txt", qm7_shuffled[:size])

    qm9_pool = sorted((repo_root / "data/QM9/train_valid_database").glob("*.xyz"))
    qm9_shuffled = qm9_pool[:]
    rng.shuffle(qm9_shuffled)
    for size in [100, 1000, 5000, 20000, 100000]:
        write_manifest(output_dir / f"qm9_train_{size}.txt", qm9_shuffled[:size])

    print(f"Wrote manifests to {output_dir}")


if __name__ == "__main__":
    main()
