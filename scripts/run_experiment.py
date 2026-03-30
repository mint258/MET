#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(command):
    print(" ".join(str(part) for part in command))
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Convenience launcher for MET experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-splits")
    prepare.add_argument("--output_dir", default="manifests")
    prepare.add_argument("--seed", type=int, default=42)

    build_lmdb = subparsers.add_parser("build-qm9-lmdb")
    build_lmdb.add_argument("--python", default=sys.executable)
    build_lmdb.add_argument("--input_root", default=str(REPO_ROOT / "data/QM9/train_valid_database"))
    build_lmdb.add_argument("--output_path", default=str(REPO_ROOT / "data/QM9/train_valid_database.lmdb"))
    build_lmdb.add_argument("--map_size_gb", type=float, default=0.064)

    build_qm7_lmdb = subparsers.add_parser("build-qm7-lmdb")
    build_qm7_lmdb.add_argument("--python", default=sys.executable)
    build_qm7_lmdb.add_argument("--input_root", default=str(REPO_ROOT / "data/QM7/full_database"))
    build_qm7_lmdb.add_argument("--output_path", default=str(REPO_ROOT / "data/QM7/full_database.lmdb"))
    build_qm7_lmdb.add_argument("--map_size_gb", type=float, default=0.064)

    pretrain = subparsers.add_parser("pretrain-qm9")
    pretrain.add_argument("--python", default=sys.executable)
    pretrain.add_argument("--device", default="cpu")
    pretrain.add_argument("--epochs", type=int, default=100)
    pretrain.add_argument("--save_path", default="pretrained_ckpt/best_model_dim128_reproduced.pth")
    pretrain.add_argument("--data_root", default=None)

    qm7 = subparsers.add_parser("finetune-qm7")
    qm7.add_argument("--python", default=sys.executable)
    qm7.add_argument("--device", default="cpu")
    qm7.add_argument("--epochs", type=int, default=100)
    qm7.add_argument("--subset_size", type=int, default=300, choices=[100, 300, 500, 800, 1000])
    qm7.add_argument("--pretrained_checkpoint", default="pretrained_ckpt/best_model_dim128.pth")
    qm7.add_argument("--save_model", default=None)

    dipole = subparsers.add_parser("finetune-qm9-dipole")
    dipole.add_argument("--python", default=sys.executable)
    dipole.add_argument("--device", default="cpu")
    dipole.add_argument("--epochs", type=int, default=100)
    dipole.add_argument("--subset_size", type=int, default=1000, choices=[1000, 5000, 20000, 100000])
    dipole.add_argument("--pretrained_checkpoint", default="pretrained_ckpt/best_model_dim128.pth")
    dipole.add_argument("--save_model", default=None)

    args = parser.parse_args()

    if args.command == "prepare-splits":
        run([sys.executable, str(REPO_ROOT / "scripts/create_manifests.py"), "--output_dir", args.output_dir, "--seed", str(args.seed)])
        return

    if args.command == "build-qm9-lmdb":
        run(
            [
                args.python,
                str(REPO_ROOT / "scripts/build_qm9_lmdb.py"),
                "--input_root",
                str(args.input_root),
                "--output_path",
                str(args.output_path),
                "--map_size_gb",
                str(args.map_size_gb),
            ]
        )
        return

    if args.command == "build-qm7-lmdb":
        run(
            [
                args.python,
                str(REPO_ROOT / "scripts/build_qm7_lmdb.py"),
                "--input_root",
                str(args.input_root),
                "--output_path",
                str(args.output_path),
                "--map_size_gb",
                str(args.map_size_gb),
            ]
        )
        return

    if args.command == "pretrain-qm9":
        default_data_root = REPO_ROOT / "data/QM9/train_valid_database.lmdb"
        if not default_data_root.exists():
            default_data_root = REPO_ROOT / "data/QM9/train_valid_database"
        run(
            [
                args.python,
                str(REPO_ROOT / "pretrain/training_charge_model.py"),
                "--data_root",
                str(Path(args.data_root).resolve()) if args.data_root else str(default_data_root),
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--save_path",
                str(REPO_ROOT / args.save_path),
            ]
        )
        return

    if args.command == "finetune-qm7":
        save_model = args.save_model or f"fine-tuned_ckpt/qm7/qm7_data{args.subset_size}_reproduced.pth"
        run(
            [
                args.python,
                str(REPO_ROOT / "fine-tune/fine_tune_training.py"),
                "--pretrained_checkpoint_path",
                str(REPO_ROOT / args.pretrained_checkpoint),
                "--data_manifest",
                str(REPO_ROOT / f"manifests/qm7_train_{args.subset_size}.txt"),
                "--target_property",
                "atomization_energy",
                "--dropout",
                "0",
                "--learning_rate",
                "1e-4",
                "--freeze_up_to_layer",
                "4",
                "--num_layers",
                "0",
                "--num_heads",
                "1",
                "--epochs",
                str(args.epochs),
                "--device",
                args.device,
                "--save_model",
                str(REPO_ROOT / save_model),
            ]
        )
        return

    if args.command == "finetune-qm9-dipole":
        save_model = args.save_model or f"fine-tuned_ckpt/dipole/dipole_data{args.subset_size}_reproduced.pth"
        run(
            [
                args.python,
                str(REPO_ROOT / "fine-tune/fine_tune_training.py"),
                "--pretrained_checkpoint_path",
                str(REPO_ROOT / args.pretrained_checkpoint),
                "--data_manifest",
                str(REPO_ROOT / f"manifests/qm9_train_{args.subset_size}.txt"),
                "--target_property",
                "dipole",
                "--dropout",
                "0",
                "--learning_rate",
                "1e-4",
                "--freeze_up_to_layer",
                "4",
                "--num_linear_layers",
                "0",
                "--epochs",
                str(args.epochs),
                "--device",
                args.device,
                "--save_model",
                str(REPO_ROOT / save_model),
            ]
        )


if __name__ == "__main__":
    main()
