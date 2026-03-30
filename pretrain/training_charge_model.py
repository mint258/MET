# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from comenet4charge import ComENetAutoEncoder
from dataset_without_charge import MoleculeDataset
from met_utils import ensure_parent_dir, resolve_device, save_json, set_random_seed


def run_epoch(model, device, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_nodes = 0
    all_predictions = []
    all_targets = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for data in tqdm(loader, desc="Training" if is_train else "Validation"):
            data = data.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            _, predictions = model(data)
            loss = criterion(predictions, data.y)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_nodes = data.y.numel()
            total_loss += loss.item() * batch_nodes
            total_nodes += batch_nodes
            all_predictions.append(predictions.detach().cpu().view(-1))
            all_targets.append(data.y.detach().cpu().view(-1))

    predictions = torch.cat(all_predictions).numpy()
    targets = torch.cat(all_targets).numpy()
    metrics = {
        "loss": total_loss / max(1, total_nodes),
        "mse": float(mean_squared_error(targets, predictions)),
        "r2": float(r2_score(targets, predictions)),
    }
    return metrics


def plot_metrics(history, save_path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE loss", color="tab:red")
    ax1.plot(epochs, history["train_loss"], color="tab:red", label="Train")
    ax1.plot(epochs, history["val_loss"], color="tab:red", linestyle="--", label="Valid")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("R2", color="tab:blue")
    ax2.plot(epochs, history["train_r2"], color="tab:blue", label="Train R2")
    ax2.plot(epochs, history["val_r2"], color="tab:blue", linestyle="--", label="Valid R2")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.legend(loc="upper right")

    plt.title("Pretraining Metrics")
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train MET for atomic charge prediction.")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--data_manifest", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_path", type=str, default="best_comenet_model.pth")
    parser.add_argument("--metrics_json", type=str, default="pretrain_metrics.json")
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--middle_channels", type=int, default=256)
    parser.add_argument("--atom_embedding_dim", type=int, default=128)
    parser.add_argument("--num_spherical", type=int, default=5)
    parser.add_argument("--num_radial", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=1)
    parser.add_argument("--transformer_heads_z", type=int, default=1)
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_random_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    start_time = time.time()
    dataset = MoleculeDataset(root=args.data_root, manifest_path=args.data_manifest)
    generator = torch.Generator().manual_seed(args.seed)
    train_size = int((1 - args.val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ComENetAutoEncoder(
        cutoff=args.cutoff,
        num_layers=args.num_layers,
        hidden_channels=args.hidden_channels,
        middle_channels=args.middle_channels,
        atom_embedding_dim=args.atom_embedding_dim,
        out_channels=1,
        num_radial=args.num_radial,
        num_spherical=args.num_spherical,
        num_output_layers=3,
        transformer_layers=args.transformer_layers,
        nhead_z=args.transformer_heads_z,
        device=str(device),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=20)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_r2": [], "val_r2": []}

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(model, device, train_loader, criterion, optimizer)
        val_metrics = run_epoch(model, device, val_loader, criterion)
        scheduler.step(val_metrics["r2"])

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_r2"].append(train_metrics["r2"])
        history["val_r2"].append(val_metrics["r2"])

        print(
            f"train_loss={train_metrics['loss']:.6f} train_r2={train_metrics['r2']:.4f} "
            f"val_loss={val_metrics['loss']:.6f} val_r2={val_metrics['r2']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_path = ensure_parent_dir(args.save_path)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cutoff": args.cutoff,
                    "num_layers": args.num_layers,
                    "hidden_channels": args.hidden_channels,
                    "middle_channels": args.middle_channels,
                    "atom_embedding_dim": args.atom_embedding_dim,
                    "out_channels": 1,
                    "num_radial": args.num_radial,
                    "num_spherical": args.num_spherical,
                    "num_output_layers": 3,
                    "transformer_layers": args.transformer_layers,
                    "nhead_z": args.transformer_heads_z,
                    "metrics": {"train": train_metrics, "valid": val_metrics},
                    "seed": args.seed,
                },
                save_path,
            )
            print(f"Saved best model to {save_path}")

    plot_metrics(history, Path("training_plot.png"))

    elapsed = time.time() - start_time
    save_json(
        args.metrics_json,
        {
            "history": history,
            "best_val_loss": best_val_loss,
            "elapsed_seconds": elapsed,
            "save_path": str(Path(args.save_path).resolve()),
            "data_root": args.data_root,
            "data_manifest": args.data_manifest,
            "seed": args.seed,
        },
    )
    print(f"Training finished in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
