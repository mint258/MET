# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Batch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from FineTunedModel import FineTunedModel
from dataset_finetune import MoleculeDataset
from graph_compat import scatter_mean
from met_utils import ensure_parent_dir, resolve_device, save_json, set_random_seed


def custom_collate_fn_factory(target_indices: List[int]):
    def _fn(batch):
        data_kept = []
        targets = []
        sample_index = []
        for sample in batch:
            values = sample["scalar_props"][target_indices]
            if torch.isnan(values).any():
                continue
            target_row = len(targets)
            targets.append(values)
            for conformer in sample["conformers"]:
                data_kept.append(conformer)
                sample_index.append(target_row)

        if not data_kept:
            sample = batch[0]
            data_kept = [sample["conformers"][0]]
            targets = [torch.zeros((len(target_indices),), dtype=torch.float)]
            sample_index = [0]

        batch_data = Batch.from_data_list(data_kept)
        target_tensor = torch.stack(targets, dim=0)
        sample_index_tensor = torch.tensor(sample_index, dtype=torch.long)
        return batch_data, target_tensor, sample_index_tensor

    return _fn


def aggregate_conformer_predictions(
    predictions: torch.Tensor,
    sample_index: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    if predictions.dim() == 1:
        predictions = predictions.unsqueeze(-1)
    return scatter_mean(predictions, sample_index, dim=0, dim_size=num_samples)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(r2_score(y_true, y_pred, multioutput="uniform_average"))
    return {"mae": float(mae), "rmse": rmse, "r2": r2}


def run_epoch(model, device, loader, criterion, optimizer=None) -> Tuple[float, Dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    total_graphs = 0
    all_preds = []
    all_targets = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_data, targets, sample_index in tqdm(loader, desc="Training" if is_train else "Validation"):
            batch_data = batch_data.to(device)
            targets = targets.to(device)
            sample_index = sample_index.to(device)

            if optimizer is not None:
                optimizer.zero_grad()

            outputs = model(batch_data)
            pooled_outputs = aggregate_conformer_predictions(outputs, sample_index, num_samples=targets.shape[0])
            loss = criterion(pooled_outputs, targets)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = targets.shape[0]
            running_loss += loss.item() * batch_size
            total_graphs += batch_size
            all_preds.append(pooled_outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())

    predictions = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    metrics = compute_regression_metrics(targets, predictions)
    metrics["loss"] = running_loss / max(1, total_graphs)
    return metrics["loss"], metrics


def freeze_layers(model: FineTunedModel, layer_list: List[Tuple[str, nn.Module]], freeze_up_to_layer: int | None) -> None:
    if freeze_up_to_layer is None:
        for param in model.autoencoder.parameters():
            param.requires_grad = True
        return

    for idx, (name, module) in enumerate(layer_list):
        if idx <= freeze_up_to_layer:
            for param in module.parameters():
                param.requires_grad = False
            print(f"Frozen layer {idx}: {name}")


def plot_training_curves(history: Dict[str, List[float]], output_dir: Path, label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="valid")
    plt.xlabel("Epoch")
    plt.ylabel("MAE loss")
    plt.title(f"Loss Curve: {label}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"loss_curve_{label}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_r2"], label="train")
    plt.plot(epochs, history["val_r2"], label="valid")
    plt.xlabel("Epoch")
    plt.ylabel("R2")
    plt.title(f"R2 Curve: {label}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"r2_curve_{label}.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MET on downstream molecular properties.")
    parser.add_argument("--pretrained_checkpoint_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--data_manifest", type=str, default=None)
    parser.add_argument("--target_property", type=str, required=True, nargs="+")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--min_dim", type=int, default=32)
    parser.add_argument("--num_linear_layers", type=int, default=0)
    parser.add_argument("--max_lr_reductions", type=int, default=5)
    parser.add_argument("--save_model", type=str, default="best_finetuned_model.pth")
    parser.add_argument("--metrics_json", type=str, default="fine_tune_metrics.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot_dir", type=str, default="plots")
    parser.add_argument("--freeze_up_to_layer", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_conformers", type=int, default=1)
    parser.add_argument("--conformer_seed", type=int, default=2024)
    parser.add_argument("--disable_uff_optimization", action="store_true")
    args = parser.parse_args()

    set_random_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    dataset = MoleculeDataset(
        root=args.data_root,
        manifest_path=args.data_manifest,
        num_conformers=args.num_conformers,
        conformer_seed=args.conformer_seed,
        use_uff_optimization=not args.disable_uff_optimization,
    )
    target_properties = [dataset.resolve_property_name(name) for name in args.target_property]
    target_indices = [dataset.property_to_index[name] for name in target_properties]
    print(f"Resolved target properties: {target_properties}")

    generator = torch.Generator().manual_seed(args.seed)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    collate_fn = custom_collate_fn_factory(target_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    ckpt = torch.load(args.pretrained_checkpoint_path, map_location=device)
    atom_embedding_dim = ckpt.get("atom_embedding_dim", ckpt["model_state_dict"]["encoder.0.weight"].shape[0])
    molecular_transformer_args = {
        "atom_embedding_dim": atom_embedding_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "output_dim": len(target_properties),
        "num_linear_layers": args.num_linear_layers,
        "min_dim": args.min_dim,
    }

    model = FineTunedModel(
        pretrained_checkpoint_path=args.pretrained_checkpoint_path,
        device=device,
        molecular_transformer_args=molecular_transformer_args,
    ).to(device)

    layer_list = []
    for name, module in model.autoencoder.named_children():
        if isinstance(module, torch.nn.ModuleList):
            for idx, sub_module in enumerate(module):
                layer_list.append((f"{name}.{idx}", sub_module))
        else:
            layer_list.append((name, module))
    freeze_layers(model, layer_list, args.freeze_up_to_layer)

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    criterion = nn.L1Loss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=20)

    history = {"train_loss": [], "val_loss": [], "train_r2": [], "val_r2": []}
    best_val_r2 = -math.inf
    best_model_score = -math.inf
    lr_reduce_count = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        _, train_metrics = run_epoch(model, device, train_loader, criterion, optimizer=optimizer)
        _, val_metrics = run_epoch(model, device, val_loader, criterion)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_r2"].append(train_metrics["r2"])
        history["val_r2"].append(val_metrics["r2"])

        scheduler_metric = val_metrics["r2"] if math.isfinite(val_metrics["r2"]) else -val_metrics["loss"]
        old_lrs = [group["lr"] for group in optimizer.param_groups]
        scheduler.step(scheduler_metric)
        new_lrs = [group["lr"] for group in optimizer.param_groups]
        if any(new_lr < old_lr for old_lr, new_lr in zip(old_lrs, new_lrs)):
            lr_reduce_count += 1
            print(f"Learning rate reduced {lr_reduce_count} time(s).")
            if lr_reduce_count > args.max_lr_reductions:
                print("Reached the maximum number of LR reductions. Stopping early.")
                break

        print(
            f"train_loss={train_metrics['loss']:.6f} train_r2={train_metrics['r2']:.4f} "
            f"val_loss={val_metrics['loss']:.6f} val_r2={val_metrics['r2']:.4f}"
        )

        model_score = val_metrics["r2"] if math.isfinite(val_metrics["r2"]) else -val_metrics["loss"]
        if model_score > best_model_score:
            best_model_score = model_score
            best_val_r2 = val_metrics["r2"]
            checkpoint_payload = {
                "model_state_dict": model.state_dict(),
                "pretrained_checkpoint_path": args.pretrained_checkpoint_path,
                "molecular_transformer_args": molecular_transformer_args,
                "target_properties": target_properties,
                "data_config": {
                    "num_conformers": args.num_conformers,
                    "conformer_seed": args.conformer_seed,
                    "use_uff_optimization": not args.disable_uff_optimization,
                },
                "metrics": {"train": train_metrics, "valid": val_metrics},
                "seed": args.seed,
            }
            save_path = ensure_parent_dir(args.save_model)
            torch.save(checkpoint_payload, save_path)
            print(f"Saved best model to {save_path}")

    plot_training_curves(history, Path(args.plot_dir), "_".join(target_properties))

    metrics_payload = {
        "target_properties": target_properties,
        "history": history,
        "best_val_r2": best_val_r2,
        "save_model": str(Path(args.save_model).resolve()),
        "data_root": args.data_root,
        "data_manifest": args.data_manifest,
        "data_config": {
            "num_conformers": args.num_conformers,
            "conformer_seed": args.conformer_seed,
            "use_uff_optimization": not args.disable_uff_optimization,
        },
        "seed": args.seed,
    }
    save_json(args.metrics_json, metrics_payload)


if __name__ == "__main__":
    main()
