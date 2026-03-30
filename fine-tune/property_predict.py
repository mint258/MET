#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from FineTunedModel import FineTunedModel
from dataset_finetune import MoleculeDataset
from fine_tune_training import aggregate_conformer_predictions, custom_collate_fn_factory
from met_utils import ensure_parent_dir, resolve_device, save_json


def load_finetuned_model(checkpoint_path: str, device: torch.device) -> Tuple[FineTunedModel, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = FineTunedModel(
        pretrained_checkpoint_path=checkpoint_path,
        device=device,
        molecular_transformer_args=checkpoint["molecular_transformer_args"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def evaluate(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    true_batches: List[torch.Tensor] = []
    pred_batches: List[torch.Tensor] = []
    with torch.no_grad():
        for batch_data, targets, sample_index in tqdm(loader, desc="Evaluating"):
            batch_data = batch_data.to(device)
            sample_index = sample_index.to(device)
            predictions = model(batch_data)
            predictions = aggregate_conformer_predictions(predictions, sample_index, num_samples=targets.shape[0])
            true_batches.append(targets.cpu())
            pred_batches.append(predictions.cpu())
    return torch.cat(true_batches, dim=0).numpy(), torch.cat(pred_batches, dim=0).numpy()


def per_property_metrics(y_true: np.ndarray, y_pred: np.ndarray, property_names: List[str]) -> Dict[str, Dict[str, float]]:
    metrics = {}
    for idx, property_name in enumerate(property_names):
        true_values = y_true[:, idx]
        pred_values = y_pred[:, idx]
        metrics[property_name] = {
            "mae": float(mean_absolute_error(true_values, pred_values)),
            "rmse": float(np.sqrt(mean_squared_error(true_values, pred_values))),
            "r2": float(r2_score(true_values, pred_values)),
        }
    return metrics


def plot_scatter(true_values: np.ndarray, pred_values: np.ndarray, property_name: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_values, pred_values, alpha=0.7, edgecolors="white", linewidths=0.8)
    min_val = min(true_values.min(), pred_values.min())
    max_val = max(true_values.max(), pred_values.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(property_name)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned MET checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_data_root", default=None)
    parser.add_argument("--test_data_manifest", default=None)
    parser.add_argument("--target_property", nargs="*", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plot_dir", default="property_plots")
    parser.add_argument("--metrics_json", default="property_metrics.json")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_conformers", type=int, default=None)
    parser.add_argument("--conformer_seed", type=int, default=None)
    parser.add_argument("--disable_uff_optimization", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model, checkpoint = load_finetuned_model(args.checkpoint, device)
    data_config = checkpoint.get("data_config", {})
    num_conformers = args.num_conformers if args.num_conformers is not None else data_config.get("num_conformers", 1)
    conformer_seed = args.conformer_seed if args.conformer_seed is not None else data_config.get("conformer_seed", 2024)
    use_uff_optimization = data_config.get("use_uff_optimization", True)
    if args.disable_uff_optimization:
        use_uff_optimization = False

    dataset = MoleculeDataset(
        root=args.test_data_root,
        manifest_path=args.test_data_manifest,
        num_conformers=num_conformers,
        conformer_seed=conformer_seed,
        use_uff_optimization=use_uff_optimization,
    )
    property_names = args.target_property or checkpoint.get("target_properties", [])
    if not property_names:
        raise ValueError("No target properties were provided and none were stored in the checkpoint.")
    property_names = [dataset.resolve_property_name(name) for name in property_names]
    target_indices = [dataset.property_to_index[name] for name in property_names]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn_factory(target_indices),
    )

    y_true, y_pred = evaluate(model, loader, device)
    metrics = per_property_metrics(y_true, y_pred, property_names)

    plot_dir = Path(args.plot_dir)
    for idx, property_name in enumerate(property_names):
        plot_scatter(y_true[:, idx], y_pred[:, idx], property_name, plot_dir / f"{property_name}.png")

    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "target_properties": property_names,
        "metrics": metrics,
        "test_data_root": args.test_data_root,
        "test_data_manifest": args.test_data_manifest,
        "data_config": {
            "num_conformers": num_conformers,
            "conformer_seed": conformer_seed,
            "use_uff_optimization": use_uff_optimization,
        },
    }
    save_json(args.metrics_json, payload)
    for name, metric in metrics.items():
        print(f"{name}: MAE={metric['mae']:.6f} RMSE={metric['rmse']:.6f} R2={metric['r2']:.4f}")


if __name__ == "__main__":
    main()
