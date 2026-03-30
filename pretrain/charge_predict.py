# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import periodictable
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from comenet4charge import ComENetAutoEncoder
from dataset_without_charge import MoleculeDataset
from met_utils import build_pretrained_model, ensure_parent_dir, resolve_device, save_json


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model, _ = build_pretrained_model(ComENetAutoEncoder, checkpoint, device)
    model.eval()
    return model


def predict(model, device, loader, charges_dir: Path) -> Tuple[List[float], List[float]]:
    charges_dir.mkdir(parents=True, exist_ok=True)
    element_dict_rev = {idx: element.symbol for idx, element in enumerate(periodictable.elements)}

    all_true = []
    all_pred = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            filenames = batch.filename
            batch = batch.to(device)
            true_charges = batch.y.view(-1).cpu().numpy()
            embeddings, predictions = model(batch)
            predictions = predictions.view(-1).cpu().numpy()
            all_true.extend(true_charges.tolist())
            all_pred.extend(predictions.tolist())

            node_counts = batch.batch.bincount().cpu().numpy()
            start = 0
            for count, filename in zip(node_counts, filenames):
                end = start + count
                molecule_preds = predictions[start:end]
                molecule_x = batch.x[start:end, 0].cpu().numpy()
                molecule_pos = batch.pos[start:end].cpu().numpy()
                start = end

                output_path = charges_dir / f"{Path(filename).stem}_charges.xyz"
                with output_path.open("w", encoding="utf-8") as f:
                    f.write(f"{len(molecule_preds)}\n")
                    f.write("predicted_charges\n")
                    for atom_type_idx, pos, charge in zip(molecule_x, molecule_pos, molecule_preds):
                        atom_type = element_dict_rev.get(int(atom_type_idx), "Unknown")
                        f.write(f"{atom_type} {pos[0]} {pos[1]} {pos[2]} {charge}\n")

    return all_true, all_pred


def plot_results(true_charges, predicted_charges, plot_path: Path):
    mse = mean_squared_error(true_charges, predicted_charges)
    mae = mean_absolute_error(true_charges, predicted_charges)
    r2 = r2_score(true_charges, predicted_charges)

    plt.figure(figsize=(8, 8))
    plt.scatter(true_charges, predicted_charges, alpha=0.5, edgecolors="w", s=40)
    min_val = min(min(true_charges), min(predicted_charges))
    max_val = max(max(true_charges), max(predicted_charges))
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("True charge")
    plt.ylabel("Predicted charge")
    plt.title("Atomic Charge Prediction")
    plt.text(
        0.05,
        0.95,
        f"R2={r2:.4f}\nMSE={mse:.6f}\nMAE={mae:.6f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return {"mse": float(mse), "mae": float(mae), "r2": float(r2)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained MET checkpoint on atomic charges.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--test_data_root", type=str, default=None)
    parser.add_argument("--test_data_manifest", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--charges_dir", type=str, default="charges")
    parser.add_argument("--plot_path", type=str, default="charge_predictions_scatter.png")
    parser.add_argument("--metrics_json", type=str, default="charge_metrics.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model = load_model(args.checkpoint_path, device)
    dataset = MoleculeDataset(root=args.test_data_root, manifest_path=args.test_data_manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    true_charges, predicted_charges = predict(model, device, loader, Path(args.charges_dir))
    metrics = plot_results(true_charges, predicted_charges, Path(args.plot_path))
    save_json(
        args.metrics_json,
        {
            "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
            "test_data_root": args.test_data_root,
            "test_data_manifest": args.test_data_manifest,
            "metrics": metrics,
            "plot_path": str(Path(args.plot_path).resolve()),
            "charges_dir": str(Path(args.charges_dir).resolve()),
        },
    )
    print(f"R2={metrics['r2']:.4f} MSE={metrics['mse']:.6f} MAE={metrics['mae']:.6f}")


if __name__ == "__main__":
    main()
