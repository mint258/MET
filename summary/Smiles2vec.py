#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_gnn_transformer.py — GNN backbone (Smiles2Vec) enhanced to align with the EGNN backbone where applicable.
- Adds atom-type embedding (nn.Embedding) instead of treating Z as a raw numeric.
- Adds two-layer MLP with Swish after each GCNConv, with residual connections.
- Adds GraphNorm per GNN layer for stabler training.
- Preserves original CLI and outputs: returns (embeddings, predictions) with per-atom predictions.
- Uses TransformerEncoder with src_key_padding_mask so padding atoms are ignored (like EGNN masking behavior).
- Keeps modules that are EGNN-specific (e.g., torsion/angle encoders) out of scope.
"""

import argparse
import os
import time
import random
from typing import List, Tuple

import numpy as np
import torch
from torch import nn, Tensor
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GraphNorm, global_mean_pool
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

from dataset_without_charge import MoleculeDataset
from rdkit import Chem


# ------------------------
# Utilities
# ------------------------

class Swish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------
# Dataset wrapper: add chirality feature like existing script
# ------------------------

class ChiralMoleculeDataset(MoleculeDataset):
    """
    Extend MoleculeDataset by appending a per-atom chirality feature to data.x.
    Keeps the original interface: x[:,0] is Z index (long), we append chirality as float.
    """
    def get(self, idx):
        data = super().get(idx)
        # Append a chirality feature per atom using chiral InChI if present
        if hasattr(data, "chiral_inchi"):
            try:
                mol = Chem.MolFromInchi(data.chiral_inchi, sanitize=True)
                if mol is None:
                    raise ValueError("MolFromInchi failed")
                # Map atom chirality to {0,1,2}
                ch = []
                for atom in mol.GetAtoms():
                    tag = atom.GetChiralTag()
                    if tag == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
                        ch.append(0.0)
                    elif tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
                        ch.append(1.0)
                    elif tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
                        ch.append(2.0)
                    else:
                        ch.append(0.0)
                # If lengths mismatch (e.g., different atom ordering), fallback to zeros
                if len(ch) != data.x.size(0):
                    ch = [0.0] * data.x.size(0)
                chirality_full = torch.tensor(ch, dtype=torch.float32).view(-1, 1)
                data.x = torch.cat([data.x, chirality_full], dim=1)
            except Exception:
                chirality_full = torch.zeros((data.x.size(0), 1), dtype=torch.float32)
                data.x = torch.cat([data.x, chirality_full], dim=1)
        else:
            chirality_full = torch.zeros((data.x.size(0), 1), dtype=torch.float32)
            data.x = torch.cat([data.x, chirality_full], dim=1)
        return data


# ------------------------
# GNN: add atom embedding, GraphNorm, residual MLP block
# ------------------------

class GNN(nn.Module):
    """
    A plain GCN backbone enhanced to mirror EGNN niceties that are not EGNN-specific:
    - Atom type embedding for Z (x[:,0] is an integer index)
    - Linear projection for extra scalar node features (e.g., chirality) to hidden_dim
    - Per-layer GraphNorm and residual MLP (two Linear+Swish) after each GCNConv
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, middle_dim: int, num_atom_types: int = 95):
        super().__init__()
        assert input_dim >= 1, "input_dim must be >=1 (includes Z index)"
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding for atom type (Z)
        self.z_embed = nn.Embedding(num_atom_types, hidden_dim)

        # Project the remaining scalar features (input_dim-1) to hidden_dim
        self.has_extra = (input_dim - 1) > 0
        if self.has_extra:
            self.extra_proj = nn.Linear(input_dim - 1, hidden_dim)
        else:
            self.extra_proj = None

        # GCN layers; keep hidden_dim across all layers
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([GraphNorm(hidden_dim) for _ in range(num_layers)])
        self.mlps  = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, middle_dim),
                Swish(),
                nn.Linear(middle_dim, hidden_dim),
                Swish(),
            ) for _ in range(num_layers)
        ])
        self.activation = Swish()

        # Optional output projection for node embeddings (kept identical dimension)
        self.out_lin = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.z_embed.weight, mean=0.0, std=0.02)
        if self.extra_proj is not None:
            nn.init.xavier_uniform_(self.extra_proj.weight); nn.init.zeros_(self.extra_proj.bias)
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()
        for mlp in self.mlps:
            for m in mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        # x[:,0] is Z index (long). Others (if any) are floats.
        z = x[:, 0].long()
        z_emb = self.z_embed(z)  # [N, hidden_dim]
        if self.has_extra:
            extra = x[:, 1:].float()
            extra = self.extra_proj(extra)  # [N, hidden_dim]
            h = z_emb + extra
        else:
            h = z_emb

        for i in range(self.num_layers):
            h_res = h
            h = self.convs[i](h, edge_index)        # message passing
            h = self.norms[i](h, batch)             # per-graph normalization
            h = self.activation(h)
            h = self.mlps[i](h) + h_res             # residual MLP
        h = self.out_lin(h)
        return h   # [num_nodes, hidden_dim]


# ------------------------
# GNN + Transformer head (per-atom prediction)
# ------------------------

class GNNTransformerWithEmbedding(nn.Module):
    """
    Keeps the original interface, but internally:
    - use the enhanced GNN above;
    - build key_padding_mask for Transformer;
    - only compute outputs for valid atoms (unpadded).
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_gnn_layers: int,
                 num_transformer_layers: int, nhead: int, dim_feedforward: int,
                 dropout: float, output_dim: int, middle_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # GNN backbone
        self.gnn = GNN(input_dim, hidden_dim, num_gnn_layers, middle_dim)

        # Transformer
        enc_layer = TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu',
            batch_first=False
        )
        self.transformer = TransformerEncoder(enc_layer, num_layers=num_transformer_layers)

        # Per-atom output head
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Init
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 1) GNN node embeddings
        node_embeddings = self.gnn(x, edge_index, batch)  # [num_nodes, hidden_dim]

        # 2) Pack to padded sequence: [S_max, B, D]
        sizes = [(batch == i).sum().item() for i in range(data.num_graphs)]
        splits = torch.split(node_embeddings, sizes)
        S_max = max(s.size(0) for s in splits) if len(splits) > 0 else 1
        B = data.num_graphs
        D = self.hidden_dim

        seq = node_embeddings.new_zeros((S_max, B, D))
        pad_mask = torch.ones((B, S_max), dtype=torch.bool, device=node_embeddings.device)  # True == pad
        for i, s in enumerate(splits):
            seq[:s.size(0), i, :] = s
            pad_mask[i, :s.size(0)] = False  # not padded

        # 3) Transformer with key padding mask
        # TransformerEncoder expects src_key_padding_mask of shape [B, S_max] with True on pads
        tr_out = self.transformer(seq, src_key_padding_mask=pad_mask)  # [S_max, B, D]

        # 4) Strip paddings and compute per-atom predictions
        tr_out_bt = tr_out.transpose(0, 1)  # [B, S_max, D]
        valid_mask = ~pad_mask              # [B, S_max]
        feats = tr_out_bt[valid_mask]       # [N_total_valid, D]
        preds = self.fc(feats)              # [N_total_valid, output_dim]

        # Return raw node embeddings from GNN (for compatibility) and per-atom predictions
        return node_embeddings, preds


# ------------------------
# Training / Validation
# ------------------------

def train(model, device, loader, optimizer, criterion) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_preds = []
    all_tgts  = []
    for data in tqdm(loader, desc="Training", leave=False):
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, pred = model(data)
        loss = criterion(pred, data.y.view(-1, model.output_dim))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        all_preds.append(pred.detach().cpu())
        all_tgts.append(data.y.view(-1, model.output_dim).detach().cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy().flatten()
    all_tgts  = torch.cat(all_tgts , dim=0).numpy().flatten()
    r2 = r2_score(all_tgts, all_preds) if len(all_tgts) > 1 else 0.0
    return total_loss / len(loader.dataset), r2


@torch.no_grad()
def validate(model, device, loader, criterion) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_tgts  = []
    for data in tqdm(loader, desc="Validation", leave=False):
        data = data.to(device)
        _, pred = model(data)
        loss = criterion(pred, data.y.view(-1, model.output_dim))
        total_loss += loss.item() * data.num_graphs
        all_preds.append(pred.detach().cpu())
        all_tgts.append(data.y.view(-1, model.output_dim).detach().cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy().flatten()
    all_tgts  = torch.cat(all_tgts , dim=0).numpy().flatten()
    r2 = r2_score(all_tgts, all_preds) if len(all_tgts) > 1 else 0.0
    return total_loss / len(loader.dataset), r2


def plot_metrics(train_losses: List[float], val_losses: List[float],
                 train_r2: List[float], val_r2: List[float], save_path: str):
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, val_losses,   label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_r2, label='Train R²')
    ax2.plot(epochs, val_r2,   label='Val R²')
    ax2.set_ylabel('R²')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# ------------------------
# Main
# ------------------------

from tqdm import tqdm

def main():
    set_seed(42)

    parser = argparse.ArgumentParser(description="Train GNN + Transformer for Atomic Charge Prediction")
    parser.add_argument('--data_root', type=str, required=True, help='Path to the dataset root directory containing .xyz files')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--save_path', type=str, default='best_gnn_transformer_model.pth', help='Path to save the best model')
    parser.add_argument('--middle_dim', type=int, default=256, help='Hidden dimension for the two-layer linear network after each GNN layer')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden embedding size for GNN and Transformer')
    parser.add_argument('--num_gnn_layers', type=int, default=4, help='Number of GNN layers')
    parser.add_argument('--num_transformer_layers', type=int, default=2, help='Number of Transformer layers')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads in Transformer')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='Feedforward network dimension in Transformer')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension (e.g., 1 for charge prediction)')
    parser.add_argument('--plot_path', type=str, default='training_plot.png', help='Path to save the training plot')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training ("cuda" or "cpu")')
    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    dataset = ChiralMoleculeDataset(root=args.data_root)

    # Train/Val split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"Total samples: {len(dataset)}, Training: {len(train_dataset)}, Validation: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

    # Model
    input_dim = 2  # (Z index) + (chirality)
    model = GNNTransformerWithEmbedding(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_transformer_layers=args.num_transformer_layers,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        output_dim=args.output_dim,
        middle_dim=args.middle_dim
    ).to(device)

    # Optim & Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_r2 = float('-inf')
    train_losses: List[float] = []
    val_losses:   List[float] = []
    train_r2_hist: List[float] = []
    val_r2_hist:   List[float] = []

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        tr_loss, tr_r2 = train(model, device, train_loader, optimizer, criterion)
        va_loss, va_r2 = validate(model, device, val_loader, criterion)
        print(f"Train Loss: {tr_loss:.8f}, Train R²: {tr_r2:.6f} | Val Loss: {va_loss:.8f}, Val R²: {va_r2:.6f}")

        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_r2_hist.append(tr_r2);  val_r2_hist.append(va_r2)

        if va_r2 > best_val_r2:
            best_val_r2 = va_r2
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved Best Model with Val R²: {va_r2:.6f}")

    print("\nTraining Complete.")
    print(f"Best Validation R²: {best_val_r2:.6f}")
    plot_metrics(train_losses, val_losses, train_r2_hist, val_r2_hist, save_path=args.plot_path)
    print(f"训练时间: {time.time() - start_time:.2f} 秒")


if __name__ == '__main__':
    main()
