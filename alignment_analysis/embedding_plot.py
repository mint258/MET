#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.loader import DataLoader
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from graph_compat import scatter_mean
import pandas as pd

group_color_map = {
    "Fluoro":    "#96cac1",  #
    "Nitro":     "#f6f6bc",  #
    "Cyano":     "#c1bed6",  #
    "Carboxy":   "#ea8e83",  #
    "Hydroxy":   "#8aafc9",  #
    "Benzene":   "#eab375",  #
    "Alkyne":    "#afcf78",  #
    #
    #
}

#
from comenet4charge import ComENetAutoEncoder
from dataloader import MoleculeDataset
import matplotlib as mpl

#
sns.set_context("notebook", font_scale=1.2)
mpl.rcParams.update({
    #
    'figure.titlesize': 16,
    #
    'axes.titlesize': 18,
    #
    'axes.labelsize': 18,
    #
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    #
    'legend.fontsize': 14,
    'legend.title_fontsize': 16,
    #
    'legend.markerscale': 1.2,
})
mpl.rcParams['svg.fonttype'] = 'none'

#########################################
#
#########################################
def mol_from_identifier(text: str):
    """Helper documentation is summarized in the manuscript and README."""
    if not text or text == "Unknown":
        return None, "Unknown"
    try:
        if text.startswith(("InChI=", "1S/")):          #
            mol = Chem.MolFromInchi(text)
        else:                                           #
            mol = Chem.MolFromSmiles(text)
        if mol is None:
            return None, "Unknown"
        smiles = Chem.MolToSmiles(mol, canonical=True)
        return mol, smiles
    except Exception:
        return None, "Unknown"

def get_functional_group(smiles):
    """Helper documentation is summarized in the manuscript and README."""
    if smiles == 'Unknown':
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 'Alkane'
    groups = [
        ("Benzene", Chem.MolFromSmarts("c1ccccc1")),
        ("Fluoro", Chem.MolFromSmarts("F")),
        ("Nitro", Chem.MolFromSmarts("[N+](=O)[O-]")),
        ("Cyano", Chem.MolFromSmarts("C#N")),
        ("Carboxy", Chem.MolFromSmarts("C(=O)O")),
        # ("Alkene", Chem.MolFromSmarts("C=C")),
        ("Alkyne", Chem.MolFromSmarts("C#C")),
        ("Hydroxy", Chem.MolFromSmarts("CO")),
        # ("Alkane", Chem.MolFromSmarts("CC"))
    ]
    # groups = [
    #     ("Azetidine", Chem.MolFromSmarts("C1CNC1")),
    #     ("Epoxide", Chem.MolFromSmarts("C1CO1")),
    #     ("Cyclobutane", Chem.MolFromSmarts("C1CCC1")),
    #     ("Oxetane", Chem.MolFromSmarts("C1CCO1")),
    #     ("Pyrrole", Chem.MolFromSmarts("c1[nH]ccc1")),
    #     ("Cyclopentene", Chem.MolFromSmarts("C1CC=CC1")),
    #     ("Oxete", Chem.MolFromSmarts("C1CC=CO1"))
    # ]
    for group_name, pattern in groups:
        if pattern is None:
            continue
        if mol.HasSubstructMatch(pattern):
            return group_name
    return None

#########################################
#
#########################################
def load_latent_vectors(model, loader, device):
    """Helper documentation is summarized in the manuscript and README."""
    model.eval()
    latent_list = []
    smiles_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting latent vectors"):
            batch = batch.to(device)
            if not hasattr(batch, "batch") or batch.batch is None:
                batch.batch = torch.zeros(batch.pos.size(0), dtype=torch.long, device=batch.pos.device)
            atomic_embeddings, _ = model(batch)  # atomic_embeddings: [total_nodes, hidden_dim]
            molecule_embeddings = scatter_mean(atomic_embeddings, batch.batch, dim=0)
            latent_list.append(molecule_embeddings.cpu().numpy())
            for text in batch.chiral_inchi:
                _, smiles = mol_from_identifier(text)
                smiles_list.append(smiles)
    latent_vectors = np.concatenate(latent_list, axis=0)
    return latent_vectors, smiles_list

#########################################
#
#########################################
def plot_alignment(tsne_results, functional_labels, db_index, save_path=None):
    fig = plt.figure(figsize=(10, 8))
    df = pd.DataFrame({
        'x': tsne_results[:, 0],
        'y': tsne_results[:, 1],
        'label': functional_labels
    })
    ax = fig.gca()
    # ax.set_facecolor('#E2EEE1')
    #
    labeled_data = df[df['label'] != "Unlabeled"]

    #
    if not labeled_data.empty:
        sns.scatterplot(
            data=labeled_data,
            x='x', y='y',
            hue='label',           #
            hue_order=group_color_map.keys(), #
            palette=group_color_map, #
            alpha=0.75, s=50, edgecolor='none'
        )

    #
    # unlabeled_data = df[df['label'] == "Unlabeled"]
    # if not unlabeled_data.empty:
    #     plt.scatter(
    #         unlabeled_data['x'],
    #         unlabeled_data['y'],
    #         c='lightgray',
    #         alpha=0.5,
    #         s=60,
    #         label="Unlabeled"
    #     )

    title_txt = "Alignment Analysis"
    if not np.isnan(db_index):
        title_txt += f" (DB={db_index:.2f})"
        
    # plt.title(title_txt, fontsize=16)
    plt.xlabel('')
    plt.ylabel('')
    ax.legend(
        title='Functional Group',
        loc='best',        #
        frameon=True,             #
        framealpha=0.9            #
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Alignment plot saved to {save_path}")

#########################################
#
#
#
#########################################
def plot_uniformity(tsne_results,
                    bin_size=5,
                    save_path_ring=None,
                    save_path_line=None):
    """Helper documentation is summarized in the manuscript and README."""
    #
    center = np.mean(tsne_results, axis=0)
    angles_deg = (np.degrees(np.arctan2(
                    tsne_results[:, 1] - center[1],
                    tsne_results[:, 0] - center[0])) + 360) % 360

    #
    bin_edges = np.arange(0, 360 + bin_size, bin_size)
    counts, _ = np.histogram(angles_deg, bins=bin_edges)
    if counts.max() > 0:
        norm_counts = counts / counts.max()
    else:
        norm_counts = counts

    #
    fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    theta_edges = np.deg2rad(bin_edges)           # len = n_bins+1
    r_edges = np.array([0.8, 1.0])                #
    Theta, R = np.meshgrid(theta_edges, r_edges)  # Theta shape (2, n_bins+1)
    Z = norm_counts.reshape(1, -1)                # shape (1, n_bins)
    pcm = ax1.pcolormesh(Theta, R, Z, cmap='viridis', shading='auto')
    ax1.set_xticklabels([]); ax1.set_yticklabels([])
    ax1.set_title("Uniformity 0–360° (bin = {}°)".format(bin_size), va='bottom')
    cbar = fig1.colorbar(pcm, ax=ax1, pad=0.1); cbar.set_label("Normalized frequency")
    if save_path_ring: plt.savefig(save_path_ring, dpi=300)

    #
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    try:
        from scipy.interpolate import make_interp_spline
        xnew = np.linspace(bin_centers.min(), bin_centers.max(), 300)
        ynew = make_interp_spline(bin_centers, norm_counts, k=3)(xnew)
    except ImportError:
        xnew, ynew = bin_centers, norm_counts

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(xnew, ynew, lw=2)
    ax2.set_xlabel("Angle (degrees)")
    ax2.set_xlim(0, 360)
    ax2.set_title("Normalized angle distribution (0–360°)")
    ax2.set_yticks([])
    if save_path_line: plt.savefig(save_path_line, dpi=300)
    
def main():
    parser = argparse.ArgumentParser(description="Latent Visualization from ComENet with DataLoader")
    parser.add_argument('--xyz_dir', type=str, required=True,
                        help="Directory containing .xyz molecule files")
    parser.add_argument('--model_path', type=str, required=False,
                        help="Path to the pre-trained ComENet model checkpoint")
    parser.add_argument('--device', type=str, default='cpu', help="Device to run the model")
    parser.add_argument('--perplexity', type=float, default=30, help="Perplexity for t-SNE")
    parser.add_argument('--output_dir', type=str, default='latent_vis_results', help="Directory to save plots")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument('--seed', type=float, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    #
    dataset = MoleculeDataset(root=args.xyz_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    #
    model = ComENetAutoEncoder(
        cutoff=8.0,
        num_layers=4,
        hidden_channels=256,
        middle_channels=256,
        out_channels=1,            #
        atom_embedding_dim=128,
        num_radial=8,
        num_spherical=5,
        num_output_layers=3,
        transformer_layers=1,
        nhead_z=1,
        device=args.device
    )
    model = model.to(device)
    if args.model_path and os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            pretrained_dict = checkpoint["model_state_dict"]
        else:
            pretrained_dict = checkpoint
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded model weights from {args.model_path} (filtered).")
    else:
        print("Using randomly initialized model (for demo purposes)")

    #
    latent_vectors, smiles_list = load_latent_vectors(model, loader, device)
    print(f"Extracted latent vectors with shape: {latent_vectors.shape}")

    #
    functional_labels = [
        get_functional_group(sm) or "Unlabeled"
        for sm in smiles_list
    ]

    #
    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent_vectors)
    tsne = TSNE(n_components=2, perplexity=args.perplexity, learning_rate=200, random_state=args.seed, init='pca')
    tsne_results = tsne.fit_transform(latent_scaled)

    #
    labeled_mask = [lab != "Unlabeled" for lab in functional_labels]
    db_index = np.nan
    if sum(labeled_mask) and len(set(np.array(functional_labels)[labeled_mask])) > 1:
        db_index = davies_bouldin_score(
            tsne_results[labeled_mask], np.array(functional_labels)[labeled_mask]
        )
    if np.isnan(db_index):
        print("Davies-Bouldin index: not computed (fewer than 2 valid clusters)")
    else:
        print(f"Davies-Bouldin index: {db_index:.2f}")

    #
    alignment_plot_path = os.path.join(args.output_dir, 'alignment_analysis.svg')
    plot_alignment(tsne_results, functional_labels, db_index, save_path=alignment_plot_path)

    #
    polar_plot_path = os.path.join(args.output_dir, 'uniformity_polar.svg')
    line_plot_path = os.path.join(args.output_dir, 'uniformity_line.svg')
    plot_uniformity(tsne_results, bin_size=5, save_path_ring=polar_plot_path, save_path_line=line_plot_path)

if __name__ == '__main__':
    main()
