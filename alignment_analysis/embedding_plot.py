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
from torch_scatter import scatter_mean
import pandas as pd

group_color_map = {
    "Fluoro":    "#96cac1",  # 蓝绿色
    "Nitro":     "#f6f6bc",  # 黄色
    "Cyano":     "#c1bed6",  # 紫色
    "Carboxy":   "#ea8e83",  # 红色
    "Hydroxy":   "#8aafc9",  # 蓝色
    "Benzene":   "#eab375",  # 橙色
    "Alkyne":    "#afcf78",  # 绿色
    # "Unlabeled": "#cccccc"   # 灰色 (用于未标记的数据)
    # 如果您启用其他官能团，也在这里为它们添加颜色
}

# 导入 ComENet 模型和数据集
from comenet4charge import ComENetAutoEncoder
from dataloader import MoleculeDataset
import matplotlib as mpl

# Seaborn 上下文：四选一 paper、notebook、talk、poster，font_scale 控制相对大小
sns.set_context("notebook", font_scale=1.2)
mpl.rcParams.update({
    # 整个 figure 标题
    'figure.titlesize': 16,
    # 坐标系（axes）标题
    'axes.titlesize': 18,
    # 坐标轴标签（xlabel / ylabel）
    'axes.labelsize': 18,
    # 刻度标签
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    # 图例文字与标题
    'legend.fontsize': 14,
    'legend.title_fontsize': 16,
    # 图例 markersize（可选）
    'legend.markerscale': 1.2,
})
mpl.rcParams['svg.fonttype'] = 'none'

#########################################
# 官能团检测函数
#########################################
def mol_from_identifier(text: str):
    """
    兼容 SMILES 与 InChI 两种字符串，返回 (mol, canonical_smiles)。
    解析失败时 (None, 'Unknown')
    """
    if not text or text == "Unknown":
        return None, "Unknown"
    try:
        if text.startswith(("InChI=", "1S/")):          # InChI 串
            mol = Chem.MolFromInchi(text)
        else:                                           # 当作 SMILES
            mol = Chem.MolFromSmiles(text)
        if mol is None:
            return None, "Unknown"
        smiles = Chem.MolToSmiles(mol, canonical=True)
        return mol, smiles
    except Exception:
        return None, "Unknown"

def get_functional_group(smiles):
    """
    根据 SMILES 返回分子的官能团标签，
    只检测以下官能团（按顺序优先）：  
      Benzene、Furan、Pyridine、Pyrrole、Nitrile、Ester、Ketone、Alcohol
    若分子同时含有多个，则返回最前面的；若不含任何，则返回 None。
    """
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
# 利用 DataLoader 处理 batch 数据，提取潜向量和 SMILES
#########################################
def load_latent_vectors(model, loader, device):
    """
    利用 DataLoader 对整个数据集分批处理，从模型获得每个分子的潜空间向量（采用 scatter_mean 聚合原子嵌入），
    同时提取每个分子的 SMILES（通过 data.chiral_inchi 转换）。

    返回：
       latent_vectors: numpy 数组，形状为 [n_molecules, latent_dim]
       smiles_list: list，每个元素为分子的 SMILES 字符串
    """
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
# 绘制对齐性图：二维散点图，各点颜色为官能团标签，并在标题中显示 DB 指数
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
    # 筛选出有标签的数据点
    labeled_data = df[df['label'] != "Unlabeled"]

    # 绘制已标注的数据点
    if not labeled_data.empty:
        sns.scatterplot(
            data=labeled_data,
            x='x', y='y',
            hue='label',           # 根据 'label' 列进行着色
            hue_order=group_color_map.keys(), # 确保图例顺序
            palette=group_color_map, # 2. 在这里使用您定义的颜色映射字典
            alpha=0.75, s=50, edgecolor='none'
        )

    # （可选）如果您也想绘制未标注的点，可以取消下面代码的注释
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
        loc='best',        # 或者 loc='best'
        frameon=True,             # 可选：保留图例背景框
        framealpha=0.9            # 可选：背景透明度
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Alignment plot saved to {save_path}")

#########################################
# 绘制均匀性分析图：
# 1. 在极坐标中画出各小扇区的频次（热频图）
# 2. 绘制折线图展示 0-180° 内每个小扇区的频次
#########################################
def plot_uniformity(tsne_results,
                    bin_size=5,
                    save_path_ring=None,
                    save_path_line=None):
    """
    0–360° 直接统计并绘图，不再复制 0–180° 数据。
    """
    # ---- 角度计算 ----
    center = np.mean(tsne_results, axis=0)
    angles_deg = (np.degrees(np.arctan2(
                    tsne_results[:, 1] - center[1],
                    tsne_results[:, 0] - center[0])) + 360) % 360

    # ---- 柱状统计 ----
    bin_edges = np.arange(0, 360 + bin_size, bin_size)
    counts, _ = np.histogram(angles_deg, bins=bin_edges)
    if counts.max() > 0:
        norm_counts = counts / counts.max()
    else:
        norm_counts = counts

    # ---- 极坐标颜色环 ----
    fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    theta_edges = np.deg2rad(bin_edges)           # len = n_bins+1
    r_edges = np.array([0.8, 1.0])                # 薄环
    Theta, R = np.meshgrid(theta_edges, r_edges)  # Theta shape (2, n_bins+1)
    Z = norm_counts.reshape(1, -1)                # shape (1, n_bins)
    pcm = ax1.pcolormesh(Theta, R, Z, cmap='viridis', shading='auto')
    ax1.set_xticklabels([]); ax1.set_yticklabels([])
    ax1.set_title("Uniformity 0–360° (bin = {}°)".format(bin_size), va='bottom')
    cbar = fig1.colorbar(pcm, ax=ax1, pad=0.1); cbar.set_label("Normalized frequency")
    if save_path_ring: plt.savefig(save_path_ring, dpi=300)

    # ---- 折线图 ----
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

    # 加载数据集
    dataset = MoleculeDataset(root=args.xyz_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # 初始化 ComENet 模型
    model = ComENetAutoEncoder(
        cutoff=8.0,
        num_layers=4,
        hidden_channels=256,
        middle_channels=256,
        out_channels=1,            # 假设检查点中只用1个输出通道
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

    # 提取潜空间向量和 SMILES
    latent_vectors, smiles_list = load_latent_vectors(model, loader, device)
    print(f"Extracted latent vectors with shape: {latent_vectors.shape}")

    # 官能团标签；失败或未命中返回 'Unlabeled'
    functional_labels = [
        get_functional_group(sm) or "Unlabeled"
        for sm in smiles_list
    ]

    # 对过滤后的潜向量进行降维（先标准化，再使用 t-SNE）
    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent_vectors)
    tsne = TSNE(n_components=2, perplexity=args.perplexity, learning_rate=200, random_state=args.seed, init='pca')
    tsne_results = tsne.fit_transform(latent_scaled)

    # 计算 Davies-Bouldin 指数
    labeled_mask = [lab != "Unlabeled" for lab in functional_labels]
    db_index = np.nan
    if sum(labeled_mask) and len(set(np.array(functional_labels)[labeled_mask])) > 1:
        db_index = davies_bouldin_score(
            tsne_results[labeled_mask], np.array(functional_labels)[labeled_mask]
        )
    if np.isnan(db_index):
        print("Davies-Bouldin 指数：未计算（有效聚类数 < 2）")
    else:
        print(f"Davies-Bouldin 指数: {db_index:.2f}")

    # 绘制对齐性图
    alignment_plot_path = os.path.join(args.output_dir, 'alignment_analysis.svg')
    plot_alignment(tsne_results, functional_labels, db_index, save_path=alignment_plot_path)

    # 绘制均匀性图：生成极坐标热频图和折线图
    polar_plot_path = os.path.join(args.output_dir, 'uniformity_polar.svg')
    line_plot_path = os.path.join(args.output_dir, 'uniformity_line.svg')
    plot_uniformity(tsne_results, bin_size=5, save_path_ring=polar_plot_path, save_path_line=line_plot_path)

if __name__ == '__main__':
    main()
