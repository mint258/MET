#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap  # 使用 umap-learn 进行降维
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean

# 导入 ComENet 模型和数据集
from comenet4charge import ComENetAutoEncoder
from dataset_without_charge import MoleculeDataset

#########################################
# 官能团检测函数
#########################################
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
        return None
    groups = [
        ("Benzene", Chem.MolFromSmarts("c1ccccc1")),
        ("Fluoro", Chem.MolFromSmarts("F")),
        ("Nitro", Chem.MolFromSmarts("[N+](=O)[O-]")),
        ("Cyano", Chem.MolFromSmarts("C#N")),
        ("CarboxylicAcid", Chem.MolFromSmarts("C(=O)O")),
        ("Alkene", Chem.MolFromSmarts("C=C")),
        ("Alkyne", Chem.MolFromSmarts("C#C")),
        ("Hydroxy", Chem.MolFromSmarts("CO"))
    ]
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
            # 确保 batch 对象中存在 batch 属性
            if not hasattr(batch, "batch") or batch.batch is None:
                batch.batch = torch.zeros(batch.pos.size(0), dtype=torch.long, device=batch.pos.device)
            # 模型前向传播，返回原子嵌入和其他输出（这里只使用原子嵌入）
            atomic_embeddings, _ = model(batch)
            # 使用 scatter_mean 对每个分子进行平均池化
            molecule_embeddings = scatter_mean(atomic_embeddings, batch.batch, dim=0)
            latent_list.append(molecule_embeddings.cpu().numpy())
            for inchi in batch.chiral_inchi:
                if inchi != 'Unknown':
                    try:
                        mol = Chem.MolFromSmiles(inchi)
                        if mol is not None:
                            smiles = Chem.MolToSmiles(mol)
                        else:
                            smiles = 'Unknown'
                    except Exception:
                        smiles = 'Unknown'
                smiles_list.append(smiles)
    latent_vectors = np.concatenate(latent_list, axis=0)
    return latent_vectors, smiles_list

#########################################
# 绘制对齐性图：二维散点图，各点颜色为官能团标签，并在标题中显示 DB 指数
#########################################
def plot_alignment(umap_results, functional_labels, db_index, save_path=None):
    plt.figure(figsize=(10, 8))
    unique_labels = list(set(functional_labels))
    palette = sns.color_palette("hsv", len(unique_labels))
    sns.scatterplot(x=umap_results[:, 0], y=umap_results[:, 1], hue=functional_labels,
                    palette=palette, alpha=0.7, s=60)
    plt.title(f'Alignment Analysis (DB index: {db_index:.2f})', fontsize=16)
    plt.xlabel('Component 1', fontsize=14)
    plt.ylabel('Component 2', fontsize=14)
    plt.legend(title='Functional Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Alignment plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Supervised Latent Visualization from ComENet with DataLoader")
    parser.add_argument('--xyz_dir', type=str, required=True,
                        help="Directory containing .xyz molecule files")
    parser.add_argument('--model_path', type=str, required=False,
                        help="Path to the pre-trained ComENet model checkpoint")
    parser.add_argument('--device', type=str, default='cpu', help="Device to run the model")
    # UMAP 参数设置（包括监督降维相关参数）
    parser.add_argument('--n_neighbors', type=int, default=15, help="n_neighbors for UMAP")
    parser.add_argument('--min_dist', type=float, default=0.5, help="min_dist for UMAP")
    parser.add_argument('--target_weight', type=float, default=0.5, help="Weight for target supervision in UMAP")
    parser.add_argument('--output_dir', type=str, default='latent_vis_results', help="Directory to save plots")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for DataLoader")
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
        out_channels=1,  # 假设检查点中只用1个输出通道
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

    # 得到每个分子的官能团标签（仅检测目标官能团，否则返回 None）
    functional_labels_all = [get_functional_group(sm) for sm in smiles_list]
    # 过滤掉未匹配到目标官能团的分子
    filtered_indices = [i for i, lab in enumerate(functional_labels_all) if lab is not None]
    if len(filtered_indices) == 0:
        raise ValueError("没有分子匹配到指定的官能团，请检查数据或官能团 SMARTS 设置。")
    latent_filtered = latent_vectors[filtered_indices]
    functional_labels = [functional_labels_all[i] for i in filtered_indices]

    # 对过滤后的潜向量进行标准化
    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent_filtered)

    # 将官能团标签转换为数值标签，用于监督降维
    unique_labels = sorted(list(set(functional_labels)))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    target = np.array([label_to_int[label] for label in functional_labels])

    # 使用 UMAP 进行监督降维
    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=42,
        target_weight=args.target_weight  # 监督信息的权重
    )
    umap_results = umap_model.fit_transform(latent_scaled, y=target)

    # 计算 Davies-Bouldin 指数
    db_index = davies_bouldin_score(umap_results, functional_labels)
    print(f"Davies-Bouldin index: {db_index:.2f}")

    # 绘制对齐性图
    alignment_plot_path = os.path.join(args.output_dir, 'alignment_analysis.png')
    plot_alignment(umap_results, functional_labels, db_index, save_path=alignment_plot_path)

if __name__ == '__main__':
    main()
