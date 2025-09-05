#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import plotly.express as px
import base64
from io import BytesIO
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
import sys
import pandas as pd

# 添加项目路径，确保能导入自定义数据集类
sys.path.append(os.path.abspath('../charge_predict'))
from dataset_without_charge import MoleculeDataset

import periodictable

# ============================
# 新增：官能团匹配函数
# ============================
def get_functional_group(smiles):
    """
    对给定 SMILES，返回匹配的官能团名称。
    若同时匹配多个，则返回第一个匹配的官能团；若没有匹配，则返回 'None'。
    """
    if smiles == 'Unknown':
        return 'Unknown'
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 'Unknown'
    # 定义官能团 SMARTS 模式（可根据实际情况微调）
    functional_groups = {
        'pyridyl': Chem.MolFromSmarts('n1ccccc1'),
        'azo': Chem.MolFromSmarts('N=N'),
        'primary amine': Chem.MolFromSmarts('[NH2]'),
        'amidine': Chem.MolFromSmarts('C(=N)[NH]'),
        'carboxamide': Chem.MolFromSmarts('C(=O)N'),
        'secondary ketimine': Chem.MolFromSmarts('[#6]=N([#6])'),
        'phenyl': Chem.MolFromSmarts('c1ccccc1'),
        'fluoro': Chem.MolFromSmarts('[F]'),
        'hydroxyl': Chem.MolFromSmarts('[OH]'),
        'carboxyl': Chem.MolFromSmarts('C(=O)[OH]'),
        'alkenyl': Chem.MolFromSmarts('C=C')
    }
    for group, pattern in functional_groups.items():
        if mol.HasSubstructMatch(pattern):
            return group
    return 'None'

# ============================
# 以下为原脚本中的各类函数
# ============================
def aggregate_embeddings(all_embeddings, method='average'):
    if method == 'average':
        aggregated = [embedding.mean(axis=0) for embedding in all_embeddings if embedding.size > 0]
    elif method == 'sum':
        aggregated = [embedding.sum(axis=0) for embedding in all_embeddings if embedding.size > 0]
    else:
        raise ValueError("Unsupported aggregation method. Choose 'average' or 'sum'.")
    if len(aggregated) == 0:
        raise ValueError("No embeddings to aggregate. Check input data.")
    return np.array(aggregated)

def perform_pca(data, n_components=2):
    if data.ndim != 2:
        raise ValueError("Data for PCA must be 2D. Got shape: {}".format(data.shape))
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_scaled)
    return principal_components

def perform_tsne(data, n_components=2, perplexity=30, random_state=42):
    if data.ndim != 2:
        raise ValueError("Data for t-SNE must be 2D. Got shape: {}".format(data.shape))
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    tsne_results = tsne.fit_transform(data_scaled)
    return tsne_results

def perform_umap(data, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    try:
        import umap
    except ImportError:
        raise ImportError("UMAP 未安装。请通过 `pip install umap-learn` 安装。")
    if data.ndim != 2:
        raise ValueError("Data for UMAP must be 2D. Got shape: {}".format(data.shape))
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    umap_results = reducer.fit_transform(data_scaled)
    return umap_results

def plot_embeddings(embeddings_2d, labels=None, title='Embedding Visualization', save_path=None):
    plt.figure(figsize=(12, 8))
    if labels is not None:
        unique_labels = list(set(labels))
        palette = sns.color_palette("hsv", len(unique_labels))
        sns.scatterplot(
            x=embeddings_2d[:, 0], y=embeddings_2d[:, 1],
            hue=labels,
            palette=palette,
            legend='full',
            alpha=0.7,
            s=50
        )
        plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        sns.scatterplot(
            x=embeddings_2d[:, 0], y=embeddings_2d[:, 1],
            alpha=0.7,
            s=50
        )
    plt.title(title, fontsize=16)
    plt.xlabel('Component 1', fontsize=14)
    plt.ylabel('Component 2', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved static plot to {save_path}")

def plot_embeddings_interactive(embeddings_2d, smiles_list, title='Molecule-Level Embedding Visualization', save_html=None):
    hover_text = []
    for smiles in smiles_list:
        if smiles == 'Unknown':
            hover_text.append('<b>SMILES:</b> Unknown')
        else:
            hover_text.append(f'<b>SMILES:</b> {smiles}')
    df = pd.DataFrame({
        'Component 1': embeddings_2d[:, 0],
        'Component 2': embeddings_2d[:, 1],
        'SMILES': smiles_list,
        'Hover_text': hover_text
    })
    fig = px.scatter(
        data_frame=df,
        x='Component 1',
        y='Component 2',
        hover_data={'Hover_text': False},
        custom_data=['Hover_text'],
        title=title,
        width=1000,
        height=800
    )
    fig.update_traces(
        marker=dict(size=8, opacity=0.7),
        hovertemplate="%{customdata[0]}"
    )
    fig.update_layout(title={'x': 0.5}, hovermode='closest')
    if save_html:
        fig.write_html(save_html)
        print(f"Saved interactive plot to {save_html}")

def plot_embeddings_interactive_property(embeddings_2d, property_values, property_name='Property',
                                         title='Property Visualization', save_html=None):
    df = pd.DataFrame({
        'Component 1': embeddings_2d[:, 0],
        'Component 2': embeddings_2d[:, 1],
        property_name: property_values
    })
    fig = px.scatter(
        data_frame=df,
        x='Component 1',
        y='Component 2',
        color=property_name,
        color_continuous_scale=px.colors.sequential.Viridis,
        title=title,
        width=1000,
        height=800
    )
    fig.update_layout(title={'x': 0.5}, hovermode='closest')
    if save_html:
        fig.write_html(save_html)
        print(f"Saved interactive plot to {save_html}")

def plot_embeddings_property(embeddings_2d, property_values, property_name='Property',
                             title='Property Visualization', save_path=None):
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          c=property_values, cmap='viridis', alpha=0.7)
    cbar = plt.colorbar(scatter)
    cbar.set_label(property_name)
    plt.title(title, fontsize=16)
    plt.xlabel('Component 1', fontsize=14)
    plt.ylabel('Component 2', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved static plot to {save_path}")

def plot_embeddings_property_filtered(embeddings_2d, property_values, property_name='Property',
                                        title='Property Visualization', save_path=None):
    x = embeddings_2d[:, 0]
    y = embeddings_2d[:, 1]
    x_q5, x_q95 = np.percentile(x, [5, 95])
    x_med = np.median(x)
    x_half_range = max(x_med - x_q5, x_q95 - x_med)
    x_min, x_max = x_med - x_half_range, x_med + x_half_range

    y_q5, y_q95 = np.percentile(y, [5, 95])
    y_med = np.median(y)
    y_half_range = max(y_med - y_q5, y_q95 - y_med)
    y_min, y_max = y_med - y_half_range, y_med + y_half_range

    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    filtered_embeddings = embeddings_2d[mask]
    filtered_property = np.array(property_values)[mask]

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(filtered_embeddings[:, 0], filtered_embeddings[:, 1],
                          c=filtered_property, cmap='viridis', alpha=0.7)
    cbar = plt.colorbar(scatter)
    cbar.set_label(property_name)
    plt.title(title, fontsize=16)
    plt.xlabel('Component 1', fontsize=14)
    plt.ylabel('Component 2', fontsize=14)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved static filtered plot to {save_path}")

def plot_embeddings_property_mass_and_coord_filtered(embeddings_2d, mass_values, property_name='Molecular Weight',
                                                       title='Molecule-Level Embeddings by Molecular Weight (Mass and Coord Filtered)',
                                                       save_path=None):
    mass_values = np.array(mass_values)
    mass_lower, mass_upper = np.percentile(mass_values, [5, 95])
    mass_mask = (mass_values >= mass_lower) & (mass_values <= mass_upper)
    embeddings_mass_filtered = embeddings_2d[mass_mask]
    mass_filtered = mass_values[mass_mask]

    x = embeddings_mass_filtered[:, 0]
    y = embeddings_mass_filtered[:, 1]
    x_q5, x_q95 = np.percentile(x, [5, 95])
    x_med = np.median(x)
    x_half_range = max(x_med - x_q5, x_q95 - x_med)
    x_min, x_max = x_med - x_half_range, x_med + x_half_range

    y_q5, y_q95 = np.percentile(y, [5, 95])
    y_med = np.median(y)
    y_half_range = max(y_med - y_q5, y_q95 - y_med)
    y_min, y_max = y_med - y_half_range, y_med + y_half_range

    coord_mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    final_embeddings = embeddings_mass_filtered[coord_mask]
    final_mass = mass_filtered[coord_mask]

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(final_embeddings[:, 0], final_embeddings[:, 1],
                          c=final_mass, cmap='viridis', alpha=0.7)
    cbar = plt.colorbar(scatter)
    cbar.set_label(property_name)
    plt.title(title, fontsize=16)
    plt.xlabel('Component 1', fontsize=14)
    plt.ylabel('Component 2', fontsize=14)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved static mass and coordinate filtered plot to {save_path}")

def calc_dbe(mol):
    ring_count = Chem.rdMolDescriptors.CalcNumRings(mol)
    double_bonds = 0
    triple_bonds = 0
    for bond in mol.GetBonds():
        bt = bond.GetBondTypeAsDouble()
        if bt == 2:
            double_bonds += 1
        elif bt == 3:
            triple_bonds += 1
    return ring_count + double_bonds + 2 * triple_bonds

def main():
    parser = argparse.ArgumentParser(description="Interactive Visualization of Molecule Embeddings")
    parser.add_argument('--embeddings_dir', type=str, required=True,
                        help='Directory containing embedding .npy files')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Directory containing .xyz files with molecular properties')
    parser.add_argument('--aggregation_method', type=str, default='average',
                        choices=['average', 'sum'], help='Method to aggregate embeddings per molecule')
    parser.add_argument('--dim_reduction', type=str, default='pca',
                        choices=['pca', 'tsne', 'umap'], help='Dimensionality reduction technique')
    parser.add_argument('--n_components', type=int, default=2,
                        help='Number of dimensions for reduction')
    parser.add_argument('--perplexity', type=float, default=30,
                        help='Perplexity parameter for t-SNE')
    parser.add_argument('--umap_n_neighbors', type=int, default=15,
                        help='Number of neighbors for UMAP')
    parser.add_argument('--umap_min_dist', type=float, default=0.1,
                        help='Minimum distance for UMAP')
    parser.add_argument('--output_dir', type=str, default='embeddings_visualization',
                        help='Directory to save the plots')
    parser.add_argument('--save_interactive_html', type=str, default='molecule_level_embeddings_interactive.html',
                        help='Filename to save the interactive HTML plot (SMILES view)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化数据集
    dataset = MoleculeDataset(root=args.data_root)

    # 创建原子类型索引到元素符号的映射
    element_list = [element.symbol for element in periodictable.elements]
    element_dict = {idx: symbol for idx, symbol in enumerate(element_list)}
    element_dict[0] = 'Unknown'

    print("Loading embeddings, SMILES, and InChI...")
    all_embeddings = []
    all_smiles = []
    all_inchis = []
    all_mol_weights = []
    all_unsaturations = []

    for data in tqdm(dataset, desc="Processing molecules"):
        filename = data.filename
        base_filename = os.path.splitext(filename)[0]
        embedding_path = os.path.join(args.embeddings_dir, base_filename + '.npy')
        if not os.path.exists(embedding_path):
            print(f"Embedding file not found for {filename}: {embedding_path}")
            continue
        try:
            embedding = np.load(embedding_path)
        except Exception as e:
            print(f"Error loading {embedding_path}: {e}")
            continue
        if embedding.ndim != 2:
            print(f"Embedding for {filename} is not 2D. Skipping.")
            continue
        all_embeddings.append(embedding)

        # 获取分子的 InChI 和 SMILES
        inchi = getattr(data, 'chiral_inchi', 'Unknown')
        if inchi != 'Unknown':
            try:
                mol = Chem.MolFromInchi(inchi)
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)
                else:
                    smiles = 'Unknown'
            except Exception as e:
                print(f"Error processing InChI for {filename}: {e}")
                smiles = 'Unknown'
        else:
            smiles = 'Unknown'
        all_smiles.append(smiles)
        all_inchis.append(inchi)

        # 计算分子质量和不饱和度
        if smiles != 'Unknown':
            mol_for_prop = Chem.MolFromSmiles(smiles)
            if mol_for_prop is not None:
                mol_weight = Descriptors.ExactMolWt(mol_for_prop)
                unsaturation = calc_dbe(mol_for_prop)
            else:
                mol_weight = 0.0
                unsaturation = 0.0
        else:
            mol_weight = 0.0
            unsaturation = 0.0
        all_mol_weights.append(mol_weight)
        all_unsaturations.append(unsaturation)

    if len(all_embeddings) == 0:
        raise ValueError("No valid molecule embeddings were loaded. Please check your dataset and embeddings directory.")

    print("Aggregating embeddings for molecule-level visualization...")
    aggregated_embeddings = aggregate_embeddings(all_embeddings, method=args.aggregation_method)
    if aggregated_embeddings.ndim != 2 or aggregated_embeddings.shape[0] == 0:
        raise ValueError("Aggregated embeddings are not in expected 2D format or empty. Check input data.")

    print(f"Applying {args.dim_reduction} for molecule-level embeddings...")
    if args.dim_reduction == 'pca':
        reduced_molecules = perform_pca(aggregated_embeddings, n_components=args.n_components)
    elif args.dim_reduction == 'tsne':
        reduced_molecules = perform_tsne(aggregated_embeddings, n_components=args.n_components, perplexity=args.perplexity)
    elif args.dim_reduction == 'umap':
        reduced_molecules = perform_umap(aggregated_embeddings, n_components=args.n_components,
                                         n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist)
    else:
        raise ValueError("Unsupported dimensionality reduction technique.")

    print("Plotting interactive molecule-level embeddings (by SMILES)...")
    plot_embeddings_interactive(
        embeddings_2d=reduced_molecules,
        smiles_list=all_smiles,
        title='Molecule-Level Embedding Visualization',
        save_html=os.path.join(args.output_dir, args.save_interactive_html)
    )

    print("Plotting molecule-level embeddings colored by Molecular Weight (interactive)...")
    plot_embeddings_interactive_property(
        embeddings_2d=reduced_molecules,
        property_values=all_mol_weights,
        property_name='Molecular Weight',
        title='Molecule-Level Embeddings by Molecular Weight',
        save_html=os.path.join(args.output_dir, 'molecule_level_by_molweight.html')
    )
    if args.dim_reduction == 'pca':
        plot_embeddings_property_filtered(
            embeddings_2d=reduced_molecules,
            property_values=all_mol_weights,
            property_name='Molecular Weight',
            title='Molecule-Level Embeddings by Molecular Weight (Coordinate Filtered)',
            save_path=os.path.join(args.output_dir, 'molecule_level_by_molweight_filtered.png')
        )
    else:
        plot_embeddings_property(
            embeddings_2d=reduced_molecules,
            property_values=all_mol_weights,
            property_name='Molecular Weight',
            title='Molecule-Level Embeddings by Molecular Weight (Static)',
            save_path=os.path.join(args.output_dir, 'molecule_level_by_molweight.png')
        )
    print("Plotting molecule-level embeddings colored by Molecular Weight (Mass and Coordinate Filtered)...")
    plot_embeddings_property_mass_and_coord_filtered(
        embeddings_2d=reduced_molecules,
        mass_values=all_mol_weights,
        property_name='Molecular Weight',
        title='Molecule-Level Embeddings by Molecular Weight (Mass & Coord Filtered)',
        save_path=os.path.join(args.output_dir, 'molecule_level_by_molweight_mass_coord_filtered.png')
    )

    print("Plotting molecule-level embeddings colored by Unsaturation (interactive)...")
    plot_embeddings_interactive_property(
        embeddings_2d=reduced_molecules,
        property_values=all_unsaturations,
        property_name='Unsaturation',
        title='Molecule-Level Embeddings by Unsaturation',
        save_html=os.path.join(args.output_dir, 'molecule_level_by_unsaturation.html')
    )
    if args.dim_reduction == 'pca':
        plot_embeddings_property_filtered(
            embeddings_2d=reduced_molecules,
            property_values=all_unsaturations,
            property_name='Unsaturation',
            title='Molecule-Level Embeddings by Unsaturation (Filtered)',
            save_path=os.path.join(args.output_dir, 'molecule_level_by_unsaturation_filtered.png')
        )
    else:
        plot_embeddings_property(
            embeddings_2d=reduced_molecules,
            property_values=all_unsaturations,
            property_name='Unsaturation',
            title='Molecule-Level Embeddings by Unsaturation (Static)',
            save_path=os.path.join(args.output_dir, 'molecule_level_by_unsaturation.png')
        )

    print("Aggregating embeddings for atom-level visualization...")
    try:
        all_atoms_embeddings = np.vstack(all_embeddings)
    except Exception as e:
        raise ValueError(f"Error stacking atom embeddings: {e}")

    print("Extracting atom types...")
    all_atom_types = []
    for data in tqdm(dataset, desc="Extracting atom types"):
        filename = data.filename
        base_filename = os.path.splitext(filename)[0]
        embedding_path = os.path.join(args.embeddings_dir, base_filename + '.npy')
        if not os.path.exists(embedding_path):
            continue
        try:
            embedding = np.load(embedding_path)
        except Exception as e:
            print(f"Error loading embedding for {filename}: {e}")
            continue
        if embedding.ndim != 2:
            continue
        try:
            atom_type_indices = data.x.squeeze(1).tolist()
        except Exception as e:
            print(f"Error processing atom types for {filename}: {e}")
            continue
        atom_types = [element_dict.get(idx, 'Unknown') for idx in atom_type_indices]
        all_atom_types.extend(atom_types)

    print(f"Applying {args.dim_reduction} for atom-level embeddings...")
    if args.dim_reduction == 'pca':
        reduced_atoms = perform_pca(all_atoms_embeddings, n_components=args.n_components)
    elif args.dim_reduction == 'tsne':
        reduced_atoms = perform_tsne(all_atoms_embeddings, n_components=args.n_components, perplexity=args.perplexity)
    elif args.dim_reduction == 'umap':
        reduced_atoms = perform_umap(all_atoms_embeddings, n_components=args.n_components,
                                     n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist)
    else:
        raise ValueError("Unsupported dimensionality reduction technique.")

    print("Plotting atom-level embeddings...")
    plot_embeddings(
        embeddings_2d=reduced_atoms,
        labels=all_atom_types,
        title='Atom-Level Embedding Visualization',
        save_path=os.path.join(args.output_dir, 'atom_level_embeddings.png')
    )

    # 新增：基于官能团分类的分子级可视化
    print("Assigning functional group labels based on SMILES...")
    functional_labels = []
    for smiles in all_smiles:
        group = get_functional_group(smiles)
        functional_labels.append(group)

    plot_embeddings(
        embeddings_2d=reduced_molecules,
        labels=functional_labels,
        title='Molecule-Level Embeddings Colored by Functional Group',
        save_path=os.path.join(args.output_dir, 'molecule_level_by_functional_group.png')
    )

if __name__ == '__main__':
    main()
