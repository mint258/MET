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
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 导入模型与数据集
from comenet4charge import ComENetAutoEncoder
from dataloader_qm9 import MoleculeDataset

# 全局风格
sns.set_context("notebook", font_scale=1.2)
mpl.rcParams.update({
    'figure.titlesize': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.title_fontsize': 16,
    'legend.markerscale': 1.2,
})
mpl.rcParams['svg.fonttype'] = 'none'

# 官能团配色（保留）
group_color_map = {
    "Fluoro":    "#96cac1",
    "Nitro":     "#f6f6bc",
    "Cyano":     "#c1bed6",
    "Carboxy":   "#ea8e83",
    "Hydroxy":   "#8aafc9",
    "Benzene":   "#eab375",
    "Alkyne":    "#afcf78",
}

# SMILES 匹配用 SMARTS
GROUP_SMARTS = {
    "Benzene": Chem.MolFromSmarts("c1ccccc1"),
    "Fluoro": Chem.MolFromSmarts("F"),
    "Nitro": Chem.MolFromSmarts("[N+](=O)[O-]"),
    "Cyano": Chem.MolFromSmarts("C#N"),
    "Carboxy": Chem.MolFromSmarts("C(=O)O"),
    "Alkyne": Chem.MolFromSmarts("C#C"),
    "Hydroxy": Chem.MolFromSmarts("CO"),
}

# 渐变色（由小到大）
GRADIENT_COLORS = ['#ea8e83', '#eab375', '#f6f6bc', '#afcf78', '#8aafc9']


# =============== 基本工具 ===============

def make_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("custom5", GRADIENT_COLORS, N=256)

def mol_from_identifier(text: str):
    """兼容 SMILES / InChI，返回 (mol, canonical_smiles)。失败则 (None, 'Unknown')"""
    if not text or text == "Unknown":
        return None, "Unknown"
    try:
        if text.startswith(("InChI=", "1S/")):
            mol = Chem.MolFromInchi(text)
        else:
            mol = Chem.MolFromSmiles(text)
        if mol is None:
            return None, "Unknown"
        smiles = Chem.MolToSmiles(mol, canonical=True)
        return mol, smiles
    except Exception:
        return None, "Unknown"

def get_functional_group(smiles: str):
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
        ("Alkyne", Chem.MolFromSmarts("C#C")),
        ("Hydroxy", Chem.MolFromSmarts("CO")),
    ]
    for name, patt in groups:
        if patt is not None and mol.HasSubstructMatch(patt):
            return name
    return None

def has_group(smiles: str, group_name: str) -> bool:
    if smiles == 'Unknown' or group_name not in GROUP_SMARTS:
        return False
    mol = Chem.MolFromSmiles(smiles)
    patt = GROUP_SMARTS[group_name]
    return mol is not None and patt is not None and mol.HasSubstructMatch(patt)

def compute_mol_weight(smiles: str) -> float:
    """用 RDKit 计算相对分子质量"""
    from rdkit.Chem import Descriptors
    if smiles == 'Unknown':
        return np.nan
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.nan
    return float(Descriptors.MolWt(m))

def nice_integer_ticks(vmin: float, vmax: float, max_ticks: int = 5):
    """为角落色标选择若干“好看”的整数刻度"""
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return []
    if vmax <= vmin:
        return [int(round(vmin))]
    rng = vmax - vmin
    base = 10 ** int(np.floor(np.log10(rng / max(1, (max_ticks - 1)))))
    candidates = np.array([1, 2, 5, 10]) * base
    step = candidates[np.argmin(np.abs(rng / (max_ticks - 1) - candidates))]
    start = int(np.ceil(vmin / step) * step)
    ticks = list(range(start, int(np.floor(vmax / step) * step) + 1, int(step)))
    if len(ticks) == 0:
        ticks = [int(round(vmin)), int(round(vmax))]
    elif len(ticks) > max_ticks:
        idx = np.round(np.linspace(0, len(ticks) - 1, max_ticks)).astype(int)
        ticks = [ticks[i] for i in idx]
    return ticks


# =============== 特征提取（返回 scalar_props） ===============

def load_latent_and_props(model, loader, device):
    """
    返回:
      latent_vectors: [N, d]
      smiles_list:    list[str]
      props_all:      [N, P]  (若数据集中存在 scalar_props)
    """
    # 从数据集对象上拿到性质数 P
    ds = getattr(loader, "dataset", None)
    if ds is not None and hasattr(ds, "property_to_index"):
        num_props = len(ds.property_to_index)
    elif ds is not None and hasattr(ds, "all_properties"):
        num_props = len(ds.all_properties)
    else:
        num_props = None

    model.eval()
    latent_list, smiles_list, props_list = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting latent & props"):
            batch = batch.to(device)
            if not hasattr(batch, "batch") or batch.batch is None:
                batch.batch = torch.zeros(batch.pos.size(0), dtype=torch.long, device=batch.pos.device)

            atomic_embeddings, _ = model(batch)
            molecule_embeddings = scatter_mean(atomic_embeddings, batch.batch, dim=0)  # [B, d]
            latent_list.append(molecule_embeddings.cpu().numpy())

            # chiral_inchi -> SMILES
            for text in batch.chiral_inchi:
                _, smi = mol_from_identifier(text)
                smiles_list.append(smi)

            # 修复: 将一维的 scalar_props 复原为 [B, P]
            if hasattr(batch, "scalar_props"):
                sp = batch.scalar_props
                if sp.dim() == 1:
                    if num_props is not None:
                        sp = sp.view(molecule_embeddings.shape[0], num_props)
                    else:
                        sp = sp.view(molecule_embeddings.shape[0], -1)
                # 若本来就是二维就直接用
                props_list.append(sp.cpu().numpy())

    latent_vectors = np.concatenate(latent_list, axis=0)
    props_all = np.concatenate(props_list, axis=0) if props_list else None
    return latent_vectors, smiles_list, props_all

# =============== 绘图 ===============

def plot_alignment(tsne_results, functional_labels, db_index, save_path=None):
    fig = plt.figure(figsize=(10, 8))
    df = pd.DataFrame({'x': tsne_results[:, 0], 'y': tsne_results[:, 1], 'label': functional_labels})
    ax = fig.gca()

    labeled = df[df['label'] != "Unlabeled"]
    if not labeled.empty:
        sns.scatterplot(
            data=labeled, x='x', y='y',
            hue='label',
            hue_order=list(group_color_map.keys()),
            palette=group_color_map,
            alpha=0.75, s=5, edgecolor='none', ax=ax
        )

    title_txt = "Alignment Analysis"
    if np.isfinite(db_index):
        title_txt += f" (DB={db_index:.2f})"
    ax.set_xlabel('Component 1'); ax.set_ylabel('Component 2')
    ax.legend(
        title='Functional Group',
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),
        bbox_transform=ax.transAxes,
        frameon=True,
        framealpha=0.9
    )    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Alignment plot saved to {save_path}")

def plot_uniformity(tsne_results, bin_size=5, save_path_ring=None, save_path_line=None):
    center = np.mean(tsne_results, axis=0)
    angles_deg = (np.degrees(np.arctan2(tsne_results[:, 1] - center[1],
                                        tsne_results[:, 0] - center[0])) + 360) % 360
    bin_edges = np.arange(0, 360 + bin_size, bin_size)
    counts, _ = np.histogram(angles_deg, bins=bin_edges)
    norm_counts = counts / counts.max() if counts.max() > 0 else counts

    # 极坐标环
    fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    theta_edges = np.deg2rad(bin_edges)
    r_edges = np.array([0.8, 1.0])
    Theta, R = np.meshgrid(theta_edges, r_edges)
    Z = norm_counts.reshape(1, -1)
    pcm = ax1.pcolormesh(Theta, R, Z, cmap='viridis', shading='auto')
    ax1.set_xticklabels([]); ax1.set_yticklabels([])
    # ax1.set_title("Uniformity 0–360° (bin = {}°)".format(bin_size), va='bottom')
    cbar = fig1.colorbar(pcm, ax=ax1, pad=0.1); cbar.set_label("Normalized frequency")
    if save_path_ring: plt.savefig(save_path_ring, dpi=300, bbox_inches='tight')

    # 折线
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    try:
        from scipy.interpolate import make_interp_spline
        xnew = np.linspace(bin_centers.min(), bin_centers.max(), 300)
        ynew = make_interp_spline(bin_centers, norm_counts, k=3)(xnew)
    except Exception:
        xnew, ynew = bin_centers, norm_counts
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(xnew, ynew, lw=2)
    ax2.set_xlabel("Angle (degrees)")
    ax2.set_xlim(0, 360)
    # ax2.set_title("Normalized angle distribution (0–360°)")
    ax2.set_yticks([])
    if save_path_line: plt.savefig(save_path_line, dpi=300, bbox_inches='tight')

def _scatter_with_corner_colorbar(tsne_results, values, title, unit_label, out_path, cbar_title=None, cbar_title_kwargs=None):
    """
    散点 + 角落色标（带整数上下界裁剪与端点标签 <min / >max）：
      - 自动选择若干“好看”的整数刻度 ticks（含最小/最大整数）
      - 将所有小于最小整数 / 大于最大整数的样本值分别裁剪到最小/最大整数
      - 色条两端文字改为 <min_int / >max_int
      - 颜色映射在 [min_int, max_int] 线性分布，确保低端=红(#ea8e83)，高端=蓝(#8aafc9)
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # 使用全局 GRADIENT_COLORS（红→橙→浅黄→绿→蓝）
    cmap = LinearSegmentedColormap.from_list("custom5", GRADIENT_COLORS, N=256)
    v = np.asarray(values, dtype=float)
    valid = np.isfinite(v)
    if valid.sum() == 0:
        print(f"[WARN] {title}: all values are NaN, skip.")
        return

    # 统计均值与标准差（打印参考）
    mu = float(np.nanmean(v[valid]))
    sd = float(np.nanstd(v[valid], ddof=0))
    # 百分位截断：10% 与 90%
    p10, p90 = np.nanpercentile(v[valid], [10, 90])
    # 退化兜底
    if not np.isfinite(p10) or not np.isfinite(p90) or p90 <= p10:
        vmin_real, vmax_real = float(np.nanmin(v[valid])), float(np.nanmax(v[valid]))
        if vmax_real <= vmin_real:
            vmax_real = vmin_real + 1.0
        p10, p90 = vmin_real, vmax_real
    # 裁剪到 [p10, p90]
    v_clamped = np.clip(v, p10, p90)

    # 为减少遮挡，按裁剪后数值从小到大绘制
    order = np.argsort(np.where(valid, v_clamped, np.inf))
    X = tsne_results[order, 0]
    Y = tsne_results[order, 1]
    V = v_clamped[order]

    # 规范化到 [p10, p90]（线性映射）
    norm = mcolors.Normalize(vmin=p10, vmax=p90)

    # 创建画布（右侧预留空间）
    fig, ax = plt.subplots(figsize=(10, 8))    
    sc = ax.scatter(X, Y, c=V, cmap=cmap, norm=norm, s=6, alpha=0.9, edgecolors='none')
    ax.set_xlabel(''); ax.set_ylabel('')
    # ax.set_title(title)

    # 角落色条
    divider = make_axes_locatable(ax)
    # 右侧外置色条（不与主图重叠）
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(unit_label)

    # === 在 colorbar 上方添加标题（可选） ===
    if cbar_title:
        kw = dict(fontsize=plt.rcParams.get('axes.titlesize', 12), fontweight='bold')
        if isinstance(cbar_title_kwargs, dict):
            kw.update(cbar_title_kwargs)
        # 竖直色条：在色条轴坐标系顶部略上方写标题
        try:
            cb.ax.text(0.5, 1.02, str(cbar_title), transform=cb.ax.transAxes,
                       ha='center', va='bottom', clip_on=False, **kw)
            # 给顶部留余量，避免导出时被裁切
            cb.ax.margins(y=0.12)
        except Exception as _e:
            # 如果某些后端不支持 margins，就忽略
            pass

    # 在 [p10, p90] 区间内等距放置 5 个颜色锚点，并标注取整后的刻度
    num_colors = len(GRADIENT_COLORS)
    color_anchors = np.linspace(p10, p90, num=num_colors)
    cb.set_ticks(color_anchors)
    cb.set_ticklabels([str(int(np.rint(x))) for x in color_anchors])    
    
    fig.subplots_adjust(right=0.85)
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    # 打印统计信息，便于核对
    print(f"[{title}] mean={mu:.4f}, std={sd:.4f}, p10={p10:.4f}, p90={p90:.4f}")

def plot_binary_group(tsne_results, smiles_list, group_name, group_color, out_path):
    has = np.array([has_group(sm, group_name) for sm in smiles_list], dtype=bool)
    colors = np.where(has, group_color, "#cccccc")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, s=6, alpha=0.9, edgecolors='none')
    ax.set_xlabel(''); ax.set_ylabel('')
    # ax.set_title(f"{group_name}")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=group_color, markersize=8, label=f"{group_name} present"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#cccccc', markersize=8, label=f"{group_name} absent"),
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),
        bbox_transform=ax.transAxes,
       frameon=True,
        framealpha=0.9
    )
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Binary group plot for {group_name} saved to {out_path}")


# =============== 主流程 ===============

def main():
    parser = argparse.ArgumentParser(description="Latent Visualization (enhanced)")
    parser.add_argument('--xyz_dir', type=str, required=True, help="Directory containing .xyz molecule files")
    parser.add_argument('--model_path', type=str, required=False, help="Path to the pre-trained ComENet model checkpoint")
    parser.add_argument('--device', type=str, default='cpu', help="Device to run the model")
    parser.add_argument('--perplexity', type=float, default=30, help="Perplexity for t-SNE")
    parser.add_argument('--output_dir', type=str, default='latent_vis_results', help="Directory to save plots")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument('--seed', type=float, default=42)

    # 新增功能开关
    parser.add_argument('--mw_plot', action='store_true', help="Enable molecular-weight gradient plot")
    parser.add_argument('--dipole_plot', action='store_true', help="Enable dipole-moment gradient plot (from dataset)")
    parser.add_argument('--binary_group_plots', action='store_true', help="Enable binary present/absent plots for each functional group")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    np.random.seed(int(args.seed))

    # 数据与模型
    dataset = MoleculeDataset(root=args.xyz_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = ComENetAutoEncoder(
        cutoff=8.0,
        num_layers=4,
        hidden_channels=256,
        middle_channels=256,
        out_channels=1,
        atom_embedding_dim=128,
        num_radial=8,
        num_spherical=5,
        num_output_layers=3,
        transformer_layers=1,
        nhead_z=1,
        device=args.device
    ).to(device)

    if args.model_path and os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        pretrained_dict = checkpoint.get("model_state_dict", checkpoint)
        model_dict = model.state_dict()
        filtered = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        print(f"Loaded model weights from {args.model_path} (filtered).")
    else:
        print("Using randomly initialized model (for demo purposes)")

    # 提取潜向量 + SMILES + 标量性质
    latent_vectors, smiles_list, props_all = load_latent_and_props(model, loader, device)
    print(f"Extracted latent vectors with shape: {latent_vectors.shape}")

    # 降维
    scaler = StandardScaler()
    latent_scaled = scaler.fit_transform(latent_vectors)
    tsne = TSNE(n_components=2, perplexity=args.perplexity, learning_rate=200, random_state=int(args.seed), init='pca')
    tsne_results = tsne.fit_transform(latent_scaled)

    # 官能团标签 & DB 指数
    functional_labels = [(get_functional_group(sm) or "Unlabeled") for sm in smiles_list]
    labeled_mask = [lab != "Unlabeled" for lab in functional_labels]
    db_index = np.nan
    if sum(labeled_mask) and len(set(np.array(functional_labels)[labeled_mask])) > 1:
        db_index = davies_bouldin_score(tsne_results[labeled_mask], np.array(functional_labels)[labeled_mask])

    # 原对齐图 & 均匀性图
    plot_alignment(tsne_results, functional_labels, db_index, save_path=os.path.join(args.output_dir, 'alignment_analysis.svg'))
    plot_uniformity(tsne_results, bin_size=5,
                    save_path_ring=os.path.join(args.output_dir, 'uniformity_polar.svg'),
                    save_path_line=os.path.join(args.output_dir, 'uniformity_line.svg'))

    # 分子量渐变图（从 SMILES 计算）
    if args.mw_plot:
        print("Computing molecular weights...")
        mw_vals = [compute_mol_weight(sm) for sm in tqdm(smiles_list, desc="MolWeight")]
        _scatter_with_corner_colorbar(
            tsne_results, mw_vals,
            title="", unit_label="",
            out_path=os.path.join(args.output_dir, "tsne_by_mw.svg")
        )

    # 偶极矩渐变图（直接使用数据集中的 dipole）
    if args.dipole_plot:
        if props_all is None or props_all.size == 0:
            print("[WARN] No scalar_props found in dataset; skip dipole plot.")
        else:
            # 兜底：props_all 必须是 [N, P]
            if props_all.ndim == 1:
                # 用数据集对象确定 P
                if hasattr(dataset, "property_to_index"):
                    P = len(dataset.property_to_index)
                elif hasattr(dataset, "all_properties"):
                    P = len(dataset.all_properties)
                else:
                    # 最后兜底：假设 dipole 是单列
                    P = 1
                props_all = props_all.reshape(-1, P)

            dip_idx = dataset.property_to_index.get("dipole", None)
            if dip_idx is None:
                print("[WARN] 'dipole' not found in dataset.property_to_index; skip dipole plot.")
            else:
                dip_vals = props_all[:, dip_idx]
                _scatter_with_corner_colorbar(
                    tsne_results, dip_vals,
                    title="", unit_label="moment",
                    out_path=os.path.join(args.output_dir, "tsne_by_dipole.svg")
                )

    # 每个官能团的二元上色图
    if args.binary_group_plots:
        print("Drawing binary functional-group plots...")
        for g, color in group_color_map.items():
            outp = os.path.join(args.output_dir, f"binary_{g}.svg")
            plot_binary_group(tsne_results, smiles_list, g, color, outp)


if __name__ == '__main__':
    main()
