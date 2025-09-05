#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
charge_predict_compare.py
---------------------------------------------------------
比较两类“原子电荷预测”模型在同一测试集上的效果，并绘制
Real vs Predicted 的对角散点图：

  - 模型 A：Smiles2vec.py 训练得到的 GNN+Transformer
  - 模型 B：training_charge_model.py 训练得到的 ComENet4Charge

自动识别 checkpoint 类型（也可手动通过 --model_a/--model_b 指定）。

依赖：
  - dataset_without_charge.MoleculeDataset
  - Smiles2vec.py 里的 GNNTransformerWithEmbedding, ChiralMoleculeDataset
  - comenet4charge.py 里的 ComENetAutoEncoder
  - torch_geometric
  - scikit-learn
---------------------------------------------------------
用法示例：
  python charge_predict_compare.py \
      --checkpoint_a best_gnn_transformer_model.pth \
      --checkpoint_b best_comenet_model.pth \
      --test_data_root ../data/charges/ \
      --device cuda \
      --batch_size 8 \
      --plot_path charge_compare.png
"""

import argparse
import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from matplotlib.ticker import MultipleLocator

from sklearn.metrics import r2_score, mean_squared_error
from torch_geometric.loader import DataLoader

# ======== 统一风格（接近 Nature 的简洁风）========
mpl.rcParams.update({
    'font.family':      'Arial',
    'font.size':        14,
    'axes.linewidth':   1.0,
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.major.width':0.8,
    'ytick.major.width':0.8,
    'lines.markersize': 8,
    'lines.linewidth':  2.0,
    'legend.frameon':   False,
})
mpl.rcParams['svg.fonttype'] = 'none'

# ======== 依赖模块 ========
try:
    # 从 Smiles2vec.py 引入（训练脚本里定义的模型与数据集）
    from Smiles2vec import GNNTransformerWithEmbedding, ChiralMoleculeDataset
except Exception as e:
    GNNTransformerWithEmbedding = None
    ChiralMoleculeDataset = None

sys.path.append(os.path.abspath('../charge_predict'))

from comenet4charge import ComENetAutoEncoder
from dataset_without_charge import MoleculeDataset


# ======== 模型加载 ========
def detect_model_type(ckpt_dict: dict) -> str:
    """
    粗略通过 checkpoint 字段自动识别模型类型：
      - 'smiles2vec'：包含 input_dim / num_gnn_layers / nhead / dim_feedforward 等
      - 'comenet_charge'：包含 cutoff / num_layers / hidden_channels / atom_embedding_dim / nhead_z 等
    """
    keys = set(ckpt_dict.keys())
    # Smiles2vec 的训练脚本保存的关键结构参数
    if {'input_dim', 'hidden_dim', 'num_gnn_layers', 'num_transformer_layers',
        'nhead', 'dim_feedforward', 'dropout', 'output_dim'}.issubset(keys):
        return 'smiles2vec'
    # ComENet4Charge 的训练脚本保存的关键结构参数
    if {'cutoff', 'num_layers', 'hidden_channels', 'middle_channels',
        'atom_embedding_dim', 'num_radial', 'num_spherical',
        'transformer_layers', 'nhead_z'}.issubset(keys):
        return 'comenet_charge'
    # 兼容：仅保存了纯 state_dict 的 Smiles2vec（没有上述键）
    # 通过典型参数名来识别
    if any(k.startswith('gnn.') for k in keys) or 'fc.weight' in keys or any(k.startswith('transformer.') for k in keys):
        return 'smiles2vec'
    raise ValueError("无法从 checkpoint 字段自动识别模型类型，请使用 --model_a/--model_b 指定。")


def load_model(ckpt_path: str,
               model_type: str,
               device: torch.device):
    """
    根据 checkpoint 和（或）类型字符串加载模型，返回 (model, label)。
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    # 自动识别
    if model_type is None:
        model_type = detect_model_type(ckpt)

    if model_type == 'smiles2vec':
        if GNNTransformerWithEmbedding is None:
            raise ImportError("未能从 Smiles2vec.py 导入 GNNTransformerWithEmbedding。"
                              "请确认与本脚本在同一目录且文件命名无误。")
        # 支持两种保存方式：带结构超参/带 model_state_dict；或纯 state_dict
        if 'model_state_dict' in ckpt:
            sd = ckpt['model_state_dict']
            hidden_dim = ckpt['hidden_dim']
            middle_dim = ckpt.get('middle_dim', None)
            if middle_dim is None:
                w = sd.get('gnn.mlps.0.0.weight', None)
                middle_dim = w.shape[0] if w is not None else hidden_dim
            model = GNNTransformerWithEmbedding(
                input_dim=ckpt['input_dim'],
                hidden_dim=hidden_dim,
                num_gnn_layers=ckpt['num_gnn_layers'],
                num_transformer_layers=ckpt['num_transformer_layers'],
                nhead=ckpt['nhead'],
                dim_feedforward=ckpt['dim_feedforward'],
                dropout=ckpt['dropout'],
                output_dim=ckpt['output_dim'],
                middle_dim=middle_dim
            ).to(device)
            model.load_state_dict(sd)
        else:
            # 纯 state_dict：从权重形状推断结构参数
            sd = ckpt
            # hidden_dim / output_dim
            fc_w = sd['fc.weight']                       # [out_dim, hidden_dim]
            hidden_dim = fc_w.shape[1]
            output_dim = fc_w.shape[0]
            # middle_dim（来自两层 MLP 的第一层）
            w_mlp0 = sd.get('gnn.mlps.0.0.weight', None) # [middle_dim, hidden_dim]
            middle_dim = w_mlp0.shape[0] if w_mlp0 is not None else hidden_dim
            # input_dim（若存在 extra_proj: [hidden_dim, input_dim-1]）
            extra_w = sd.get('gnn.extra_proj.weight', None)
            input_dim = (extra_w.shape[1] + 1) if extra_w is not None else 1
            # GNN 层数
            gnn_layers = max(
                [int(k.split('.')[2]) for k in sd.keys() if k.startswith('gnn.convs.')]+[-1]
            ) + 1
            # Transformer 层数 & FF 维度
            tr_layers = max(
                [int(k.split('.')[2]) for k in sd.keys() if k.startswith('transformer.layers.')]+[-1]
            ) + 1
            lin1_w = sd.get('transformer.layers.0.linear1.weight', None)
            dim_feedforward = lin1_w.shape[0] if lin1_w is not None else (4 * hidden_dim)
            # 选择一个能整除 hidden_dim 的 nhead（权重形状对 nhead 不敏感）
            def pick_nhead(d):
                for h in (8, 4, 2, 1):
                    if d % h == 0:
                        return h
                return 1
            nhead = pick_nhead(hidden_dim)
            # dropout 不影响权重形状
            dropout = 0.0
            model = GNNTransformerWithEmbedding(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_gnn_layers=gnn_layers,
                num_transformer_layers=tr_layers,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                output_dim=output_dim,
                middle_dim=middle_dim
            ).to(device)
            model.load_state_dict(sd)
        label = 'GNN'

    elif model_type == 'comenet_charge':
        model = ComENetAutoEncoder(
            cutoff=ckpt.get('cutoff', 8.0),
            num_layers=ckpt.get('num_layers', 4),
            hidden_channels=ckpt.get('hidden_channels', 256),
            middle_channels=ckpt.get('middle_channels', 256),
            out_channels=1,  # 确保与预训练一致
            atom_embedding_dim=ckpt.get('atom_embedding_dim', 128),
            num_radial=ckpt.get('num_radial', 8),
            num_spherical=ckpt.get('num_spherical', 5),
            num_output_layers=3,
            transformer_layers=ckpt.get('transformer_layers', 1),
            nhead_z=ckpt.get('nhead_z', 1),
            device=device
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        label = 'EGNN'
    else:
        raise ValueError(f"不支持的模型类型：{model_type}")

    model.eval()
    return model, label


# ======== 预测 ========
@torch.no_grad()
def predict_charges(model, loader, device):
    """
    兼容两种 forward：
      - Smiles2vec:    forward(data) -> (embeddings, predictions)
      - ComENet4Charge:forward(data) -> (embeddings, predictions)
    其中 predictions 形如 [total_valid_atoms, 1]
    """
    true_all, pred_all = [], []
    for data in loader:
        data = data.to(device)
        out = model(data)
        # 两种模型都返回 (emb, pred)
        if isinstance(out, tuple):
            preds = out[1]
        else:
            # 以防万一（若未来某实现只返回 preds）
            preds = out

        # 目标：data.y（期望形状与 preds 对齐）
        y = data.y
        # 展平成 1D
        preds = preds.view(-1).detach().cpu().numpy()
        y     = y.view(-1).detach().cpu().numpy()

        pred_all.append(preds)
        true_all.append(y)

    pred_all = np.concatenate(pred_all, axis=0)
    true_all = np.concatenate(true_all, axis=0)
    return true_all.tolist(), pred_all.tolist()


# ======== 绘图 ========
def plot_compare(true_a, pred_a, true_b, pred_b,
                 label_a, label_b, save_path='charge_compare.png'):
    fig, ax = plt.subplots(figsize=(6, 6))

    # 确定坐标轴范围与对角线
    all_vals = np.array(true_a + pred_a + true_b + pred_b)
    mn, mx = float(all_vals.min()), float(all_vals.max())
    pad = (mx - mn) * 0.05 if mx > mn else 1e-3
    mn_pad = mn - pad
    mx_pad = mx + pad    
    ax.set_xlim(mn_pad, mx_pad)
    ax.set_ylim(mn_pad, mx_pad)

    # 统一 x/y 轴刻度和标签
    tick_vals   = np.linspace(mn_pad, mx_pad, 5)
    tick_lbs    = [str(round(v, 1)) for v in tick_vals]
    ax.set_xticks(tick_vals); ax.set_yticks(tick_vals)
    ax.set_xticklabels(tick_lbs); ax.set_yticklabels(tick_lbs)
    ax.set_aspect('equal', 'box')
    plt.axis('square')
    
    # 散点
    ax.scatter(true_a, pred_a, s=64, c='#ea8e83', alpha=0.75,
               edgecolor='white', linewidth=1.0, label=label_a)
    ax.scatter(true_b, pred_b, s=64, c='#8aafc9', alpha=0.75,
               edgecolor='white', linewidth=1.0, label=label_b)


    # 对角线
    ax.plot(ax.get_xlim(), ax.get_xlim(), '--', color='black', linewidth=1.0)

    # 轴标签
    ax.set_xlabel('Real Charge')
    ax.set_ylabel('Predicted Charge')

    # 指标
    mse_a = mean_squared_error(true_a, pred_a); rmse_a = math.sqrt(mse_a); r2_a = r2_score(true_a, pred_a)
    mse_b = mean_squared_error(true_b, pred_b); rmse_b = math.sqrt(mse_b); r2_b = r2_score(true_b, pred_b)

    # 第一段（红色）
    txt_a = f'R²={r2_a:.3f}\n'
    ax.text(0.05, 0.95, txt_a,
            transform=ax.transAxes,
            ha='left', va='top',
            color='#ea8e83',
            zorder=3)

    # 第二段（蓝色）
    txt_b = f'R²={r2_b:.3f}'
    ax.text(0.05, 0.95 - 0.06, txt_b,  # 调整 y 偏移对齐
            transform=ax.transAxes,
            ha='left', va='top',
            color='#8aafc9',
            zorder=3)

    # 美化
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()

    # 调整顺序：把原来第二个放到前面
    handles = [handles[1], handles[0]]
    labels = [labels[1], labels[0]]

    # 按调整后的顺序创建 legend
    ax.legend(handles, labels, loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'✔ Saved compare plot to {save_path}')

    # 同时把指标也打印出来便于日志检索
    print("\n=== Metrics ===")
    print(f"A: {label_a}\n    R²={r2_a:.4f}  MSE={mse_a:.6f}  RMSE={rmse_a:.6f}")
    print(f"B: {label_b}\n    R²={r2_b:.4f}  MSE={mse_b:.6f}  RMSE={rmse_b:.6f}")


# ======== CLI ========
def main():
    ap = argparse.ArgumentParser("Compare two charge-prediction checkpoints")
    ap.add_argument('--checkpoint_a', required=True, help='第一个模型的 .pth')
    ap.add_argument('--checkpoint_b', required=True, help='第二个模型的 .pth')
    ap.add_argument('--model_a', choices=['smiles2vec', 'comenet_charge'],
                    default=None, help='可选：显式指定模型 A 类型')
    ap.add_argument('--model_b', choices=['smiles2vec', 'comenet_charge'],
                    default=None, help='可选：显式指定模型 B 类型')
    ap.add_argument('--test_data_root', required=True, help='测试集根目录（与训练时一致）')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--plot_path', default='charge_compare.png')
    ap.add_argument('--max_points', type=int, default=0,
                    help='为避免点过多导致图太密，可随机子采样绘图点数（0 表示不采样）')
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    print(f'Using device: {device}')

    # 选择测试集数据类：若可能用 Smiles2vec 的手性增强版
    dataset_cls = ChiralMoleculeDataset if ChiralMoleculeDataset is not None else MoleculeDataset
    dataset = dataset_cls(root=args.test_data_root)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 加载两种模型
    model_a, label_a = load_model(args.checkpoint_a, args.model_a, device)
    print(f'Loaded A: {label_a}')
    model_b, label_b = load_model(args.checkpoint_b, args.model_b, device)
    print(f'Loaded B: {label_b}')

    # 预测
    true_a, pred_a = predict_charges(model_a, loader, device)
    true_b, pred_b = predict_charges(model_b, loader, device)  # 注意：真实值应与 true_a 相同

    # 可选：随机子采样用于绘图（不影响指标打印）
    if args.max_points and args.max_points > 0:
        rng = np.random.RandomState(114514)
        idx_a = rng.choice(len(true_a), size=min(args.max_points, len(true_a)), replace=False)
        idx_b = rng.choice(len(true_b), size=min(args.max_points, len(true_b)), replace=False)
        true_a_plot = [true_a[i] for i in idx_a]; pred_a_plot = [pred_a[i] for i in idx_a]
        true_b_plot = [true_b[i] for i in idx_b]; pred_b_plot = [pred_b[i] for i in idx_b]
    else:
        true_a_plot, pred_a_plot = true_a, pred_a
        true_b_plot, pred_b_plot = true_b, pred_b

    # 绘图与指标
    plot_compare(true_a_plot, pred_a_plot, true_b_plot, pred_b_plot,
                 label_a, label_b, args.plot_path)


if __name__ == '__main__':
    main()
