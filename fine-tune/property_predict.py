#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
property_predict_single.py
---------------------------------------------------------
单模型性质预测散点图（Nature风格）：
  - 支持 “FineTunedModel (finetune)” 与 “ComENet-for-property (property)”
  - 统一评估与作图：真实值 vs 预测值
---------------------------------------------------------
"""

import matplotlib as mpl
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

import argparse, os, torch, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
from torch_geometric.loader import DataLoader

# 依赖（与你的环境保持一致）
from dataset_finetune import MoleculeDataset
from fine_tune_training import custom_collate_fn_factory
from FineTunedModel import FineTunedModel
from comenet4property import ComENetAutoEncoder as PropertyNet

# ======================= 数据集封装 ======================= #
class PropertyPredictionDataset(MoleculeDataset):
    """返回 (Data, target_tensor)，同时在 data.y 中写入 target 便于 PropertyNet 使用"""
    def __init__(self, root, target_property):
        super().__init__(root)
        if target_property not in self.all_properties:
            raise ValueError(f"{target_property} 不在数据集中可用属性范围 {self.all_properties}")
        self.idx = self.property_to_index[target_property]

    def get(self, idx):
        d = super().get(idx)
        target = d.scalar_props[self.idx:self.idx+1]      # shape [1]
        d.y = target                                      # 供 PropertyNet 内部断言使用
        return d, target

# ========================  模型加载  ====================== #
def load_model_generic(ckpt_path: str,
                       model_type: str,
                       device: torch.device):
    """
    根据 model_type 或 checkpoint 字段自动恢复模型.
    返回：.eval() 后的 model 与可读标签 label
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    # 自动判别类型（如未指定）
    if not model_type:
        if 'pretrained_checkpoint_path' in ckpt and 'molecular_transformer_args' in ckpt:
            model_type = 'finetune'
        elif 'weights' in ckpt and 'loss' in ckpt:
            model_type = 'property'
        else:
            raise ValueError(f"无法自动识别 {ckpt_path} 的模型类型，请用 --model 指定（finetune/property）")

    if model_type == 'finetune':
        model = FineTunedModel(
            pretrained_checkpoint_path = ckpt_path,
            device                     = device,
            molecular_transformer_args = ckpt['molecular_transformer_args']
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        label = 'Finetuned Model'

    elif model_type == 'property':
        model = PropertyNet(
            cutoff=ckpt.get('cutoff', 8.0),
            num_layers=ckpt.get('num_layers',4),
            hidden_channels=ckpt.get('hidden_channels',256),
            middle_channels=ckpt.get('middle_channels',256),
            out_channels=ckpt.get('out_channels', 1),
            num_radial=ckpt.get('num_radial',8),
            num_spherical=ckpt.get('num_spherical',5),
            num_output_layers=1,
            device=device
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        label = 'Property Model'

    else:
        raise ValueError(f"暂不支持的 model_type '{model_type}'")

    model.eval()
    return model, label

# ============ 统一预测 (兼容两种 forward) ============== #
@torch.no_grad()
def predict_once(model, loader, device):
    true_vals, pred_vals = [], []
    for batch_data, targets in tqdm(loader, leave=False):
        batch_data = batch_data.to(device)
        out = model(batch_data)
        if isinstance(out, tuple):      # PropertyNet 返回 (emb, preds)
            preds = out[1]
        else:                           # FineTunedModel 直接返回 preds
            preds = out
        preds = preds.squeeze(-1).cpu().numpy()
        tars  = targets.squeeze(-1).cpu().numpy()
        true_vals.extend(tars.tolist())
        pred_vals.extend(preds.tolist())
    return true_vals, pred_vals

# =============== 单模型散点图 (Nature 风格) ================= #
def plot_single_nature(true_vals, pred_vals, label, prop_name, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))  # 方形画布

    # 坐标轴范围 & 等比例
    all_values = true_vals + pred_vals
    mn, mx = min(all_values), max(all_values)
    padding = (mx - mn) * 0.05 if mx > mn else 1.0
    mn_pad, mx_pad = mn - padding, mx + padding
    ax.set_xlim(mn_pad, mx_pad); ax.set_ylim(mn_pad, mx_pad)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('square')

    # 散点
    ax.scatter(true_vals, pred_vals,
               marker='o', s=64, c='#8aafc9', alpha=0.7,
               edgecolor='white', linewidth=1.0, label=label, zorder=2)

    # 理想对角线
    ax.plot(ax.get_xlim(), ax.get_xlim(),
            linestyle='--', color='black', linewidth=1.2, zorder=1)

    # 标签
    ax.set_xlabel('Real'); ax.set_ylabel('Predicted')

    # 统计指标
    mse = mean_squared_error(true_vals, pred_vals)
    r2  = r2_score(true_vals, pred_vals)
    ax.text(0.05, 0.95, f'{label}\nR²={r2:.3f}, MSE={mse:.3f}',
            transform=ax.transAxes, ha='left', va='top',
            color='#4a4a4a', zorder=3)

    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'✔ Saved plot to {save_path} | R²={r2:.4f}, MSE={mse:.4f}')

# =========================== CLI =========================== #
def main():
    p = argparse.ArgumentParser("Single-model property prediction evaluation")
    p.add_argument('--checkpoint', required=True, help='待评估模型 checkpoint')
    p.add_argument('--model', choices=['finetune','property'], default=None,
                   help='模型类型（默认自动识别）')
    p.add_argument('--test_data_root', required=True, help='测试数据根目录')
    p.add_argument('--target_property', required=True, help='评估的单一目标性质名称')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--plot_path', default='single_model_plot.png')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    print(f'Using device: {device}')

    # 数据
    dataset = PropertyPredictionDataset(args.test_data_root, args.target_property)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=4,
                         collate_fn=custom_collate_fn_factory)

    # 模型
    model, label = load_model_generic(args.checkpoint, args.model, device)
    print(f'Loaded {label}')

    # 预测
    true_vals, pred_vals = predict_once(model, loader, device)

    # 作图
    plot_single_nature(true_vals, pred_vals, label, args.target_property, args.plot_path)

if __name__ == '__main__':
    main()
