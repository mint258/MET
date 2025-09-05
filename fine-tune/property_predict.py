#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
property_predict_compare_v2.py
---------------------------------------------------------
支持 “FineTunedModel (finetune)”  与  “ComENet-for-property (property)”
两个（或相同）模型的性能对比散点图：
    蓝色 = 模型 A
    红色 = 模型 B
---------------------------------------------------------
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
# === 1. 全局 Nature 风格设置 ===
mpl.rcParams.update({
    # 字体
    'font.family':    'serif',
    'font.serif':     ['Arial'],
    'font.size':       14,

    # 坐标轴线 & 刻度
    'axes.linewidth':  0.8,           # 坐标轴粗细
    'xtick.direction':'in',
    'ytick.direction':'in',
    'xtick.major.size':3,
    'ytick.major.size':3,
    'xtick.major.width':0.8,
    'ytick.major.width':0.8,
    # 图例
    'legend.frameon': False,          # 图例无边框
})


import argparse, os, torch, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
from torch_geometric.loader import DataLoader

# 依赖
from dataset_without_charge import MoleculeDataset
from fine_tune_training import custom_collate_fn
from FineTunedModel import FineTunedModel
from comenet4property   import ComENetAutoEncoder   as PropertyNet
from comenet4charge     import ComENetAutoEncoder   as ChargeNet  # 预留：如果以后需要

molecular_args = {
    'atom_embedding_dim': 256,
    'num_layers': 5,
    'num_heads': 4,
    'dim_feedforward': 256,
    'dropout': 0,
    'output_dim': 1,  
    'num_linear_layers': 0,
    'min_dim': 32
}

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

    返回：
        model  –  已 .eval()
        label  –  用于图例的短标识
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    # ------- 自动判别（如未显式指定） ------- #
    if not model_type:
        if 'pretrained_checkpoint_path' in ckpt and 'molecular_transformer_args' in ckpt:
            model_type = 'finetune'
        elif 'weights' in ckpt and 'loss' in ckpt:
            model_type = 'property'
        else:
            raise ValueError(f"无法自动识别 {ckpt_path} 的模型类型，请用 --model_a / --model_b 指定")

    # ------- 不同类型的重建 ------- #
    if model_type == 'finetune':
        model = FineTunedModel(
            pretrained_checkpoint_path = ckpt_path,
            device                     = device,
            molecular_transformer_args = ckpt['molecular_transformer_args']
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        label = 'finetune'

    elif model_type == 'property':
        # PropertyNet 结构参数训练时通常沿用缺省值；如有改动请在此扩展
        model = PropertyNet(
            cutoff=ckpt.get('cutoff', 8.0),
            num_layers=ckpt.get('num_layers',6),
            hidden_channels=ckpt.get('hidden_channels',256),
            middle_channels=ckpt.get('middle_channels',256),
            out_channels=ckpt.get('out_channels', 1),
            num_radial=ckpt.get('num_radial',8),
            num_spherical=ckpt.get('num_spherical',5),
            num_output_layers=1,
            device=device
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        label = 'direct'

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
        # 两种模型输出格式不同，做一次兼容：
        out = model(batch_data)
        if isinstance(out, tuple):          # PropertyNet 返回 (emb, preds)
            preds = out[1]
        else:                               # FineTunedModel 直接返回 preds
            preds = out
        preds = preds.squeeze(-1).cpu().numpy()
        tars  = targets.squeeze(-1).cpu().numpy()
        true_vals.extend(tars.tolist())
        pred_vals.extend(preds.tolist())
    return true_vals, pred_vals

# =============== 径向渐变背景辅助函数 ================ #
def add_radial_gradient(ax, cmap, origin=(0, 0), max_dist=None):
    """
    在指定的坐标轴上添加一个从原点出发的径向渐变背景。

    Args:
        ax (matplotlib.axes.Axes): 要添加背景的坐标轴对象。
        cmap (matplotlib.colors.Colormap): 用于渐变的颜色映射。
        origin (tuple): 渐变效果的中心点。
        max_dist (float, optional): 用于颜色归一化的最大距离。
                                    如果为 None，则自动从坐标轴范围计算。
    """
    # 获取坐标轴的范围
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 创建一个覆盖整个绘图区域的网格
    x = np.linspace(xlim[0], xlim[1], 500)
    y = np.linspace(ylim[0], ylim[1], 500)
    X, Y = np.meshgrid(x, y)

    # 计算网格上每个点到原点的距离
    distance = np.sqrt((X - origin[0])**2 + (Y - origin[1])**2)

    # 确定用于颜色映射归一化的最大距离
    if max_dist is None:
        # 自动计算：从原点到绘图区域四个角的最远距离
        corners = np.array([[xlim[0], ylim[0]], [xlim[0], ylim[1]],
                            [xlim[1], ylim[0]], [xlim[1], ylim[1]]])
        max_dist = np.max(np.sqrt(np.sum((corners - origin)**2, axis=1)))

    # 将距离归一化到 [0, 1] 区间，以便映射到颜色
    # 使用 clip 避免超出范围
    normalized_distance = np.clip(distance / max_dist, 0, 1)

    # 使用 imshow 绘制颜色网格作为背景
    # zorder=0 确保它在最底层
    im = ax.imshow(normalized_distance,
                   origin='lower',
                   extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                   cmap=cmap,
                   aspect='auto',
                   interpolation='bilinear', # 使渐变平滑
                   zorder=0)
    return im

# ===============  双模型散点图 (含径向渐变背景) ================= #
def plot_compare_nature(true_a, pred_a, true_b, pred_b,
                        label_a, label_b,
                        prop_name, save_path):
    # 2. 创建画布
    fig, ax = plt.subplots(figsize=(5,5))  # 方形画布，5×5英寸

    # --- 核心改动 1: 提前确定坐标轴范围 ---
    all_values = true_a + true_b + pred_a + pred_b
    mn = min(all_values)
    mx = max(all_values)
    padding = (mx - mn) * 0.05  # 增加 5% 的边距
    ax.set_xlim(mn - padding, mx + padding)
    ax.set_ylim(mn - padding, mx + padding)

    # # --- 核心改动 2: 创建自定义颜色映射并添加背景 ---
    # # 定义渐变色：从蓝色系 (#55B7E6) 到橙色系 (#F09739)
    # colors = ['#55B7E6', '#F09739']
    # custom_cmap = LinearSegmentedColormap.from_list('radial_gradient', colors)
    # # 调用辅助函数添加背景
    # add_radial_gradient(ax, cmap=custom_cmap)

    # 3. 散点：A 模型 (zorder=2, 在背景和对角线之上)
    ax.scatter(true_a, pred_a,
               marker='o',
               s=50,
               c='#96cac1',
               alpha=0.7,  # 透明度略微调高，以便更好地看到背景
               edgecolor='white',
               linewidth=0.6,
               label=label_a,
               zorder=2)

    # 4. 散点：B 模型 (zorder=2)
    ax.scatter(true_b, pred_b,
               marker='o',
               s=50,
               c='#f6f6bc',
               alpha=0.7,
               edgecolor='white',
               linewidth=0.6,
               label=label_b,
               zorder=2)

    # 5. 理想对角线 (zorder=1, 在背景之上，散点之下)
    # 直接使用已设定的坐标轴范围绘制
    ax.plot(ax.get_xlim(), ax.get_xlim(),
            linestyle='--',
            color='black',
            linewidth=1.2,
            zorder=1)

    # 6. 坐标轴标签
    ax.set_xlabel(f'Real {prop_name}', labelpad=4)
    ax.set_ylabel(f'Predicted {prop_name}', labelpad=4)
    
    # 7. 标题（可选，Nature 通常不放大标题）
    # ax.set_title(f'{prop_name}: Real vs Predicted', pad=6)

    # 8. 注释框
    mse_a, r2_a = mean_squared_error(true_a, pred_a), r2_score(true_a, pred_a)
    mse_b, r2_b = mean_squared_error(true_b, pred_b), r2_score(true_b, pred_b)
    txt = (f'{label_a}\n'
           f'  R²={r2_a:.3f}, MSE={mse_a:.3f}\n'
           f'{label_b}\n'
           f'  R²={r2_b:.3f}, MSE={mse_b:.3f}')
    # 您可以取消注释这段代码来显示性能指标
    # ax.text(0.05, 0.95, txt,
    #         transform=ax.transAxes,
    #         ha='left', va='top', fontsize=8,
    #         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
    #                   edgecolor='none', alpha=0.7),
    #         zorder=3)  # zorder=3 确保在最顶层

    # # 9. 去除顶部和右侧脊线
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # 10. 轻度网格（可选）
    ax.grid(False)

    # 11. 图例 (zorder=3 确保在最顶层)
    ax.legend(loc='lower right', fontsize=10)
    # if legend:
    #     legend.set_zorder(3)

    # 12. 保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'✔ Saved Nature-style plot with gradient background to {save_path}')

# =========================== CLI =========================== #
def main():
    p = argparse.ArgumentParser("Compare two property-prediction checkpoints")
    p.add_argument('--checkpoint_a', required=True)
    p.add_argument('--checkpoint_b', required=True)
    p.add_argument('--model_a', choices=['finetune','property'], default=None,
                   help='模型 A 的类型；缺省则自动识别')
    p.add_argument('--model_b', choices=['finetune','property'], default=None,
                   help='模型 B 的类型；缺省则自动识别')
    p.add_argument('--test_data_root', required=True)
    p.add_argument('--target_property', required=True,
                   help='要评估的单一目标性质名称')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--plot_path', default='compare_plot.png')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    print(f'Using device: {device}')

    # ---------- 数据集 & loader ---------- #
    dataset  = PropertyPredictionDataset(args.test_data_root, args.target_property)
    loader   = DataLoader(dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=4,
                          collate_fn=custom_collate_fn)

    # ---------- 模型 A ---------- #
    model_a, label_a = load_model_generic(args.checkpoint_a, args.model_a, device)
    print(f'Loaded {label_a}')

    # ---------- 模型 B ---------- #
    model_b, label_b = load_model_generic(args.checkpoint_b, args.model_b, device)
    print(f'Loaded {label_b}')

    # ---------- 预测 ---------- #
    true_a, pred_a = predict_once(model_a, loader, device)
    true_b, pred_b = predict_once(model_b, loader, device)   # true_b 与 true_a 一致

    # ---------- 绘图 ---------- #
    plot_compare_nature(true_a, pred_a, true_b, pred_b,
                 label_a, label_b,
                 args.target_property, args.plot_path)

if __name__ == '__main__':
    main()
