#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
from torch_geometric.loader import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

# 从微调训练脚本中加载 FineTunedModel 和自定义 collate 函数
from fine_tune_training import FineTunedModel, custom_collate_fn

# 定义一个用于能量预测的测试数据集类
class EnergyPredictionDataset(torch.utils.data.Dataset):
    """
    继承自 MoleculeDataset，对每个样本返回 (Data, target) 元组，其中 target 为一个 [1] 张量，
    代表指定能量属性（默认使用 "U0"，你可以根据实际需要修改为 "U298"、"H298" 等）。
    """
    def __init__(self, root, target_property="U0", transform=None, pre_transform=None):
        # 调用 MoleculeDataset 的初始化（文件中的 all_properties 定义见 dataset_without_charge.py）
        from dataset_without_charge import MoleculeDataset  # 局部引入
        self.dataset = MoleculeDataset(root, transform=transform, pre_transform=pre_transform)
        
        # 如果传入的是字符串，则转换为单元素列表
        if isinstance(target_property, str):
            target_property = [target_property]
            
        # 检查目标属性是否在所有可能的属性列表中
        for prop in target_property:
            if prop not in self.dataset.all_properties:
                raise ValueError(f"目标属性 '{prop}' 不存在，可选属性有：{self.dataset.all_properties}")
        self.target_properties = target_property
        self.target_indices = [self.dataset.property_to_index[prop] for prop in self.target_properties]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset.get(idx)
        # data.scalar_props 的形状为 [num_properties]，选择目标属性（若只有一个，则结果为标量张量）
        target_values = data.scalar_props[self.target_indices]
        # 确保返回的 target 为浮点型张量
        return data, target_values.float()

def load_finetuned_model(checkpoint_path, device):
    """
    加载微调后的模型。检查点中应包含以下键：
        - 'pretrained_checkpoint_path'
        - 'molecular_transformer_args'
        - 'model_state_dict'
        - 'target_properties'（记录了预测的属性名称）
    此处只通过一次实例化并使用检查点中的超参数确保模型结构与训练时一致。
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在：{checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pretrained_checkpoint_path = checkpoint['pretrained_checkpoint_path']
    molecular_transformer_args = checkpoint['molecular_transformer_args']
    
    # 直接使用检查点中的超参数初始化模型，保证与训练时结构一致
    model = FineTunedModel(pretrained_checkpoint_path, device, molecular_transformer_args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    return model

def predict_energy(model, device, loader):
    """
    使用微调模型对测试数据进行能量预测。
    返回两个列表：真实能量和预测能量（单位均为 kcal/mol）。
    """
    all_true_energy = []
    all_pred_energy = []
    
    with torch.no_grad():
        for batch_data, targets in tqdm(loader, desc="预测中"):
            batch_data = batch_data.to(device)
            targets = targets.to(device)
            outputs = model(batch_data)  # 输出形状为 [batch_size, output_dim]，对于能量预测 output_dim 应为 1
            # 将张量转为一维列表（每个样本对应一个能量值）
            all_true_energy.extend(targets.cpu().numpy().flatten().tolist())
            all_pred_energy.extend(outputs.cpu().numpy().flatten().tolist())
            
    return all_true_energy, all_pred_energy

def plot_energy_results(true_energy, pred_energy, plot_path):
    """
    绘制真实能量与预测能量的散点图，并在图上标注 R² 和 MSE。
    横轴和纵轴均标注单位：kcal/mol。
    """
    mse = mean_squared_error(true_energy, pred_energy)
    r2 = r2_score(true_energy, pred_energy)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(true_energy, pred_energy, alpha=0.5, label='数据点', edgecolors='w', s=50)
    min_val = min(min(true_energy), min(pred_energy))
    max_val = max(max(true_energy), max(pred_energy))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想线')
    plt.xlabel('真实能量 (kcal/mol)', fontsize=14)
    plt.ylabel('预测能量 (kcal/mol)', fontsize=14)
    plt.title('真实能量 vs 预测能量', fontsize=16)
    plt.text(0.05, 0.95, f'$R^2$ = {r2:.4f}\nMSE = {mse:.4f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="使用微调后的模型进行能量预测，并计算性能指标（单位均为 kcal/mol）"
    )
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="微调模型的检查点路径 (.pth 文件)")
    parser.add_argument('--test_data_root', type=str, required=True,
                        help="测试数据所在目录（包含 .xyz 文件）")
    parser.add_argument('--target_property', type=str, default="U0",
                        help="要预测的能量属性名称，默认为 'U0'，请确保该属性存在于数据集的 all_properties 中")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="预测时的批次大小")
    parser.add_argument('--plot_path', type=str, default='energy_prediction_scatter.png',
                        help="散点图保存路径")
    parser.add_argument('--device', type=str, default='cuda',
                        help='运行设备 ("cuda" 或 "cpu")')
    parser.add_argument('--num_workers', type=int, default=16,
                        help="DataLoader 使用的线程数")
    
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    
    print("加载微调模型...")
    model = load_finetuned_model(args.checkpoint_path, device)
    print("模型加载成功。")
    
    print("加载测试数据集...")
    dataset = EnergyPredictionDataset(root=args.test_data_root, target_property=args.target_property)
    print(f"测试数据集样本总数: {len(dataset)}")
    
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=custom_collate_fn, num_workers=args.num_workers)
    print("DataLoader 创建成功。")
    
    print("开始能量预测...")
    true_energy, pred_energy = predict_energy(model, device, test_loader)
    print("预测结束。")
    
    print("计算性能指标并绘制散点图...")
    plot_energy_results(true_energy, pred_energy, args.plot_path)
    print(f"散点图保存至: {args.plot_path}")
    
    mse = mean_squared_error(true_energy, pred_energy)
    r2 = r2_score(true_energy, pred_energy)
    print(f"均方误差 (MSE): {mse:.4f} kcal/mol")
    print(f"R²: {r2:.4f}")

if __name__ == '__main__':
    main()
