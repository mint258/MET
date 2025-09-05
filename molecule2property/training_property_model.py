# -*- coding: utf-8 -*-

import argparse
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.optim as optim
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 假设ComENet和MoleculeDataset在当前文件中定义或已导入
from comenet4property import ComENetAutoEncoder
from dataset_without_charge import MoleculeDataset

class PropertyPredictionDataset(MoleculeDataset):
    def __init__(self, root, target_properties, transform=None, pre_transform=None):
        """
        初始化数据集，支持多个目标性质的预测。
        
        参数：
            root (str): 存储 .xyz 文件的文件夹路径。
            target_properties (list of str): 要预测的分子性质名称列表。
            transform (callable, optional): 对数据进行转换的函数。
            pre_transform (callable, optional): 在数据被加载之前应用的转换函数。
        """
        super(PropertyPredictionDataset, self).__init__(root, transform, pre_transform)
        
        self.target_properties = target_properties
        self.target_indices = [self.property_to_index[prop] for prop in target_properties]
    
    def get(self, idx):
        # 调用父类的 get 方法，获取 Data 对象
        data = super(PropertyPredictionDataset, self).get(idx)
        
        # 获取目标分子性质值
        target_values = data.scalar_props[self.target_indices]  # 多个标量
        
        # 设置 data.y 为包含所有目标性质的张量
        data.y = target_values  # 形状为 [num_properties]
        
        return data

# 自定义损失函数
class WeightedMSELoss(nn.Module):
    def __init__(self, num_properties):
        super(WeightedMSELoss, self).__init__()
        # 初始化权重向量为全1
        self.weights = nn.Parameter(torch.ones(num_properties))

    def forward(self, predictions, targets):
        # 计算误差
        error = predictions - targets
        # 计算加权误差
        weighted_error = self.weights * (error ** 2)
        # 计算损失
        loss = torch.mean(weighted_error)
        return loss

def train(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    for data in tqdm(loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        _, predictions = model(data)  # 只需要 final_output
        loss = criterion(predictions, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

        # 收集预测值和目标值用于R²计算
        all_predictions.append(predictions.detach().cpu())
        all_targets.append(data.y.detach().cpu())
    
    # 合并所有批次的预测和目标
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    # 计算R²
    r2 = r2_score(all_targets, all_predictions, multioutput='uniform_average')
    return total_loss / len(loader.dataset), r2

def validate(model, device, loader, criterion):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for data in tqdm(loader, desc="Validation"):
            data = data.to(device)
            _, predictions = model(data)  # 只需要 final_output
            loss = criterion(predictions, data.y)
            total_loss += loss.item() * data.num_graphs

            # 收集预测值和目标值用于R²计算
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(data.y.detach().cpu())
    
    # 合并所有批次的预测和目标
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    # 计算R²
    r2 = r2_score(all_targets, all_predictions, multioutput='uniform_average')
    return total_loss / len(loader.dataset), r2

def plot_metrics(train_losses, val_losses, train_r2, val_r2, save_path='training_plot.png'):
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_losses, label='Train Loss', color='tab:red', linestyle='-')
    ax1.plot(epochs, val_losses, label='Validation Loss', color='tab:red', linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 200)  # 设置左侧坐标轴取值范围为 0 到 200
    
    ax2 = ax1.twinx()  # 共享x轴

    color = 'tab:blue'
    ax2.set_ylabel('R²', color=color)
    ax2.plot(epochs, train_r2, label='Train R²', color='tab:blue', linestyle='-')
    ax2.plot(epochs, val_r2, label='Validation R²', color='tab:blue', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    ax2.set_ylim(-1, 1)  # 设置右侧坐标轴取值范围为 -1 到 1
    
    plt.title('Training and Validation Loss & R²')
    fig.tight_layout()
    plt.savefig(save_path)
    # plt.show()

def main():
    parser = argparse.ArgumentParser(description="Train ComENet for Molecular Property Prediction")
    parser.add_argument('--data_root', type=str, required=True, help='Path to the dataset root directory containing .xyz files')
    parser.add_argument('--properties', type=str, nargs='+', required=True, help='List of properties to predict')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--save_path', type=str, default='best_comenet_model.pth', help='Path to save the best model')
    parser.add_argument('--atom_embedding_dim', type=int, default=128, help='The required dim of output molecular embedding vector')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden embedding size')
    parser.add_argument('--middle_channels', type=int, default=256, help='Middle embedding size for two layer linear block')
    parser.add_argument('--num_spherical', type=int, default=5, help='Number of spherical bisis')
    parser.add_argument('--num_radial', type=int, default=8, help='Number of radial bisis')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of interaction blocks')
    parser.add_argument('--cutoff', type=float, default=8.0, help='Cutoff distance for interatomic interactions')
    parser.add_argument('--device', type=str, default='cpu', help='The device for training model')

    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # 加载数据集
    start_time = time.time()
    dataset = PropertyPredictionDataset(root=args.data_root, target_properties=args.properties)
    dataset = dataset.shuffle()

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"Total samples: {len(dataset)}, Training: {len(train_dataset)}, Validation: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 初始化模型
    model = ComENetAutoEncoder(
        cutoff=args.cutoff,
        num_layers=args.num_layers,
        hidden_channels=args.hidden_channels,
        middle_channels=args.middle_channels,
        out_channels=len(args.properties),  # out_channels根据properties数量自动调整
        num_radial=args.num_radial,
        num_spherical=args.num_spherical,
        num_output_layers=1,
        device=args.device
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = WeightedMSELoss(num_properties=len(args.properties)).to(device)

    # 定义学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_r2 = []
    val_r2 = []

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_r2_epoch = train(model, device, train_loader, optimizer, criterion)
        val_loss, val_r2_epoch = validate(model, device, val_loader, criterion)
        print(f"Train Loss: {train_loss:.8f}, Train R²: {train_r2_epoch:.6f}, Val Loss: {val_loss:.8f}, Val R²: {val_r2_epoch:.6f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_r2.append(train_r2_epoch)
        val_r2.append(val_r2_epoch)
        
        # 更新学习率调度器
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'loss': val_loss,
                'weights': criterion.weights.data
            }, args.save_path)
            print(f"Saved best model with validation R2 {val_r2_epoch:.6f} to best_finetuned_model.pth")

    print("Training Complete. Best Val Loss:", best_val_loss)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"训练时间: {elapsed_time:.2f} 秒")

    # 绘制训练和验证损失以及R²
    plot_metrics(train_losses, val_losses, train_r2, val_r2, save_path='training_plot.png')

if __name__ == '__main__':
    main()

