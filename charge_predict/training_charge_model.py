# -*- coding: utf-8 -*-

import argparse
import torch
from torch import nn
from torch_geometric.loader import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 假设 ComENetAutoEncoder 和 MoleculeDataset 在当前文件中定义或已导入
from comenet4charge import ComENetAutoEncoder
from dataset_without_charge import MoleculeDataset

def train(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    for data in tqdm(loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        _, predictions = model(data)  # 解包模型输出
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
    print('all_predictions:',all_predictions)
    print('all_targets:',all_targets)
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
            _, predictions = model(data)  # 解包模型输出
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
    ax1.plot(epochs, train_losses, label='Train Loss', color=color, linestyle='-')
    ax1.plot(epochs, val_losses, label='Validation Loss', color=color, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    # 根据实际损失范围调整
    ax1.set_ylim(0, max(max(train_losses), max(val_losses)) * 1.1)
    
    ax2 = ax1.twinx()  # 共享x轴

    color = 'tab:blue'
    ax2.set_ylabel('R²', color=color)
    ax2.plot(epochs, train_r2, label='Train R²', color=color, linestyle='-')
    ax2.plot(epochs, val_r2, label='Validation R²', color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    ax2.set_ylim(-1, 1)  # 设置右侧坐标轴取值范围为 -1 到 1
    
    plt.title('Training and Validation Loss & R²')
    fig.tight_layout()
    plt.savefig(save_path)
    # plt.show()

def main():
    parser = argparse.ArgumentParser(description="Train ComENet for Atomic Charge Prediction")
    parser.add_argument('--data_root', type=str, required=True, help='Path to the dataset root directory containing .xyz files')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--save_path', type=str, default='best_comenet_model.pth', help='Path to save the best model')
    parser.add_argument('--hidden_channels', type=int, default=256, help='Hidden embedding size')
    parser.add_argument('--middle_channels', type=int, default=256, help='Middle embedding size for two layer linear block')
    parser.add_argument('--atom_embedding_dim', type=int, default=128, help='The output dim of pretraining model')
    parser.add_argument('--num_spherical', type=int, default=5, help='Number of spherical bisis')
    parser.add_argument('--num_radial', type=int, default=8, help='Number of radial bisis')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of interaction blocks')
    parser.add_argument('--transformer_layers', type=int, default=1, help='Number of transformer layers in each interaction block')
    parser.add_argument('--transformer_heads_z', type=int, default=1, help='Number of attention heads in transformer for z')
    parser.add_argument('--cutoff', type=float, default=8.0, help='Cutoff distance for interatomic interactions')
    parser.add_argument('--device', type=str, default='cpu', help='The device for training model')

    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f'Using device: {device}')

    # 加载数据集
    start_time = time.time()
    dataset = MoleculeDataset(root=args.data_root)
    print(dataset)
    dataset = dataset.shuffle()

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 初始化模型
    model = ComENetAutoEncoder(
        cutoff=args.cutoff,
        num_layers=args.num_layers,
        hidden_channels=args.hidden_channels,
        middle_channels=args.middle_channels,
        atom_embedding_dim=args.atom_embedding_dim,
        out_channels=1,  # 每个节点一个电荷值
        num_radial=args.num_radial,
        num_spherical=args.num_spherical,
        num_output_layers=3,
        transformer_layers=args.transformer_layers,
        nhead_z=args.transformer_heads_z,
        device=args.device
    ).to(device)

    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print("模型的总参数量:", total_params)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()  # 或使用 WeightedMSELoss(num_properties=1) 如果需要权重

    # 定义学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=20, verbose=True)
    
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
        print(f"Train Loss: {train_loss:.6f}, Train R²: {train_r2_epoch:.4f}, Val Loss: {val_loss:.6f}, Val R²: {val_r2_epoch:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_r2.append(train_r2_epoch)
        val_r2.append(val_r2_epoch)
        
        # 更新学习率调度器
        scheduler.step(val_r2_epoch)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'cutoff': args.cutoff,
                'num_layers': args.num_layers,
                'hidden_channels': args.hidden_channels,
                'middle_channels': args.middle_channels,
                'atom_embedding_dim': args.atom_embedding_dim,
                'out_channels': 1,  # 每个节点一个电荷值
                'num_radial': args.num_radial,
                'num_spherical': args.num_spherical,
                'num_output_layers': 3,
                'transformer_layers': args.transformer_layers,
                'nhead_z': args.transformer_heads_z,
                'device': args.device
            }, args.save_path)
            print(f"Saved Best Model with Val Loss: {val_loss:.6f}")

    print("Training Complete. Best Val Loss:", best_val_loss)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"训练时间: {elapsed_time:.2f} 秒")

    # 绘制训练和验证损失以及R²
    plot_metrics(train_losses, val_losses, train_r2, val_r2, save_path='training_plot.png')

if __name__ == '__main__':
    main()
