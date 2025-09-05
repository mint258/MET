# finetune_training.py
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import torch.optim as optim
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Batch

# 确保路径正确
sys.path.append(os.path.abspath('../charge_predict'))

from dataset_without_charge import MoleculeDataset
from embedding2property import MolecularTransformer
from comenet4charge import ComENetAutoEncoder

class FineTunedModel(nn.Module):
    def __init__(self, pretrained_checkpoint_path, device, molecular_transformer_args):
        super(FineTunedModel, self).__init__()
        
        # 加载预训练的 ComENetAutoEncoder
        checkpoint = torch.load(pretrained_checkpoint_path, map_location=device)
        self.autoencoder = ComENetAutoEncoder(
            cutoff=checkpoint.get('cutoff', 8.0),
            num_layers=checkpoint.get('num_layers', 4),
            hidden_channels=checkpoint.get('hidden_channels', 256),
            middle_channels=checkpoint.get('middle_channels', 256),
            out_channels=1,  # 确保与预训练一致
            atom_embedding_dim=checkpoint.get('atom_embedding_dim', 128),
            num_radial=checkpoint.get('num_radial', 8),
            num_spherical=checkpoint.get('num_spherical', 5),
            num_output_layers=3,
            transformer_layers=checkpoint.get('transformer_layers', 1),
            nhead_z=checkpoint.get('nhead_z', 1),
            device=device
        )
        
        # 加载预训练权重
        if 'model_state_dict' in checkpoint:
            self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise KeyError("Checkpoint does not contain 'model_state_dict'")
        
        self.autoencoder.to(device)
        
        # 初始化 MolecularTransformer
        self.molecular_transformer = MolecularTransformer(**molecular_transformer_args).to(device)
        
    def forward(self, data):
        """
        数据流：
        Data 对象 -> ComENetAutoEncoder -> 原子嵌入 -> Padding 和 Masking -> MolecularTransformer -> 输出预测
        """
        # 通过 ComENetAutoEncoder 获取原子嵌入
        atomic_embeddings, _ = self.autoencoder._forward(data)  # [num_nodes, hidden_channels], [hidden_channels]
        embeddings = self.autoencoder.encoder(atomic_embeddings)  # [num_nodes, atom_embedding_dim]
        
        # 使用 torch_geometric.utils.to_dense_batch 进行填充和掩码生成
        padded_embeddings, mask = to_dense_batch(embeddings, batch=data.batch)  # [batch_size, max_num_nodes, atom_embedding_dim], [batch_size, max_num_nodes]
        
        # 将掩码转换为 float 类型
        mask = mask.float()
        
        # 传递给 MolecularTransformer 进行属性预测
        output = self.molecular_transformer(padded_embeddings, mask)  # [batch_size, output_dim]
        
        return output

def custom_collate_fn(batch):
    """
    自定义的 collate_fn，用于处理分子数据对象，生成批次。
    
    参数：
        batch (list of tuples): 每个元素是 (Data, target_values)。
    
    返回:
        Tuple:
            - batch_data (Batch): 批次的分子数据对象
            - targets (Tensor): 目标分子性质，形状为 [batch_size, num_properties].
    """
    data_list, targets = zip(*batch)
    batch_data = Batch.from_data_list(data_list)
    targets = torch.stack(targets)  # [batch_size, num_properties]
    return batch_data, targets

def train_epoch(model, device, train_loader, optimizer, criterion):
    """
    训练模型一个 epoch。
    
    参数：
        model (nn.Module): 训练的模型。
        device (torch.device): 训练设备。
        train_loader (DataLoader): 训练数据加载器。
        optimizer (torch.optim.Optimizer): 优化器。
        criterion (nn.Module): 损失函数。
    
    返回:
        Tuple[float, float]: 训练损失和平均 R2 分数。
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    for step, (batch_data, targets) in enumerate(tqdm(train_loader, desc="Training")):
        batch_data = batch_data.to(device)
        targets = targets.to(device)  # [batch_size, num_properties]
        
        optimizer.zero_grad()
        outputs = model(batch_data)  # [batch_size, output_dim]
        loss = criterion(outputs, targets)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item() * batch_data.num_graphs
        all_preds.append(outputs.detach().cpu())
        all_targets.append(targets.detach().cpu())
    
    epoch_loss = running_loss / len(train_loader.dataset)
    all_preds = torch.cat(all_preds, dim=0).numpy()  # [total_samples, output_dim]
    all_targets = torch.cat(all_targets, dim=0).numpy()  # [total_samples, output_dim]
    
    # 计算每个属性的 R²，并取平均
    r2 = r2_score(all_targets, all_preds, multioutput='uniform_average')
    return epoch_loss, r2

def validate_epoch(model, device, val_loader, criterion):
    """
    验证模型一个 epoch。
    
    参数：
        model (nn.Module): 验证的模型。
        device (torch.device): 验证设备。
        val_loader (DataLoader): 验证数据加载器。
        criterion (nn.Module): 损失函数。
    
    返回:
        Tuple[float, float]: 验证损失和平均 R2 分数。
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data, targets in tqdm(val_loader, desc="Validation"):
            batch_data = batch_data.to(device)
            targets = targets.to(device)  # [batch_size, num_properties]
            
            outputs = model(batch_data)  # [batch_size, output_dim]
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * batch_data.num_graphs
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    epoch_loss = running_loss / len(val_loader.dataset)
    all_preds = torch.cat(all_preds, dim=0).numpy()  # [total_samples, output_dim]
    all_targets = torch.cat(all_targets, dim=0).numpy()  # [total_samples, output_dim]
    
    # 计算每个属性的 R²，并取平均
    r2 = r2_score(all_targets, all_preds, multioutput='uniform_average')
    return epoch_loss, r2

def freeze_layers(model, layer_list, freeze_up_to_layer):
    """
    冻结模型中指定层及之前的所有层。

    参数：
        model (FineTunedModel): 要冻结层的模型。
        layer_list (List[Tuple[str, nn.Module]]): 层名称和模块的列表。
        freeze_up_to_layer (int or None): 要冻结的最后一层的索引（包含）。如果为 None，则使用默认冻结策略。
    """
    if freeze_up_to_layer is None:
        # 默认冻结到 encoder 层之前的所有层
        for param in model.autoencoder.encoder.parameters():
            param.requires_grad = False
        print("默认冻结到 encoder 层之前的所有层。")
    else:
        for idx, (name, module) in enumerate(layer_list):
            if idx <= freeze_up_to_layer:
                for param in module.parameters():
                    param.requires_grad = False
                print(f"冻结层 {idx}: {name} ({module.__class__.__name__})")
            else:
                break

def main():
    parser = argparse.ArgumentParser(description="Fine-Tune Molecular Property Predictor using Autoencoder and Transformer")
    parser.add_argument('--pretrained_checkpoint_path', type=str, required=True, help='Path to the pretrained ComENetAutoEncoder checkpoint (.pth file)')
    parser.add_argument('--data_root', type=str, required=True, help='Directory containing .xyz files with molecular properties')
    parser.add_argument('--target_property', type=str, required=True, nargs='+', help='Molecular properties to predict. Specify one or more properties from the available options.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of Transformer encoder layers')  # 修改默认值为6
    parser.add_argument('--dim_feedforward', type=int, default=256, help='Hidden dimension in feedforward network')  # 修改默认值为256
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--min_dim', type=int, default=32, help='Minimum dimension for linear layers')    
    parser.add_argument('--num_linear_layers', type=int, default=0, help='Number of linear perceptron layers to reduce embedding dimension by half each layer')
    parser.add_argument('--max_lr_reductions', type=int, default=5, help='Maximum number of learning rate reductions before stopping training early')
    parser.add_argument('--save_model', type=str, default='best_finetuned_model.pth', help='Path to save the trained model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training ("cuda" or "cpu")')
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of data to use for validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--plot_dir', type=str, default=os.getcwd(), help='Directory to save the loss and R2 plots')
    parser.add_argument('--freeze_up_to_layer', type=str, default=None, help='Freeze layers up to (and including) this layer index. If not set, default to freeze up to the penultimate layer.')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # 创建数据集
    dataset = MoleculeDataset(
        root=args.data_root
    )
    
    # 创建 PropertyPredictionDataset
    class PropertyPredictionDataset(MoleculeDataset):
        def __init__(self, root, target_property, transform=None, pre_transform=None):
            super(PropertyPredictionDataset, self).__init__(root, transform, pre_transform)
            
            # target_property 现在是一个列表
            if not isinstance(target_property, list):
                raise TypeError("target_property 应该是一个列表，包含一个或多个属性名称。")
            
            # 验证每个 target_property 是否在 all_properties 中
            for prop in target_property:
                if prop not in self.all_properties:
                    raise ValueError(f"Target property '{prop}' is not in the list of all_properties.")
            
            self.target_properties = target_property
            self.target_indices = [self.property_to_index[prop] for prop in self.target_properties]
        
        def get(self, idx):
            """
            获取指定索引的数据。
            
            返回：
                Tuple[Data, Tensor]: (分子数据对象, 目标分子性质值的张量)
            """
            data = super(PropertyPredictionDataset, self).get(idx)
            target_values = data.scalar_props[self.target_indices]
            return data, target_values
    
    # 实例化 PropertyPredictionDataset
    dataset = PropertyPredictionDataset(
        root=args.data_root,
        target_property=args.target_property,
        transform=None,
        pre_transform=None
    )
    
    # 提取可用的目标属性（从数据集中获取）
    available_properties = dataset.all_properties
    
    # 验证用户输入的 target_property 是否有效
    for prop in args.target_property:
        if prop not in available_properties:
            raise ValueError(f"Invalid target_property '{prop}'. Available options are: {available_properties}")
    
    print(f"Selected target_properties: {args.target_property}")
    print(f"Available target properties: {available_properties}")
    
    # 划分训练集和验证集
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"Total samples: {len(dataset)}, Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4
    )
    
    # 定义 MolecularTransformer 的参数
    # 从预训练检查点获取 atom_embedding_dim
    checkpoint = torch.load(args.pretrained_checkpoint_path, map_location=device)
    atom_embedding_dim = checkpoint.get('atom_embedding_dim', 128)
    
    # 动态设置 output_dim 为 target_property 的数量
    output_dim = len(args.target_property)
    
    molecular_transformer_args = {
        'atom_embedding_dim': atom_embedding_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout,
        'output_dim': output_dim,  # 动态设置
        'num_linear_layers': args.num_linear_layers,
        'min_dim': args.min_dim
    }
    
    # 实例化微调模型
    model = FineTunedModel(
        pretrained_checkpoint_path=args.pretrained_checkpoint_path,
        device=device,
        molecular_transformer_args=molecular_transformer_args
    ).to(device)
    
    # # 动态计算 input_dim 并初始化模型
    # # 假设 One-Hot 编码，input_dim = max_atomic_num + 1
    # max_atomic_num = dataset.data_df['smile'].apply(lambda x: Chem.MolFromSmiles(x)).apply(
    #     lambda mol: max([atom.GetAtomicNum() for atom in mol.GetAtoms()]) if mol else 0
    # ).max()
    
    # num_features = max_atomic_num + 1  # One-Hot 编码的维度
    
    # print(f"Max atomic number in dataset: {max_atomic_num}")
    # print(f"One-Hot encoded feature dimension: {num_features}")
    
    # 如果 `FineTunedModel` 的初始化需要 `input_dim`，确保传递正确
    # 这里假设 `MolecularTransformer` 已经在 `FineTunedModel` 中正确处理 `output_dim`
    
    # 打印预训练模型的所有层（简洁形式）
    print("Pretrained ComENetAutoEncoder layers:")
    layer_list = []
    index = 0

    for name, module in model.autoencoder.named_children():
        if isinstance(module, torch.nn.ModuleList):
            for i, sub_module in enumerate(module):
                layer_name = f"{name}.{i}"
                layer_list.append((layer_name, sub_module))
                print(f"{index}: {layer_name} ({sub_module.__class__.__name__})")
                index += 1
        else:
            layer_name = name
            layer_list.append((layer_name, module))
            print(f"{index}: {layer_name} ({module.__class__.__name__})")
            index += 1

    # 处理冻结层索引（通过命令行参数）
    freeze_up_to_layer = args.freeze_up_to_layer
    
    # 处理输入
    if freeze_up_to_layer is not None:
        try:
            freeze_up_to_layer = int(freeze_up_to_layer)
            if freeze_up_to_layer < 0 or freeze_up_to_layer >= len(layer_list):
                print(f"Invalid layer index for freezing. Please enter a value between 0 and {len(layer_list)-1}. No layers will be frozen.")
                freeze_up_to_layer = None
        except ValueError:
            print("Invalid input for freeze_up_to_layer. It should be an integer.")
            freeze_up_to_layer = None
    else:
        freeze_up_to_layer = None
    
    # 调用 freeze_layers 函数
    freeze_layers(model, layer_list, freeze_up_to_layer)
    
    # 打印冻结状态以确认
    print("\nFrozen layers status:")
    for name, param in model.autoencoder.named_parameters():
        status = "Frozen" if not param.requires_grad else "Trainable"
        print(f"{name}: {status}")    
        
    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("模型的总可训练参数量:", total_params)
    
    # 定义损失函数和优化器
    criterion = torch.nn.L1Loss() # 这是 MAE 损失函数
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    # 定义学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)
    
    # 初始化记录列表
    train_losses = []
    val_losses = []
    train_r2s = []
    val_r2s = []
    final_epochs = int(1e9)
    
    # 创建保存图像的文件夹
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # 初始化学习率减少次数计数器
    lr_reduce_count = 0
    max_lr_reductions = args.max_lr_reductions
    
    # 初始化之前的学习率
    previous_lrs = [group['lr'] for group in optimizer.param_groups]
    
    # 初始化最佳 R2 分数
    best_val_r2 = -float('inf')
    
    # 训练和验证循环
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_r2 = train_epoch(model, device, train_loader, optimizer, criterion)
        val_loss, val_r2 = validate_epoch(model, device, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_r2s.append(train_r2)
        val_r2s.append(val_r2)
    
        # 记录当前学习率
        current_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # 调度器根据验证集的 R2 调整学习率
        scheduler.step(val_r2)
        
        # 记录学习率变化
        lr_decreased = False
        for old_lr, new_lr in zip(previous_lrs, current_lrs):
            if new_lr < old_lr:
                lr_decreased = True
                break
        if lr_decreased:
            lr_reduce_count += 1
            print(f"学习率已减少 {lr_reduce_count} 次")
            if lr_reduce_count > max_lr_reductions:
                print(f"学习率已减少 {lr_reduce_count} 次，达到最大允许次数 {max_lr_reductions}，提前终止训练")
                final_epochs = epoch
                break
        
        # 更新之前的学习率
        previous_lrs = current_lrs.copy()
    
        print(f"Training Loss: {train_loss:.8f}, Training R2: {train_r2:.6f}")
        print(f"Validation Loss: {val_loss:.8f}, Validation R2: {val_r2:.6f}")
        
        # 保存最佳模型
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save({
                'model_state_dict': model.state_dict(),
                'pretrained_checkpoint_path': args.pretrained_checkpoint_path,
                'molecular_transformer_args': molecular_transformer_args,
                'target_properties': args.target_property  # 改为复数形式
            }, args.save_model)
            print(f"Saved best model with validation R2 {best_val_r2:.6f} to {args.save_model}")
    
    print("\nTraining completed.")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, min(args.epochs, final_epochs) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, min(args.epochs, final_epochs) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {", ".join(args.target_property)}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = os.path.join(args.plot_dir, f'loss_curve_{"_".join(args.target_property)}.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Saved loss curve to {loss_plot_path}")
    
    # 绘制 R2 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, min(args.epochs, final_epochs) + 1), train_r2s, label='Training R2')
    plt.plot(range(1, min(args.epochs, final_epochs) + 1), val_r2s, label='Validation R2')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.title(f'Training and Validation R2 for {", ".join(args.target_property)}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(-1, 1)
    r2_plot_path = os.path.join(args.plot_dir, f'r2_curve_{"_".join(args.target_property)}.png')
    plt.savefig(r2_plot_path)
    plt.close()
    print(f"Saved R2 curve to {r2_plot_path}")

# 运行主函数
if __name__ == '__main__':
    main()
