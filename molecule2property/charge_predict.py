# -*- coding: utf-8 -*-

import argparse
import torch
from torch_geometric.loader import DataLoader
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
import pickle
import periodictable
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# 假设 ComENetAutoEncoder 和 MoleculeDataset 在当前文件中定义或已导入
from comenet4charge import ComENetAutoEncoder
from dataset_without_charge import MoleculeDataset

def load_model(checkpoint_path, device):
    """
    加载训练好的模型。

    参数：
        checkpoint_path (str): 模型检查点文件的路径。
        device (torch.device): 设备（CPU或GPU）。

    返回：
        model (ComENetAutoEncoder): 加载了权重的模型。
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # print("Checkpoint state_dict keys and shapes:")
    # for k, v in checkpoint['model_state_dict'].items():
    #     print(k, v.shape)
        
    # 假设模型初始化参数已知并与训练时相同
    model = ComENetAutoEncoder(
        cutoff=checkpoint.get('cutoff', 8.0),
        num_layers=checkpoint.get('num_layers', 4),
        hidden_channels=checkpoint.get('hidden_channels', 256),
        middle_channels=checkpoint.get('middle_channels', 128),
        out_channels=1,  # 每个节点一个电荷值
        atom_embedding_dim=checkpoint.get('atom_embedding_dim', 128),
        num_radial=checkpoint.get('num_radial', 8),
        num_spherical=checkpoint.get('num_spherical', 5),
        num_output_layers=3,
        transformer_layers=checkpoint.get('transformer_layers', 1),
        nhead_z=checkpoint.get('nhead_z', 1),
        device=device
    )

    # print("\nCurrent model state_dict keys and shapes:")
    # for k, v in model.state_dict().items():
    #     print(k, v.shape)
        
    model.load_state_dict(checkpoint['model_state_dict'])

    # load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # if load_result.missing_keys:
    #     print("Warning: The following keys were missing in the checkpoint and not loaded:")
    #     for k in load_result.missing_keys:
    #         print(k)
    # if load_result.unexpected_keys:
    #     print("Warning: The following keys in the checkpoint were unexpected and not loaded:")
    #     for k in load_result.unexpected_keys:
    #         print(k)
            
    model.to(device)
    model.eval()  # 设置为评估模式
    return model

def predict(model, device, loader, embeddings_dir, charges_dir):
    """
    使用模型进行预测，并提取嵌入向量和电荷预测。

    参数：
        model (ComENetAutoEncoder): 训练好的模型。
        device (torch.device): 设备（CPU或GPU）。
        loader (DataLoader): 数据加载器。
        embeddings_dir (str): 保存嵌入向量的文件夹路径。
        charges_dir (str): 保存电荷预测结果的文件夹路径。

    返回：
        None
    """

    # 创建输出文件夹（如果不存在）
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(charges_dir, exist_ok=True)

    # 定义索引到元素符号的映射
    element_dict_rev = {idx: element.symbol for idx, element in enumerate(periodictable.elements)}

    all_true_charges = []
    all_predicted_charges = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            # 打印 Batch 对象的属性以进行调试
            # print('Batch :', batch)

            # 访问自定义属性
            filenames = batch.filename  # List[str], 长度为 batch_size
            scalar_props = batch.scalar_props  # Tensor [batch_size, num_properties]
            scalar_props = scalar_props.view(len(filenames),-1)

            # print('filenames:', filenames)
            # print('scalar_props:', scalar_props)

            batch_data = batch.to(device)
            
            # 获取真实电荷值
            true_charges = batch_data.y.view(-1).cpu().numpy()  # [total_num_nodes]

            embeddings, predictions = model(batch_data)  # 获取嵌入向量和电荷预测
            embeddings = embeddings.cpu().numpy()  # [total_num_nodes, atom_embedding_dim]
            predictions = predictions.cpu().numpy()  # [total_num_nodes, 1]

            all_true_charges.extend(true_charges.tolist())
            all_predicted_charges.extend(predictions.tolist())
            
            # 获取每个分子在批次中的节点数
            node_counts = batch_data.batch.bincount().cpu().numpy()  # [batch_size]
            start = 0
            for count, filename, props in zip(node_counts, filenames, scalar_props):
                end = start + count
                molecule_emb = embeddings[start:end]  # [num_nodes, atom_embedding_dim]
                molecule_preds = predictions[start:end].flatten()  # [num_nodes]
                molecule_x = batch_data.x[start:end, 0].cpu().numpy()  # [num_nodes]
                molecule_pos = batch_data.pos[start:end].cpu().numpy()  # [num_nodes, 3]
                molecule_true_charges = true_charges[start:end]  # [num_nodes]
                start = end

                # 保存嵌入向量
                emb_filename = os.path.splitext(filename)[0] + '.npy'
                emb_path = os.path.join(embeddings_dir, emb_filename)
                # df = pd.DataFrame(molecule_emb)
                # df.to_csv(emb_path, index=False)
                np.save(emb_path, molecule_emb)

                # 保存电荷预测
                charges_filename = os.path.splitext(filename)[0] + '_charges.csv'
                charges_path = os.path.join(charges_dir, charges_filename)
                with open(charges_path, 'w') as f:
                    # 写入每个原子的类型、坐标和预测电荷
                    for atom_type_idx, pos, charge in zip(molecule_x, molecule_pos, molecule_preds):
                        atom_type = element_dict_rev.get(atom_type_idx, 'Unknown')
                        x, y, z = pos
                        f.write(f"{atom_type} {x} {y} {z} {charge}\n")
    return all_true_charges,  [item for sublist in all_predicted_charges for item in sublist]

def plot_results(true_charges, predicted_charges, plot_path):
    """
    绘制真实值与预测值的散点图，并在图上标注 R² 和 MSE。

    参数：
        true_charges (list): 真实的电荷值。
        predicted_charges (list): 预测的电荷值。
        plot_path (str): 绘图保存路径。

    返回：
        None
    """
    mse = mean_squared_error(true_charges, predicted_charges)
    r2 = r2_score(true_charges, predicted_charges)

    plt.figure(figsize=(8, 8))
    plt.scatter(true_charges, predicted_charges, alpha=0.5, label='data point', edgecolors='w', s=50)
    min_val = min(min(true_charges), min(predicted_charges))
    max_val = max(max(true_charges), max(predicted_charges))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='ideal line')
    plt.xlabel('real charge', fontsize=14)
    plt.ylabel('predict charge', fontsize=14)
    plt.title('real vs predict', fontsize=16)
    plt.text(0.05, 0.95, f'$R^2$ = {r2:.4f}\nMSE = {mse:.4f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
def main():
    parser = argparse.ArgumentParser(description="Predict Atomic Charges and Extract Embeddings using Trained ComENetAutoEncoder on Test Set")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--test_data_root', type=str, required=True, help='Path to the test dataset directory containing .xyz files for prediction')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for prediction (set to 1 for filename mapping)')
    parser.add_argument('--embeddings_dir', type=str, default='embeddings', help='Directory to save embeddings')
    parser.add_argument('--charges_dir', type=str, default='charges', help='Directory to save charge predictions')
    parser.add_argument('--plot_path', type=str, default='charge_predictions_scatter.png', help='Path to save the scatter plot')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (e.g., "cpu" or "cuda")')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    print("Loading model...")
    model = load_model(args.checkpoint_path, device)
    print("Model loaded successfully.")

    print("Loading test dataset...")
    dataset = MoleculeDataset(root=args.test_data_root)
    print(f"Total molecules in test dataset: {len(dataset)}")
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )    
    print("DataLoader created.")

    print("Starting prediction...")
    true_charges, predicted_charges = predict(model, device, loader, args.embeddings_dir, args.charges_dir)
    print("Prediction completed.")

    print(f"Embeddings saved in folder: {args.embeddings_dir}")
    print(f"Charge predictions saved in folder: {args.charges_dir}")

    # 计算并绘制性能指标
    print("Calculating performance metrics and plotting results...")
    plot_results(true_charges, predicted_charges, args.plot_path)
    print(f"Scatter plot saved at: {args.plot_path}")

if __name__ == '__main__':
    main()