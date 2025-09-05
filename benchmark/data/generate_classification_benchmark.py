# csv_to_pyg_dataset.py
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm

class MoleculeDataset(Dataset):
    def __init__(self, root, csv_file, target_properties, task_type='classification', threshold='median', allowed_elements=None, transform=None, pre_transform=None):
        """
        初始化数据集。

        参数：
            root (str): 数据集保存路径。
            csv_file (str): CSV 文件路径。
            target_properties (list of str): 要预测的属性名称列表。
            task_type (str): 任务类型，'classification' 或 'regression'。
            threshold (str or float): 用于分类的阈值，'median' 表示使用中位数（仅在分类任务中）。
            transform (callable, optional): 应用于每个样本的变换。
            pre_transform (callable, optional): 应用于每个样本的预变换。
        """
        self.csv_file = csv_file
        self.target_properties = target_properties
        self.task_type = task_type
        self.threshold = threshold
        self.preprocessed = False
        self.allowed_elements = allowed_elements if allowed_elements is not None else {'C', 'H', 'O', 'N', 'F'}
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)

        # 加载数据
        self.data_df = pd.read_csv(self.csv_file)
        self._prepare_data()

    def _prepare_data(self):
        """
        处理 CSV 数据，转换为分类或回归标签，并构建分子图。
        """
        # 确保目标属性存在
        for prop in self.target_properties:
            if prop not in self.data_df.columns:
                raise ValueError(f"Target property '{prop}' not found in CSV.")

        # 提取 SMILES 和目标属性
        smiles = self.data_df['smiles'].values
        y = self.data_df[self.target_properties].values

        # 处理缺失值
        mask = ~pd.isnull(smiles) & ~np.isnan(y).any(axis=1)
        smiles = smiles[mask]
        y = y[mask]

        # 转换为分类标签（如果是分类任务）
        if self.task_type == 'classification':
            if self.threshold == 'median':
                thresh_values = np.median(y, axis=0)
            elif isinstance(self.threshold, (int, float)):
                thresh_values = np.full(y.shape[1], self.threshold)
            else:
                raise ValueError("threshold must be 'median' or a numeric value.")

            labels = (y > thresh_values).astype(int)  # 1: 高于阈值, 0: 低于或等于阈值
        else:
            labels = y  # 回归任务无需转换

        # 保存到 DataFrame
        self.processed_df = pd.DataFrame({
            'smiles': smiles,
        })
        for idx, prop in enumerate(self.target_properties):
            self.processed_df[prop] = labels[:, idx]

        # 编码标签（仅适用于分类任务，虽然已经是0和1，但为了通用性）
        if self.task_type == 'classification':
            self.label_encoders = {}
            for prop in self.target_properties:
                le = LabelEncoder()
                self.processed_df[prop] = le.fit_transform(self.processed_df[prop])
                self.label_encoders[prop] = le

        # 划分训练集和验证集
        self.train_df, self.val_df = train_test_split(
            self.processed_df,
            test_size=0.2,
            random_state=42,
            stratify=self.processed_df[self.target_properties].values if self.task_type == 'classification' else None
        )

        # 构建 Data 对象列表
        self.data_list = []
        print("Processing Training Data...")
        for _, row in tqdm(self.train_df.iterrows(), total=self.train_df.shape[0], desc="Processing Training Data"):
            data = self._create_pyg_data(row)
            if data is not None:
                self.data_list.append(data)
        print("Processing Validation Data...")
        for _, row in tqdm(self.val_df.iterrows(), total=self.val_df.shape[0], desc="Processing Validation Data"):
            data = self._create_pyg_data(row)
            if data is not None:
                self.data_list.append(data)

        self.preprocessed = True

    def _create_pyg_data(self, row):
        """
        将一行数据转换为 PyTorch Geometric 的 Data 对象。

        参数：
            row (pd.Series): 数据行。

        返回：
            Data: PyTorch Geometric 的数据对象，或 None（如果SMILES解析失败、缺少3D坐标或包含不允许的元素）。
        """
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"无法解析 SMILES: {smiles}")
            return None

        # 检查是否只包含允许的元素
        atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
        if not atom_symbols.issubset(self.allowed_elements):
            print(f"分子包含不允许的元素: {smiles} - 元素: {atom_symbols}")
            return None

        # 生成3D坐标
        mol = Chem.AddHs(mol)
        if not AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == 0:
            print(f"无法生成分子的3D坐标: {smiles}")
            return None
        AllChem.UFFOptimizeMolecule(mol)  # 优化分子结构

        # 获取原子类型（原子序数）
        atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        x = torch.tensor(atom_types, dtype=torch.long).unsqueeze(-1)  # [num_nodes, 1]

        # 获取原子位置
        pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)

        # 构建边（无向图）
        edge_index = []
        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            edge_index.append([start, end])
            edge_index.append([end, start])  # 无向边
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, num_edges]
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # 创建 Data 对象
        data = Data(x=x, edge_index=edge_index, pos=pos)

        # 添加 SMILES 作为属性
        data.smiles = smiles

        # 添加标签
        if self.task_type == 'classification':
            labels = []
            for prop in self.target_properties:
                labels.append(torch.tensor([row[prop]], dtype=torch.float))
            data.y = torch.cat(labels)  # [num_properties]
        else:
            labels = []
            for prop in self.target_properties:
                labels.append(torch.tensor([row[prop]], dtype=torch.float))
            data.y = torch.cat(labels)  # [num_properties]

        return data

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

def save_pyg_dataset(dataset, save_path):
    """
    保存 PyTorch Geometric 数据集。

    参数：
        dataset (Dataset): PyTorch Geometric 数据集。
        save_path (str): 保存路径（如 'dataset.pt'）。
    """
    torch.save(dataset, save_path)
    print(f"Saved dataset to {save_path}")

def save_xyz_files(dataset, save_dir, filename_prefix='molecule'):
    """
    将数据集中的分子保存为单独的 .xyz 文件。

    参数：
        dataset (Dataset): PyTorch Geometric 数据集。
        save_dir (str): 保存 .xyz 文件的目录。
        filename_prefix (str): 文件名前缀。
    """
    os.makedirs(save_dir, exist_ok=True)
    for idx, data in enumerate(tqdm(dataset, desc="Saving XYZ files")):
        smiles = data.smiles  # 确保 Data 对象中包含 'smiles' 属性
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"无法解析 SMILES: {smiles}")
            continue
        mol = Chem.AddHs(mol)
        # 生成3D坐标
        if not AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == 0:
            print(f"无法生成分子的3D坐标: {smiles}")
            continue
        AllChem.UFFOptimizeMolecule(mol)  # 优化分子结构
        # 转换为 XYZ 块
        xyz_block = Chem.MolToXYZBlock(mol)
        # 修改第二行
        # First line: number of atoms
        num_atoms = mol.GetNumAtoms()
        # Second line: dataset name, data index, target properties
        # Assuming filename_prefix is the dataset name
        # Extract data index from the loop index (idx)
        dataset_name = filename_prefix
        data_index = idx
        # Get target properties
        target_props = data.y.tolist()  # list of floats
        target_props_str = ' '.join(map(str, target_props))
        # Split the xyz_block into lines
        lines = xyz_block.strip().split('\n')
        if len(lines) < 2:
            print(f"Invalid XYZ block for SMILES: {smiles}")
            continue
        # Replace the second line
        new_second_line = f"{dataset_name} {data_index} {target_props_str}"
        lines[1] = new_second_line
        # Reconstruct the xyz_block
        modified_xyz_block = '\n'.join(lines) + '\n'  # Ensure ending newline
        # Write to file
        xyz_path = os.path.join(save_dir, f"{filename_prefix}_{data_index}.xyz")
        with open(xyz_path, 'w') as f:
            f.write(modified_xyz_block)
    print(f"Saved all XYZ files to {save_dir}")

def save_target_properties_list(target_properties, save_dir, filename='target_properties.txt'):
    """
    保存目标属性名称列表到txt文件。

    参数：
        target_properties (list of str): 目标属性名称列表。
        save_dir (str): 保存txt文件的目录。
        filename (str): txt文件名。
    """
    os.makedirs(save_dir, exist_ok=True)
    txt_path = os.path.join(save_dir, filename)
    with open(txt_path, 'w') as f:
        for prop in target_properties:
            f.write(f"{prop}\n")
    print(f"Saved target properties list to {txt_path}")
    
def plot_class_distribution(dataset, plot_path, task_type='classification'):
    """
    绘制类别分布图或回归目标属性分布图。

    参数：
        dataset (Dataset): PyTorch Geometric 数据集。
        plot_path (str): 保存图像的路径。
        task_type (str): 任务类型，'classification' 或 'regression'。
    """
    if task_type == 'classification':
        labels = [data.y.numpy() for data in dataset]
        labels = np.array(labels)
        num_classes = 2  # 假设二分类
        for i, prop in enumerate(dataset.target_properties):
            counts = np.bincount(labels[:, i].astype(int))
            plt.figure(figsize=(6, 4))
            plt.bar(range(num_classes), counts, tick_label=['Class 0', 'Class 1'])
            plt.xlabel('Classes')
            plt.ylabel('Number of Samples')
            plt.title(f'Class Distribution for {prop}')
            plt.savefig(os.path.join(os.path.dirname(plot_path), f"{prop}_class_distribution.png"))
            plt.close()
        print(f"Saved class distribution plots to {os.path.dirname(plot_path)}")
    else:
        # 回归任务绘制目标属性分布
        y_values = [data.y.numpy() for data in dataset]
        y_values = np.array(y_values)
        for i, prop in enumerate(dataset.target_properties):
            plt.figure(figsize=(6, 4))
            plt.hist(y_values[:, i], bins=50, alpha=0.7)
            plt.xlabel(prop)
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {prop}')
            plt.savefig(os.path.join(os.path.dirname(plot_path), f"{prop}_distribution.png"))
            plt.close()
        print(f"Saved target attribute distribution plots to {os.path.dirname(plot_path)}")

def main():
    parser = argparse.ArgumentParser(description="Convert CSV dataset to PyTorch Geometric Dataset for Classification or Regression")
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file containing the dataset')
    parser.add_argument('--target_properties', type=str, nargs='+', required=True, help='List of target properties to predict')
    parser.add_argument('--task_type', type=str, default='classification', choices=['classification', 'regression'], help='Task type: classification or regression')
    parser.add_argument('--threshold', type=str, default='median', help="Threshold for classification ('median' or a numeric value). Ignored for regression.")
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the processed dataset and plots')
    parser.add_argument('--save_xyz_dir', type=str, default=None, help='Directory to save the processed molecules as XYZ files. If not set, XYZ files will not be saved.')

    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 初始化数据集
    dataset = MoleculeDataset(
        root=args.save_dir,
        csv_file=args.csv_file,
        target_properties=args.target_properties,
        task_type=args.task_type,
        threshold=args.threshold
    )

    # 保存为 PyTorch Geometric 的 Dataset 对象
    csv_basename = os.path.splitext(os.path.basename(args.csv_file))[0]
    save_path = os.path.join(args.save_dir, f"{csv_basename}_{'_'.join(args.target_properties)}_{args.task_type}.pt")
    save_pyg_dataset(dataset, save_path)

    # 保存目标属性名称列表到txt文件
    target_properties_txt = os.path.join(args.save_dir, f"{csv_basename}_target_properties.txt")
    save_target_properties_list(args.target_properties, args.save_dir, filename=f"{csv_basename}_target_properties.txt")

    # 绘制类别分布图或目标属性分布图
    plot_path = os.path.join(args.save_dir, f"{csv_basename}_{'_'.join(args.target_properties)}_{args.task_type}_distribution.png")
    plot_class_distribution(dataset, plot_path, task_type=args.task_type)

    # 如果指定了保存 XYZ 文件的目录，则保存
    if args.save_xyz_dir:
        save_xyz_files(dataset, args.save_xyz_dir, filename_prefix=csv_basename)
        
if __name__ == "__main__":
    main()
