#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
from torch_geometric.loader import DataLoader
from dataset_without_charge import MoleculeDataset

def debug_batch(dataset_dir, batch_size, device):
    # 加载数据集
    dataset = MoleculeDataset(root=dataset_dir)
    print(f"Loaded {len(dataset)} molecules from {dataset_dir}")

    # 使用 DataLoader 生成批次
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 取第一个 batch 进行调试
    batch = next(iter(loader))
    batch = batch.to(device)
    
    # 打印 batch 的关键信息
    print("Batch keys:", batch.keys)
    if hasattr(batch, "batch"):
        print("Batch tensor shape:", batch.batch.shape)
        print("Batch tensor values (first 20):", batch.batch[:20])
    else:
        print("没有检测到 batch 属性。")
    
    if hasattr(batch, "x"):
        print("Node features (x) shape:", batch.x.shape)
    if hasattr(batch, "pos"):
        print("Node positions (pos) shape:", batch.pos.shape)
    if hasattr(batch, "filename"):
        print("Filenames:", batch.filename)
    if hasattr(batch, "scalar_props"):
        print("Scalar properties shape:", batch.scalar_props.shape)
    if hasattr(batch, "chiral_inchi"):
        print("Chiral InChI (first 5):", batch.chiral_inchi[:5])
    
    # 如果 batch 中不存在 batch 属性，则说明 DataLoader 没有生成
    # 此时可以尝试打印整个 batch 对象以查看内容
    print("\nComplete batch object:")
    print(batch)

def main():
    parser = argparse.ArgumentParser(description="Debug DataLoader Batch Generation")
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help="Directory containing .xyz molecule files")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use (cpu or cuda)")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    debug_batch(args.dataset_dir, args.batch_size, device)

if __name__ == '__main__':
    main()
