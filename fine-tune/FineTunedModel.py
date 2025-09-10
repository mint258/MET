import sys
import os
import math
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

# 确保路径正确
sys.path.append(os.path.abspath('../charge_predict'))

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
        # 初始化 MolecularTransformer
        self.molecular_transformer = MolecularTransformer(**molecular_transformer_args).to(device)
        
        # 加载预训练权重
        if 'model_state_dict' in checkpoint:
            full_sd = checkpoint['model_state_dict']

            # 1) 给 autoencoder 加载 autoencoder.* 的那部分
            auto_sd = {
                k[len('autoencoder.'):]: v
                for k, v in full_sd.items()
                if k.startswith('autoencoder.')
            }
            self.autoencoder.load_state_dict(auto_sd)

            # 2) 给 molecular_transformer 加载 molecular_transformer.* 的那部分
            trans_sd = {
                k[len('molecular_transformer.'):]: v
                for k, v in full_sd.items()
                if k.startswith('molecular_transformer.')
            }
            self.molecular_transformer.load_state_dict(trans_sd)
        else:
            raise KeyError("Checkpoint does not contain 'model_state_dict'")

        self.autoencoder.to(device)
        
        
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
