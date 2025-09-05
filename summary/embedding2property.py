import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(module):
    """
    通用权重初始化函数，根据模块类型应用不同的初始化策略。
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')  # 使用Kaiming初始化
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.uniform_(module.weight, -0.05, 0.05)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    # 如果有其他层类型，可以在这里添加初始化方法

class CustomMultiHeadSelfAttention(nn.Module):
    def __init__(self, atom_embedding_dim, num_heads=4):
        """
        自定义多头自注意力层。

        参数：
            atom_embedding_dim (int): 原子嵌入向量的维度。
            num_heads (int): 注意力头的数量。
        """
        super(CustomMultiHeadSelfAttention, self).__init__()
        assert atom_embedding_dim % num_heads == 0, "atom_embedding_dim 必须能被 num_heads 整除"

        self.num_heads = num_heads
        self.d_k = atom_embedding_dim // num_heads  # 每个头的维度

        self.W_q = nn.Linear(atom_embedding_dim, atom_embedding_dim)
        self.W_k = nn.Linear(atom_embedding_dim, atom_embedding_dim)
        self.W_v = nn.Linear(atom_embedding_dim, atom_embedding_dim)
        self.fc = nn.Linear(atom_embedding_dim, atom_embedding_dim)  # 最终线性层

        self.apply(initialize_weights)  # 添加权重初始化
        
    def forward(self, x, mask=None):
        """
        前向传播。

        参数：
            x (Tensor): 输入嵌入向量，形状为 [batch_size, max_num_nodes, atom_embedding_dim]。
            mask (Tensor, optional): 掩码，形状为 [batch_size, max_num_nodes]，1 表示有效节点，0 表示填充节点。

        返回：
            Tensor: 注意力输出，形状为 [batch_size, max_num_nodes, atom_embedding_dim]。
        """
        batch_size, max_num_nodes, atom_embedding_dim = x.size()

        # 线性变换并分割为多个头
        Q = self.W_q(x).view(batch_size, max_num_nodes, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, max_num_nodes, d_k]
        K = self.W_k(x).view(batch_size, max_num_nodes, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, max_num_nodes, d_k]
        V = self.W_v(x).view(batch_size, max_num_nodes, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, max_num_nodes, d_k]

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # [batch_size, num_heads, max_num_nodes, max_num_nodes]

        if mask is not None:
            # mask: [batch_size, max_num_nodes]
            # 扩展为 [batch_size, 1, 1, max_num_nodes] 以匹配 scores 的形状
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, max_num_nodes]
            scores = scores.masked_fill(~mask.bool(), float('-inf'))  # 将填充位置的分数设为负无穷

        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, max_num_nodes, max_num_nodes]

        # 应用注意力权重到 V
        attn_output = torch.matmul(attn_weights, V)  # [batch_size, num_heads, max_num_nodes, d_k]

        # 合并多个头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, max_num_nodes, atom_embedding_dim)  # [batch_size, max_num_nodes, atom_embedding_dim]

        # 最终线性变换
        output = self.fc(attn_output)  # [batch_size, max_num_nodes, atom_embedding_dim]
        return output  # [batch_size, max_num_nodes, atom_embedding_dim]

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, atom_embedding_dim, num_heads=4, dim_feedforward=128, dropout=0.1):
        """
        自定义 Transformer 编码器层，使用多头自注意力。

        参数：
            atom_embedding_dim (int): 原子嵌入向量的维度。
            num_heads (int): 注意力头的数量。
            dim_feedforward (int): 前馈网络的隐藏层维度。
            dropout (float): Dropout 比例。
        """
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = CustomMultiHeadSelfAttention(atom_embedding_dim, num_heads=num_heads)
        self.linear1 = nn.Linear(atom_embedding_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, atom_embedding_dim)  # 保持维度一致

        self.norm1 = nn.LayerNorm(atom_embedding_dim)
        self.norm2 = nn.LayerNorm(atom_embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

        self.apply(initialize_weights)  # 添加权重初始化
        
    def forward(self, src, mask=None):
        """
        前向传播。

        参数：
            src (Tensor): 输入嵌入向量，形状为 [batch_size, max_num_nodes, atom_embedding_dim]。
            mask (Tensor, optional): 掩码，形状为 [batch_size, max_num_nodes]，1 表示有效节点，0 表示填充节点。

        返回：
            Tensor: 编码器层的输出，形状为 [batch_size, max_num_nodes, atom_embedding_dim]。
        """
        # 自注意力机制
        attn_output = self.self_attn(src, mask)  # [batch_size, max_num_nodes, atom_embedding_dim]
        src = self.norm1(attn_output + src)  # 残差连接和层归一化
        src = self.dropout1(src)

        # 前馈网络
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))  # [batch_size, max_num_nodes, atom_embedding_dim]
        src = self.norm2(ff_output + src)  # 残差连接和层归一化
        src = self.dropout2(src)
        return src  # [batch_size, max_num_nodes, atom_embedding_dim]
    
class MolecularTransformer(nn.Module):
    def __init__(self, atom_embedding_dim=128, num_layers=6, num_heads=4, dim_feedforward=128, dropout=0.1, output_dim=1, num_linear_layers=2, min_dim=32):
        """
        分子属性预测的 Transformer 模型。

        参数：
            atom_embedding_dim (int): 原子嵌入向量的维度。
            num_layers (int): Transformer 编码器层的数量。
            num_heads (int): 注意力头的数量。
            dim_feedforward (int): 前馈网络的隐藏层维度。
            dropout (float): Dropout 比例。
            output_dim (int): 输出维度（通常为1，如 HOMO 能量）。
            num_linear_layers (int): 线性感知层的数量，每层将嵌入维度减半。
            min_dim (int): 线性层中允许的最小维度，防止过度缩减。
        """
        super(MolecularTransformer, self).__init__()
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(atom_embedding_dim, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 定义线性感知层
        linear_layers = []
        current_dim = atom_embedding_dim
        for i in range(num_linear_layers):
            next_dim = current_dim // 2  # 每层减半
            if next_dim < min_dim:
                next_dim = min_dim  # 保证最小维度
            linear_layers.append(nn.Linear(current_dim, next_dim))
            linear_layers.append(nn.LeakyReLU())  # 添加激活函数
            current_dim = next_dim

        self.linear_layers = nn.Sequential(*linear_layers)

        # 全局注意力层，用于加权聚合节点特征
        self.global_attn = nn.Linear(current_dim, 1)

        # 最终全连接层，输出单个值
        self.fc = nn.Linear(current_dim, output_dim)

        self.apply(initialize_weights)  # 添加权重初始化
        
    def forward(self, x, mask):
        """
        前向传播。

        参数：
            x (Tensor): 输入嵌入向量，形状为 [batch_size, max_num_nodes, atom_embedding_dim]。
            mask (Tensor): 掩码，形状为 [batch_size, max_num_nodes]，1 表示有效节点，0 表示填充节点。

        返回：
            Tensor: 分子属性的预测值，形状为 [batch_size]。
        """
        out = x  # [batch_size, max_num_nodes, atom_embedding_dim]

        for layer in self.encoder_layers:
            out = layer(out, mask)  # [batch_size, max_num_nodes, atom_embedding_dim]

        # 通过线性感知层逐步减少嵌入维度
        out = self.linear_layers(out)  # [batch_size, max_num_nodes, reduced_dim]

        # 使用全局注意力进行加权聚合
        scores = self.global_attn(out)  # [batch_size, max_num_nodes, 1]
        scores = scores.masked_fill(~mask.unsqueeze(-1).bool(), float('-inf'))  # 忽略填充节点
        attn_weights = F.softmax(scores, dim=1)  # [batch_size, max_num_nodes, 1]
        out = (out * attn_weights).sum(dim=1)  # [batch_size, reduced_dim]

        # 通过最终全连接层映射到单个输出
        out = self.fc(out)  # [batch_size, output_dim]
        return out  # [batch_size, output_dim]