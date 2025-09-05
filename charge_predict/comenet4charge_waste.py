# -*- coding: utf-8 -*-

from torch_cluster import radius_graph
from torch_geometric.nn import GraphConv, GraphNorm
from torch_geometric.nn import inits

from features import angle_emb, torsion_emb

from torch_scatter import scatter, scatter_min

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

import math
from math import sqrt

try:
    import sympy as sym
except ImportError:
    sym = None


def swish(x):
    return x * torch.sigmoid(x)


def initialize_weights(weight, method='glorot', bias=None):
    """
    初始化权重的通用函数。
    :param weight: 要初始化的权重张量
    :param method: 初始化方法，可以是 'glorot', 'uniform', 'kaiming_uniform', 'zeros' 等
    :param bias: 可选的偏置张量，用于相应的初始化
    """
    if method == 'glorot':
        inits.glorot(weight)
    elif method == 'glorot_orthogonal':
        inits.glorot_orthogonal(weight, scale=2.0)
    elif method == 'uniform':
        bound = 1.0 / math.sqrt(weight.size(-1))
        torch.nn.init.uniform_(weight.data, -bound, bound)
    elif method == 'kaiming_uniform':
        inits.kaiming_uniform(weight, fan=weight.size(-1), a=math.sqrt(5))
    elif method == 'zeros':
        inits.zeros(weight)
    elif method is None:
        inits.kaiming_uniform(weight, fan=weight.size(-1), a=math.sqrt(5))
    else:
        raise RuntimeError(f"Weight initializer '{method}' is not supported")

    if bias is not None:
        inits.zeros(bias)  # 这里也可以添加其他初始化方法


# 自定义的Linear函数，用于替换标准的线性变换
class Linear(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True,
                 weight_initializer='glorot',
                 bias_initializer='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        assert in_channels > 0
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        initialize_weights(self.weight, self.weight_initializer, self.bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


# 将原子列表嵌入进潜空间
class EmbeddingBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_atom_types=95, act=swish):
        super(EmbeddingBlock, self).__init__()
        self.act = act
        self.emb = nn.Embedding(num_atom_types, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        initialize_weights(self.emb.weight, method='uniform')

    def forward(self, x):
        return self.act(self.emb(x))


# 使用Linear中的方法，一个两层的，使用sigmoid函数作为激活函数的感知机
class TwoLayerLinear(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            middle_channels,
            out_channels,
            bias=False,
            act=False,
    ):
        super(TwoLayerLinear, self).__init__()
        self.lin1 = Linear(in_channels, middle_channels, bias=bias)
        self.lin2 = Linear(middle_channels, out_channels, bias=bias)
        self.act = act

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = self.lin1(x)
        if self.act:
            x = swish(x)
        x = self.lin2(x)
        if self.act:
            x = swish(x)
        return x


# 给普通的图神经网络加入一个边系数变量
class EdgeGraphConv(GraphConv):

    def message(self, x_j, edge_weight) -> Tensor:
        return x_j if edge_weight is None else edge_weight * x_j


# 用来处理由边信息、角度信息、二面角信息单层神经网络
class SimpleInteractionBlock(torch.nn.Module):
    def __init__(
            self,
            hidden_channels,
            middle_channels,
            num_radial,  # 径向基函数的数量
            num_spherical,  # 球面谐波的数量
            num_layers,
            output_channels,
            act=swish
    ):
        super(SimpleInteractionBlock, self).__init__()
        self.act = act

        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.lin_cat = Linear(2 * hidden_channels, hidden_channels)
        self.norm = GraphNorm(hidden_channels)

        # Transformations of Bessel and spherical basis representations.
        self.lin_feature1 = TwoLayerLinear(num_radial * num_spherical ** 2, middle_channels, hidden_channels)
        self.lin_feature2 = TwoLayerLinear(num_radial * num_spherical, middle_channels, hidden_channels)

        # Dense transformations of input messages.
        self.lin = Linear(hidden_channels, hidden_channels)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.final = Linear(hidden_channels, output_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.norm.reset_parameters()

        self.lin_feature1.reset_parameters()
        self.lin_feature2.reset_parameters()

        self.lin.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.lin_cat.reset_parameters()

        for lin in self.lins:
            lin.reset_parameters()

        self.final.reset_parameters()

    def forward(self, x, feature1, feature2, edge_index, batch):
        x = self.act(self.lin(x))

        feature1 = self.lin_feature1(feature1)  # [num_edges, hidden_channels]
        h1 = self.conv1(x, edge_index, feature1)
        h1 = self.lin1(h1)
        h1 = self.act(h1)

        feature2 = self.lin_feature2(feature2)  # [num_edges, hidden_channels]
        h2 = self.conv2(x, edge_index, feature2)
        h2 = self.lin2(h2)
        h2 = self.act(h2)

        h = self.lin_cat(torch.cat([h1, h2], 1))  # [num_nodes, hidden_channels]
        h = h + x  # Residual connection

        for lin in self.lins:
            h = self.act(lin(h)) + h  # Residual connection

        h = self.norm(h, batch)
        h = self.final(h)  # [num_nodes, output_channels]
        return h


class ComENetAutoEncoder(nn.Module):
    r"""
    修改后的 ComENet 模型，不使用电荷作为输入特征，仅使用原子类型，预测分子的能量等量子化学信息。

    Args:
        cutoff (float, optional): 原子间相互作用的截止距离。默认值：8.0
        num_layers (int, optional): 构建块的数量。默认值：4
        hidden_channels (int, optional): 隐藏嵌入大小。默认值：256
        middle_channels (int, optional): 两层线性块的中间嵌入大小。默认值：64
        out_channels (int, optional): 输出的物理量个数，如分子能量、HOMO、LUMO等。默认值：3
        atom_embedding_dim (int, optional): 原子嵌入向量的维度。默认值：128
        num_radial (int, optional): 径向基函数的数量。默认值：3
        num_spherical (int, optional): 球面谐波的数量。默认值：2
        num_output_layers (int, optional): 输出块的线性层数量。默认值：1
        transformer_layers (int, optional): Transformer的层数。默认值：1
        nhead_z (int, optional): Transformer中注意力头的数量。默认值：1
        device (str, optional): 设备类型。默认值：'cpu'
    """
    def __init__(
            self,
            cutoff=8.0,
            num_layers=4,
            hidden_channels=256,
            middle_channels=256,
            out_channels=3,  # 输出的物理量个数，如分子能量、HOMO、LUMO等
            atom_embedding_dim=128,  # atom_embedding_dim < hidden_channels
            num_radial=3,
            num_spherical=2,
            num_output_layers=1,  # 经过图神经网络后的隐藏层，再经过几层线性变换得到最后的输出
            transformer_layers=1,  # Transformer的层数
            nhead_z=1,
            device='cpu'
    ):
        super(ComENetAutoEncoder, self).__init__()
        self.out_channels = out_channels
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.atom_embedding_dim = atom_embedding_dim
        self.hidden_channels = hidden_channels
        self.transformer_layers = transformer_layers
        
        self.device = torch.device(device)

        act = swish
        self.act = act

        # 特征嵌入
        self.feature1 = torsion_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.feature2 = angle_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)

        # 实例化 EmbeddingBlock，仅使用原子类型嵌入
        self.emb_z = EmbeddingBlock(hidden_channels, num_atom_types=95, act=act)

        # 实例化多个交互块
        self.interaction_blocks = torch.nn.ModuleList(
            [
                SimpleInteractionBlock(
                    self.hidden_channels,
                    middle_channels,
                    num_radial,
                    num_spherical,
                    num_layers,
                    self.hidden_channels,
                    act,
                )
                for _ in range(num_layers)
            ]
        )

        # 经过图神经网络后再经过几层线性层得到输出
        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(Linear(self.hidden_channels, self.hidden_channels))

        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_channels, self.atom_embedding_dim)
        )

        # 解码器部分，使用与out_channels数相同的Transformer模型，用于预测分子的量子化学信息
        self.transformer_decoders_z = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.atom_embedding_dim, nhead=nhead_z, dim_feedforward=512, activation='gelu'),
                num_layers=transformer_layers
            )
            for _ in range(out_channels)
        ])

        self.output_layer = nn.Linear(self.atom_embedding_dim, 1)
        # self.output_layers_z = nn.ModuleList([
        #     nn.Linear(self.atom_embedding_dim, 1)  # 每个 Transformer 对应一个输出通道
        #     for _ in range(out_channels)
        # ])

        self.reset_parameters()

    def reset_parameters(self):
        self.emb_z.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

        # 编码器的初始化
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                initialize_weights(layer.weight, method='glorot')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Transformer 解码器和输出层的初始化
        for transformer in self.transformer_decoders_z:
            for param in transformer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

        initialize_weights(self.output_layer.weight, method='glorot')

    def _forward(self, data):
        """
        在图神经网络（GNN）中，输入的 `data` 对象通常是一个包含图结构及其相关特征的类实例，通常遵循以下格式：

        1. `batch`：一个一维张量，表示每个节点或边所属图的索引，便于在同一批次中处理多个图。
           例如，假设有三个图，`batch` 的内容可能是 `[0, 0, 1, 1, 2]`，表示前两个节点属于图0，接下来的两个节点属于图1，最后一个节点属于图2。

        2. `x`：一个二维张量，表示节点的特征。每一行对应一个节点的特征向量。
           例如，如果有五个节点，每个节点有三个特征，那么 `x` 的形状可能是 `(5, 3)`。

        3. `pos`：一个二维张量，表示节点的位置或坐标，形状通常为 `(num_nodes, 3)`，表示每个节点的三维坐标（如果是三维空间）或 `(num_nodes, 2)`（如果是二维空间）。

        4. 其他可能的属性：
           - `edge_index`：一个二维张量，表示图的边。通常是形状为 `(2, num_edges)` 的张量，其中每一列表示一个边的起始节点和终止节点的索引。
           - `edge_attr`：边的特征，形状通常为 `(num_edges, num_edge_features)`。
           - `y`：标签或目标值，通常用于监督学习任务，形状可能是 `(num_graphs, num_classes)`。

        示例格式：

        ```python
        class Data:
            def __init__(self, batch, x, pos, edge_index, edge_attr=None, y=None):
                self.batch = batch
                self.x = x
                self.pos = pos
                self.edge_index = edge_index
                self.edge_attr = edge_attr
                self.y = y

        # 示例数据
        batch = torch.tensor([0, 0, 1, 1, 2])  # 3个图
        x = torch.rand((5, 2))  # 5个节点，每个节点2个特征（z 和 charge）
        pos = torch.rand((5, 2))  # 5个节点的二维坐标
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])  # 边连接信息
        data = Data(batch, x, pos, edge_index)
        ```
        """
        batch = data.batch  # 对于单个分子，batch应为[0, 0, ..., 0]（原子数个）
        z = data.x[:, 0].long()  # 原子类型索引，确保为 Long
        pos = data.pos
        num_nodes = z.size(0)

        # 根据节点的空间位置（pos）和设定的半径（r），构建一个邻接关系
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        start_nodes, end_nodes = edge_index

        # 得到所有边的向量信息
        vecs = pos[start_nodes] - pos[end_nodes]
        dist = vecs.norm(dim=-1)  # 对每个边的向量计算 L2 范数（欧几里得距离）,产生一个一维张量，例如[1.0, 2.0, 0.5, 1.5, 0.7]

        # 嵌入块，仅使用 z
        x = self.emb_z(z)  # [num_nodes, hidden_channels]

        # 计算距离和角度等特征
        _, argmin0 = scatter_min(dist, end_nodes, dim_size=num_nodes)  # argmin0[i] 的值表示对于第 i 个节点与该节点距离最短的向量的索引
        argmin0[argmin0 >= len(end_nodes)] = 0  # 确保 argmin0 中的索引保持在有效范围内
        closest_end_index = start_nodes[argmin0]  # 每个节点中最近的点的索引
        add = torch.zeros_like(dist).to(dist.device)
        add[argmin0] = self.cutoff
        dist1 = dist + add  # 最终距离范围变为了[第二近的边的长度，到最短的边加截断半径的长度]，即除最短的加上了截断半径外，其他的不变

        _, argmin1 = scatter_min(dist1, end_nodes, dim_size=num_nodes)  # 第二小距离向量对应的索引
        argmin1[argmin1 >= len(end_nodes)] = 0
        second_closest_end_index = start_nodes[argmin1]  # 每个节点中第二近的点的索引

        # 与上面的一样，只不过换成了边向量中的起始点
        _, argmin0_j = scatter_min(dist, start_nodes, dim_size=num_nodes)
        argmin0_j[argmin0_j >= len(start_nodes)] = 0
        closest_start_index = end_nodes[argmin0_j]

        add_j = torch.zeros_like(dist).to(dist.device)
        add_j[argmin0_j] = self.cutoff
        dist1_j = dist + add_j

        _, argmin1_j = scatter_min(dist1_j, start_nodes, dim_size=num_nodes)
        argmin1_j[argmin1_j >= len(start_nodes)] = 0
        second_closest_start_index = end_nodes[argmin1_j]

        # 将原本按start_nodes顺序排布的closest_end_index更改为和end_nodes顺序一致
        # closest_end_index, n1 for end(i)
        closest_end_index = closest_end_index[end_nodes]
        second_closest_end_index = second_closest_end_index[end_nodes]

        # n0, n1 for start(j)
        closest_start_index = closest_start_index[start_nodes]
        second_closest_start_index = second_closest_start_index[start_nodes]

        # 计算角度
        mask_iref = closest_end_index == start_nodes
        iref = torch.clone(closest_end_index)
        iref[mask_iref] = second_closest_end_index[mask_iref]
        idx_iref = argmin0[end_nodes]
        idx_iref[mask_iref] = argmin1[end_nodes][mask_iref]

        mask_jref = closest_start_index == end_nodes
        jref = torch.clone(closest_start_index)
        jref[mask_jref] = second_closest_start_index[mask_jref]
        idx_jref = argmin0_j[start_nodes]
        idx_jref[mask_jref] = argmin1_j[start_nodes][mask_jref]

        pos_ji, pos_in0, pos_in1, pos_iref, pos_jref_j = (
            vecs,
            vecs[argmin0][end_nodes],
            vecs[argmin1][end_nodes],
            vecs[idx_iref],
            vecs[idx_jref]
        )

        # 计算角度 theta
        a = ((-pos_ji) * pos_in0).sum(dim=-1)
        b = torch.cross(-pos_ji, pos_in0).norm(dim=-1)
        theta = torch.atan2(b, a)
        theta[theta < 0] = theta[theta < 0] + math.pi

        # 计算扭转角 phi
        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(-pos_ji, pos_in0)
        plane2 = torch.cross(-pos_ji, pos_in1)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        phi = torch.atan2(b, a)
        phi[phi < 0] = phi[phi < 0] + math.pi

        # 计算右扭转角 tau
        plane1 = torch.cross(pos_ji, pos_jref_j)
        plane2 = torch.cross(pos_ji, pos_iref)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        tau = torch.atan2(b, a)
        tau[tau < 0] = tau[tau < 0] + math.pi

        # 每个样本的输出张量表示基于输入的距离 dist、扭转角 theta 和 phi 的特征表示。
        feature1 = self.feature1(dist, theta, phi)  # [num_edges, n * k]
        feature2 = self.feature2(dist, tau)  # [num_edges, n ** 2 * k]

        # 交互块，x为表示原子信息的数据z生成的嵌入张量
        for interaction_block in self.interaction_blocks:
            x = interaction_block(x, feature1, feature2, edge_index, batch)

        # 在输出前再经过指定个线性层
        for lin in self.lins:
            x = self.act(lin(x))

        # 计算分子能量（如果需要的话，可以保留）
        energy = scatter(x, batch, dim=0)
        energy = energy.view(-1, self.hidden_channels)
        return x, energy  # 返回原子嵌入和分子能量，x的尺寸为(num_nodes, hidden_channels)，energy的尺寸为(hidden_channels,)

    def forward(self, data):
        atomic_embeddings, molecular_properties = self._forward(data)

        # 编码器获取原子嵌入向量
        embeddings = self.encoder(atomic_embeddings)  # [num_nodes, atom_embedding_dim]

        # 按分子分组原子嵌入向量
        embeddings_split = torch.split(
            embeddings,
            tuple((data.batch == i).sum().item() for i in range(data.num_graphs))
        )

        # 找到每个分子中原子的最大数量，用于填充
        max_num_atoms = max([emb.size(0) for emb in embeddings_split])

        # 填充每个分子的原子嵌入向量，使其具有相同的长度
        embeddings_padded = torch.zeros(
            max_num_atoms,
            data.num_graphs,
            self.atom_embedding_dim,
            device=self.device
        )
        for i, emb in enumerate(embeddings_split):
            embeddings_padded[:emb.size(0), i, :] = emb

        # 初始化一个列表来存储所有Transformer的合并输出
        all_concatenated_outputs = []

        # 使用多个Transformer解码器分别预测每个量子化学性质
        if self.transformer_layers >=1 :
            for transformer in self.transformer_decoders_z:
                # Transformer期望的输入形状为 [seq_length, batch_size, embedding_dim]
                transformer_output = transformer(embeddings_padded)  # [max_num_atoms, batch_size, atom_embedding_dim]

                # 创建掩码以忽略填充的部分
                mask = torch.zeros_like(embeddings_padded[:, :, 0], dtype=torch.bool, device=self.device)  # [max_num_atoms, batch_size]
                for i, emb in enumerate(embeddings_split):
                    mask[:emb.size(0), i] = 1

                # 将填充的部分设置为0
                transformer_output = transformer_output * mask.unsqueeze(-1).float()  # [max_num_atoms, batch_size, atom_embedding_dim]

                # 将 Transformer 输出转置为 [batch_size, max_num_atoms, atom_embedding_dim]
                transformer_output = transformer_output.transpose(0, 1)  # [batch_size, max_num_atoms, atom_embedding_dim]
                mask = mask.transpose(0, 1)  # [batch_size, max_num_atoms]

                # 使用掩码选择有效的输出
                transformer_output = transformer_output.reshape(-1, self.atom_embedding_dim)  # [batch_size * max_num_atoms, atom_embedding_dim]
                mask = mask.reshape(-1)  # [batch_size * max_num_atoms]

                # 选择 mask 为 True 的部分，保持顺序
                transformer_output = transformer_output[mask]  # [total_valid_atoms, atom_embedding_dim]
        else:
            transformer_output = embeddings
        # 通过输出层得到最终预测
        output = self.output_layer(transformer_output)  # [total_valid_atoms, 1]
        all_concatenated_outputs.append(output)

        # 将所有 Transformer 的输出合并
        final_output = torch.cat(all_concatenated_outputs, dim=0)  # [total_valid_atoms, 1]

        # 确保输出形状与目标一致
        assert final_output.size(0) == data.y.size(0), (
            f"Final output size {final_output.size(0)} does not match target size {data.y.size(0)}"
        )

        return embeddings, final_output
