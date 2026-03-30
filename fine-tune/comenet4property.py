# -*- coding: utf-8 -*-

from torch_geometric.nn import GraphConv, GraphNorm
from torch_geometric.nn import inits, global_mean_pool

from features import angle_emb, torsion_emb

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

import math
from math import sqrt
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from graph_compat import radius_graph, scatter, scatter_min

try:
    import sympy as sym
except ImportError:
    sym = None


def swish(x):
    return x * torch.sigmoid(x)


def initialize_weights(weight, method='glorot', bias=None):
    """Documentation available in the manuscript and code structure."""
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
        inits.zeros(bias)


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


class EdgeGraphConv(GraphConv):

    def message(self, x_j, edge_weight) -> Tensor:
        return x_j if edge_weight is None else edge_weight * x_j


class SimpleInteractionBlock(torch.nn.Module):
    def __init__(
            self,
            hidden_channels,
            middle_channels,
            num_radial,
            num_spherical,
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
    """ComENet encoder-decoder used throughout the MET workflow."""
    def __init__(
            self,
            cutoff=8.0,
            num_layers=4,
            hidden_channels=256,
            middle_channels=256,
            out_channels=3,
            atom_embedding_dim=128,  # atom_embedding_dim < hidden_channels
            num_radial=3,
            num_spherical=2,
            num_output_layers=1,
            device='cpu'
    ):
        super(ComENetAutoEncoder, self).__init__()
        self.out_channels = out_channels
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.atom_embedding_dim = atom_embedding_dim
        self.hidden_channels = hidden_channels

        self.device = torch.device(device)

        act = swish
        self.act = act

        self.feature1 = torsion_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.feature2 = angle_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)

        self.emb_z = EmbeddingBlock(hidden_channels, num_atom_types=95, act=act)

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

        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(Linear(self.hidden_channels, self.hidden_channels))

        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_channels, self.atom_embedding_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.atom_embedding_dim, 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.emb_z.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                initialize_weights(layer.weight, method='glorot')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    def _forward(self, data):
        """Encode a 3D molecular graph and return atom-level representations."""
        batch = data.batch
        z = data.x[:, 0].long()
        pos = data.pos
        num_nodes = z.size(0)

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        start_nodes, end_nodes = edge_index

        vecs = pos[start_nodes] - pos[end_nodes]
        dist = vecs.norm(dim=-1)

        x = self.emb_z(z)  # [num_nodes, hidden_channels]

        _, argmin0 = scatter_min(dist, end_nodes, dim_size=num_nodes)
        argmin0[argmin0 >= len(end_nodes)] = 0
        closest_end_index = start_nodes[argmin0]
        add = torch.zeros_like(dist).to(dist.device)
        add[argmin0] = self.cutoff
        dist1 = dist + add

        _, argmin1 = scatter_min(dist1, end_nodes, dim_size=num_nodes)
        argmin1[argmin1 >= len(end_nodes)] = 0
        second_closest_end_index = start_nodes[argmin1]

        _, argmin0_j = scatter_min(dist, start_nodes, dim_size=num_nodes)
        argmin0_j[argmin0_j >= len(start_nodes)] = 0
        closest_start_index = end_nodes[argmin0_j]

        add_j = torch.zeros_like(dist).to(dist.device)
        add_j[argmin0_j] = self.cutoff
        dist1_j = dist + add_j

        _, argmin1_j = scatter_min(dist1_j, start_nodes, dim_size=num_nodes)
        argmin1_j[argmin1_j >= len(start_nodes)] = 0
        second_closest_start_index = end_nodes[argmin1_j]

        # closest_end_index, n1 for end(i)
        closest_end_index = closest_end_index[end_nodes]
        second_closest_end_index = second_closest_end_index[end_nodes]

        # n0, n1 for start(j)
        closest_start_index = closest_start_index[start_nodes]
        second_closest_start_index = second_closest_start_index[start_nodes]

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

        a = ((-pos_ji) * pos_in0).sum(dim=-1)
        b = torch.cross(-pos_ji, pos_in0).norm(dim=-1)
        theta = torch.atan2(b, a)
        theta[theta < 0] = theta[theta < 0] + math.pi

        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(-pos_ji, pos_in0)
        plane2 = torch.cross(-pos_ji, pos_in1)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        phi = torch.atan2(b, a)
        phi[phi < 0] = phi[phi < 0] + math.pi

        plane1 = torch.cross(pos_ji, pos_jref_j)
        plane2 = torch.cross(pos_ji, pos_iref)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        tau = torch.atan2(b, a)
        tau[tau < 0] = tau[tau < 0] + math.pi

        feature1 = self.feature1(dist, theta, phi)  # [num_edges, n * k]
        feature2 = self.feature2(dist, tau)  # [num_edges, n ** 2 * k]

        for interaction_block in self.interaction_blocks:
            x = interaction_block(x, feature1, feature2, edge_index, batch)

        for lin in self.lins:
            x = self.act(lin(x))

        energy = scatter(x, batch, dim=0)
        energy = energy.view(-1, self.hidden_channels)
        return x, energy

    def forward(self, data):
        atomic_embeddings, molecular_properties = self._forward(data)

        embeddings = self.encoder(atomic_embeddings)  # [num_nodes, atom_embedding_dim]

        molecular_embeddings = global_mean_pool(embeddings, data.batch)  # [batch_size, atom_embedding_dim]

        final_output = self.decoder(molecular_embeddings)  # [batch_size, 1]

        assert final_output.size(0) == data.y.size(0), (
            f"Final output size {final_output.size(0)} does not match target size {data.y.size(0)}"
        )

        return molecular_embeddings, final_output.squeeze(-1)
