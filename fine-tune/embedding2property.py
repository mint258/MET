import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    """Apply a consistent initialization scheme across transformer layers."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.uniform_(module.weight, -0.05, 0.05)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class CustomMultiHeadSelfAttention(nn.Module):
    def __init__(self, atom_embedding_dim, num_heads=4):
        """A lightweight multi-head self-attention block for molecule embeddings."""
        super().__init__()
        assert atom_embedding_dim % num_heads == 0, "atom_embedding_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = atom_embedding_dim // num_heads

        self.W_q = nn.Linear(atom_embedding_dim, atom_embedding_dim)
        self.W_k = nn.Linear(atom_embedding_dim, atom_embedding_dim)
        self.W_v = nn.Linear(atom_embedding_dim, atom_embedding_dim)
        self.fc = nn.Linear(atom_embedding_dim, atom_embedding_dim)

        self.apply(initialize_weights)

    def forward(self, x, mask=None):
        batch_size, max_num_nodes, atom_embedding_dim = x.size()

        q = self.W_q(x).view(batch_size, max_num_nodes, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, max_num_nodes, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, max_num_nodes, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask.bool(), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, max_num_nodes, atom_embedding_dim)
        return self.fc(attn_output)


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, atom_embedding_dim, num_heads=4, dim_feedforward=256, dropout=0.1):
        """Transformer encoder layer built on the custom self-attention block."""
        super().__init__()
        self.self_attn = CustomMultiHeadSelfAttention(atom_embedding_dim, num_heads=num_heads)
        self.linear1 = nn.Linear(atom_embedding_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, atom_embedding_dim)

        self.norm1 = nn.LayerNorm(atom_embedding_dim)
        self.norm2 = nn.LayerNorm(atom_embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

        self.apply(initialize_weights)

    def forward(self, src, mask=None):
        attn_output = self.self_attn(src, mask)
        src = self.norm1(attn_output + src)
        src = self.dropout1(src)

        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(ff_output + src)
        src = self.dropout2(src)
        return src


class MolecularTransformer(nn.Module):
    def __init__(
        self,
        atom_embedding_dim=128,
        num_layers=6,
        num_heads=4,
        dim_feedforward=256,
        dropout=0.1,
        output_dim=1,
        num_linear_layers=2,
        min_dim=32,
    ):
        """Predict molecular properties from atom-level embeddings."""
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [
                CustomTransformerEncoderLayer(
                    atom_embedding_dim,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        linear_layers = []
        current_dim = atom_embedding_dim
        for _ in range(num_linear_layers):
            next_dim = max(current_dim // 2, min_dim)
            linear_layers.append(nn.Linear(current_dim, next_dim))
            linear_layers.append(nn.LeakyReLU())
            current_dim = next_dim

        self.linear_layers = nn.Sequential(*linear_layers)
        self.global_attn = nn.Linear(current_dim, 1)
        self.fc = nn.Linear(current_dim, output_dim)

        self.apply(initialize_weights)

    def forward(self, x, mask):
        out = x
        for layer in self.encoder_layers:
            out = layer(out, mask)

        out = self.linear_layers(out)
        scores = self.global_attn(out)
        scores = scores.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))
        attn_weights = F.softmax(scores, dim=1)
        out = (out * attn_weights).sum(dim=1)
        return self.fc(out)
