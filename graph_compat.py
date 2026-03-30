from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch_geometric.utils import scatter as pyg_scatter

try:
    from torch_cluster import radius_graph as _radius_graph
except ImportError:  # pragma: no cover - exercised only when torch_cluster is unavailable.
    _radius_graph = None

try:
    from torch_scatter import scatter as _scatter
    from torch_scatter import scatter_mean as _scatter_mean
    from torch_scatter import scatter_min as _scatter_min
except ImportError:  # pragma: no cover - exercised only when torch_scatter is unavailable.
    _scatter = None
    _scatter_mean = None
    _scatter_min = None


def radius_graph(pos: torch.Tensor, r: float, batch: Optional[torch.Tensor] = None, loop: bool = False) -> torch.Tensor:
    if _radius_graph is not None:
        return _radius_graph(pos, r=r, batch=batch, loop=loop)

    if batch is None:
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

    edge_pieces = []
    for graph_id in batch.unique(sorted=True):
        node_indices = torch.nonzero(batch == graph_id, as_tuple=False).view(-1)
        graph_pos = pos[node_indices]
        dist = torch.cdist(graph_pos, graph_pos, p=2)
        mask = dist <= r
        if not loop:
            mask &= ~torch.eye(graph_pos.size(0), dtype=torch.bool, device=graph_pos.device)
        row, col = torch.nonzero(mask, as_tuple=True)
        if row.numel() == 0:
            continue
        edge_pieces.append(torch.stack([node_indices[row], node_indices[col]], dim=0))

    if not edge_pieces:
        return torch.empty((2, 0), dtype=torch.long, device=pos.device)
    return torch.cat(edge_pieces, dim=1)


def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None, reduce: str = "sum"):
    if _scatter is not None:
        return _scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)
    return pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None):
    if _scatter_mean is not None:
        return _scatter_mean(src, index, dim=dim, dim_size=dim_size)
    return pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce="mean")


def scatter_min(src: torch.Tensor, index: torch.Tensor, dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if _scatter_min is not None:
        return _scatter_min(src, index, dim_size=dim_size)

    if src.dim() != 1 or index.dim() != 1:
        raise ValueError("Fallback scatter_min only supports 1D src and index tensors.")

    dim_size = dim_size if dim_size is not None else int(index.max().item()) + 1
    min_values = torch.full((dim_size,), float("inf"), dtype=src.dtype, device=src.device)
    argmin = torch.full((dim_size,), src.numel(), dtype=torch.long, device=index.device)

    for i in range(src.numel()):
        dst = int(index[i].item())
        value = src[i]
        if value < min_values[dst]:
            min_values[dst] = value
            argmin[dst] = i

    min_values[torch.isinf(min_values)] = 0
    return min_values, argmin
