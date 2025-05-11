# ▶️  install PyG if you haven't already
# pip install torch torch_geometric torch_sparse torch_scatter

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv           # multi-head Graph Attention
from torch_geometric.data import Data
from typing import Optional
#from data_reader import gene_matrix
#from make_adj import adj_radius

# ────────────────────────────────────────────────────────────────────────────
# 1. helpers:  CSR  →  edge_index
# ────────────────────────────────────────────────────────────────────────────
def csr_to_edge_index(adj_csr, to_tensor=True, device="cpu"):
    """
    Convert a SciPy CSR adjacency (symmetric, incl. self-loops) to the
    edge_index format expected by PyG: shape (2, |E|)
    """
    src, dst = adj_csr.nonzero()
    edge_index = np.vstack([src, dst])           # shape (2, |E|)
    if to_tensor:
        edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    return edge_index

# ────────────────────────────────────────────────────────────────────────────
# 2.  Encoder definition
# ────────────────────────────────────────────────────────────────────────────
class GATEncoder(nn.Module):
    """
    Stack of GATConv layers → latent_dim
    Parameters
    ----------
    n_feat        : # input genes (G)
    hidden_dims   : list[int] – hidden channels per GAT layer
    heads         : list[int] – attention heads per layer
    latent_dim    : int       – final output dimension
    dropout       : float
    """
    def __init__(self,
                 n_feat: int,
                 hidden_dims: list[int],
                 heads: Optional[list[int]] = None,
                 latent_dim: int = 16,
                 dropout: float = 0.2):
        super().__init__()

        if heads is None:
            heads = [8] * len(hidden_dims)

        dims = [n_feat] + hidden_dims
        assert len(heads) == len(hidden_dims), "`heads` must match hidden layers"

        convs = []
        for l, (in_ch, out_ch, n_head) in enumerate(zip(dims[:-1], hidden_dims, heads)):
            convs.append(
                GATConv(in_channels=in_ch,
                        out_channels=out_ch // n_head,
                        heads=n_head,
                        concat=True,
                        dropout=dropout,
                        add_self_loops=False)         # our adj already has them
            )
        # last layer → latent_dim (single head, no concat)
        convs.append(
            GATConv(in_channels=hidden_dims[-1],
                    out_channels=latent_dim,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=False)
        )
        self.convs = nn.ModuleList(convs)
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.ELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        x : (N, G)   – gene matrix
        edge_index : (2, |E|)
        returns
        -------
        z : (N, latent_dim)
        """
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)
        z = self.convs[-1](x, edge_index)          # no activation here
        return z


# device     = "cuda" if torch.cuda.is_available() else "cpu"
# x_tensor   = torch.tensor(gene_matrix, dtype=torch.float32, device=device)
# edge_index = csr_to_edge_index(adj_radius, device=device)

# # --(c) build and run encoder -----------------------------------------------
# encoder = GATEncoder(n_feat      = gene_matrix.shape[1],     # G
#                      hidden_dims = [128, 64],
#                      heads       = [8, 8],            # → 128 & 64 channels
#                      latent_dim  = 16,
#                      dropout     = 0.3).to(device)

# encoder.train()                        # or .eval() when inferring
# z = encoder(x_tensor, edge_index)      # z : (N, 16)

# print("latent shape:", z.shape)        # (num_cells, 16)
