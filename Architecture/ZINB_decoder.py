import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

class ZINBDecoder(nn.Module):
    """
    Zero-Inflated NB decoder with an arbitrary stack of hidden layers.

    Parameters
    ----------
    in_dim       : int
        Dimensionality of the latent space z.
    hidden_dims  : Sequence[int]
        Width of each hidden layer, e.g. [256, 128, 64].  Empty list ⇒ no hidden.
    n_genes      : int
        Number of genes (output features).
    dropout      : float
        Drop probability applied after each hidden layer.
    """

    def __init__(self,
                 in_dim: int,
                 hidden_dims: Sequence[int],
                 n_genes: int,
                 dropout: float = 0.1):
        super().__init__()

        layers = []
        prev = in_dim
        for h in hidden_dims:                 # build the flexible torso
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev = h

        self.hidden = nn.Sequential(*layers)  # may be empty

        # ----- output heads -------------------------------------------------
        self.pi   = nn.Linear(prev, n_genes)   # extra-zero probability π
        self.disp = nn.Linear(prev, n_genes)   # NB dispersion θ
        self.mean = nn.Linear(prev, n_genes)   # NB mean μ

        self.softplus_clip = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.exp_clip      = lambda x: torch.clamp(torch.exp(x),  1e-5, 1e6)

    # ----------------------------------------------------------------------
    def forward(self, z):
        """
        z : (N, in_dim) latent vectors
        returns π, θ, μ each (N, n_genes)
        """
        h = self.hidden(z) if self.hidden else z   # skip if no hidden layers
        pi    = torch.sigmoid(self.pi(h))
        theta = self.softplus_clip(self.disp(h))    # should be positive but not large as mean
        mu    = self.exp_clip(self.mean(h))         # mean should be a positive real number
        return pi, theta, mu

