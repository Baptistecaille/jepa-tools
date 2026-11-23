# Next steps
# Let the user define between bf16 and float32 for the regularizer, but default to float32 for now

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from jepa_tools.core.registry import register
from jepa_tools.core.types import StepOutputs

def _flatten_emb(emb:Tensor) -> torch.Tensor:

    """ Data shape (B, N, D) -> (B*N, D) in float 32 to avoid bf16 problems but slower """
    return emb.reshape(-1, emb.shape[-1]).float()

def _off_diagonal(x:Tensor) -> torch.Tensor:

    """Return a flattened view of the off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m, "Input must be a square matrix."

    # x.flatten()[:-1] remove the last element (which is on the diagonal)
    # .view(n-1, n+1) creates a array of 2 rows and 4 columns, 
    """         col 0   col 1   col 2   col 3
    ligne 0   : [ 00 ]    01      02      10
    ligne 1   : [ 11 ]    12      20      21"""
    # Therefore all the diagonal values are in the first column.
    # We delete it with [:, 1:] and flatten the result to get a 1D array of off-diagonal elements.
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


@register("regularizer", "none")
class NoRegularizer:
    """No regularization applied."""
    active = False

    def __call__(self, out:StepOutputs) -> Tensor:
        return out.emb.new_zeros(())


@register("regularizer", "vicreg")
class VicReg:
    """VicReg regularization term."""

    active = True

    def __init__(self, mu_var: float = 25.0, nu_cov: float = 1.0, gamma: float = 1.0, eps: float = 1e-4):
        self.mu_var = mu_var
        self.nu_cov = nu_cov
        self.gamma = gamma
        self.eps = eps

    def __call__(self, out:StepOutputs) -> Tensor:
        emb = _flatten_emb(out.emb)  # (B*N, D)
        # Compute the mean and standard deviation of the emb

        z = _flatten_emb(out.emb)  # (B*N, D)
        n, dim = z.shape
        std = torch.sqrt(z.var(dim=0, unbiased=False) + self.eps) 
        var_loss = F.relu(self.gamma - std).mean()
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / max(n - 1, 1)
        cov_loss = off_diagonal(cov).pow(2).sum()  dim if dim > 1 else z.new_zeros(())
        return self.mu_var * var_loss + self.mu_cov * cov_loss
    