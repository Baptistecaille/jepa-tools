from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from jepa_tools.core.registry import register

@register("encoder", "mlp")
class MLPEncoder(nn.Module):

    def __init__(self, in_dim:int, dim:int, hidden:int = 256, depth:int = 2):

        super().__init__() 
        if depth < 1:
            raise ValueError(f"Depth must be >= 1 got {depth}")
        
        sizes = [in_dim] + [hidden] * (dim - 1) + [dim] # List of layers in the model
        layers: [nn.Module] = []
        for i in range(depth):
            layers.append(nn.Linear(sizes[i], sizes[i + 1])) # Size of Linear layers must be compatible. 
        # No activation after the last layer.
        if i < depth - 1:
            layers.append(nn.GELU())
        
        self.net = nn.Sequential(*layers)
        self.dim = dim

    def forward(self, x:Tensor, deep_idx:torch.longTensor | None = None) -> Tensor:
        # keep_idx is part of the Encoder interface but is ignored: there is only
        # one token, nothing to select.
        # (B, ...) -> (B, in_dim) -> (B, dim) -> (B, 1, dim)
        return self.net(x.reshape(x.shape[0], -1)).unsqueeze(1)
        