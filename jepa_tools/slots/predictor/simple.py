from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from jepa_tools.core.registry import register

def _reject_cond(cond: Tensor | None) -> None:
    # `cond` (actions, language...) is reserved in the interface, not implemented yet.
    if cond is not None:
        raise NotImplementedError("cond is reserved for future conditioning and not implemented")


@register("predictor", "identity")
class IdentityPredictor(nn.Module):
    """ Returns the context tokens unchanged
        Basicly the identity function f: x -> x
    """

    def forward( self, ctx_tokens: Tensor, ctx_idx: torch.LongTensor | None = None, tgt_idx: torch.LongTensor | None = None, cond: Tensor | None = None,) -> Tensor :
        _reject_cond(cond)
        return ctx_tokens

@register("predictor", "mlp")
class MLPPredictor(nn.Module):

    def __init__(self, dim: int, hidden: int = 256, depth: int = 2):
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

    def forward( self, ctx_tokens: Tensor, ctx_idx: torch.LongTensor | None = None, tgt_idx: torch.LongTensor | None = None, cond: Tensor | None = None,) -> Tensor :
        _reject_cond(cond)
        if tgt_idx is not None and tgt_idx.shape[1] != ctx.tokens.shape[1]:
            raise ValueError(
                f"mlp predictor keeps the token count: got {ctx_tokens.shape[1]} context "
                f"tokens but tgt_idx asks for {tgt_idx.shape[1]}"
            )
        return self.net(ctx_tokens)
        