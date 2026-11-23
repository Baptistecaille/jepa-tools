from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from jepa_tools.core.registry import register


def _check_shapes(pred: Tensor, target: Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {tuple(pred.shape)} vs {tuple(target.shape)}"
        )

@register("distance", "l2")
class L2:
    """L2 distance metric."""

    def __call__(self, pred, target) -> Tensor:
        _check_shapes(pred, target)
        return F.mse_loss(pred, target)

@register("distance", "smooth_l1")
class SmoothL1:
    """Smooth L1 distance metric."""


    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def __call__(self, pred, target) -> Tensor:
        _check_shapes(pred, target)
        return F.smooth_l1_loss(pred, target, beta=self.beta)

@register("distance", "cosine")
class Cosine:
    """Cosine distance metric."""

    def __call__(self, pred, target) -> Tensor:
        _check_shapes(pred, target)
        return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
