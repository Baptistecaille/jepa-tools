from __future__ import annotations

import torch
import torch.nn as nn

from jepa_tools.core import registry
from jepa_tools.core.types import StepOutputs

# Importing the modules runs their @register decorators.
from . import distances, regularizers  # noqa: F401

class Objective(nn.Module):

    def __init__(self, distance: str = "l2", regularizer: str = "none", distance_weight:float = 1.0, regularizer_weight:float = 1.0, **regularizer_kwargs):
        super().__init__()
        self.distance = registry.get("distance", distance)()
        self.regularizer = registry.get("regularizer", regularizer)(**regularizer_kwargs)
        self.distance_weight = distance_weight
        self.regularizer_weight = regularizer_weight
    
    def forward(self, out: StepOutputs) -> tuple[torch.Tensor, dict[str, float]]:
        distance_loss = self.distance(out.pred, out.target)
        regularizer_loss = self.regularizer(out)
        loss = self.distance_weight * dist + self.reg_weight * reg
        logs = {
            "loss": loss.detach().item(),
            "distance": dist.detach().item(),
            "reg": reg.detach().item(),
        }
        return loss, logs

    @property
    def has_regularizer(self) -> bool:
        return self.regularizer.active

    