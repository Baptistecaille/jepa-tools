import torch
import torch.nn.functional as F
from typing import Callable, Type

# import personnal classes
from .distances import *
from .regularizers import *

class Objective(nn.Module):

    def __init__(self, distance: str = "l2", regularizer: str = "none", distance_weight:float = 1.0, regularizer_weight:float = 1.0, **regularizer_kwargs):
        super().__init__()
        self.distance = registry.get("distance", distance)()
        self.regularizer = registry.get("regularizer", regularizer)(**regularizer_kwargs)
        self.distance_weight = distance_weight
        self.regularizer_weight = regularizer_weight
    
    def forward(self, out: StepOutputs) -> tuple[torch.Tensor, dict[str, float]]:
        distance_loss = self.distance(out.pred, out.target)
        regularizer_loss = self.regularizer(out.emb)
        total_loss = self.distance_weight * distance_loss + self.regularizer_weight * regularizer_loss
        return total_loss

    @property
    def has_regularizer(self) -> bool:
        return self.regularizer.active