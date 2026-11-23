from typing import Callable, Type
import torch.nn.functional as F


# Global register which maps distance names (str) to their classes
DISTANCE: Dict[str, Type["BaseDistance"]] = {}

def register_distance(name: str) -> Callable[[Type["BaseDistance"]], Type["BaseDistance"]]:

    """Decorator to register a distance class with a given name."""
    
    def decorator(cls: Type["BaseDistance"]) -> Type["BaseDistance"]:
        DISTANCE[name] = cls
        return cls
    return decorator


def _check_shapes(pred, target) -> None:

    """Check that the prediction and target have compatible shapes."""
    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction and target must have the same shape, "
            f"got {pred.shape} vs {target.shape}"
        )

class BaseDistance:

    """Base class for distance metrics."""

    # Id of the distance metric
    name:str

    # Callable interface returning a float distance between two spaces
    def __call__(self, pred, target) -> float:
        raise NotImplementedError("Distance metrics must implement the __call__ method.")

@register_distance("l2")
class L2(BaseDistance):

    """L2 distance metric."""
    name = "l2"

    def __call__(self, pred, target) -> float:
        _check_shapes(pred, target)
        return F.mse_loss(pred, target).item()

@register_distance("smooth_l1")
class SmoothL1(BaseDistance):

    """Smooth L1 distance metric."""
    name = "smooth_l1"

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def __call__(self, pred, target) -> float:
        _check_shapes(pred, target)
        return F.smooth_l1_loss(pred, target, beta=self.beta).item()

@register_distance("cosine")
class Cosine(BaseDistance):

    """Cosine distance metric."""
    name = "cosine"

    def __call__(self, pred, target) -> float:
        _check_shapes(pred, target)
        return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean().item()
