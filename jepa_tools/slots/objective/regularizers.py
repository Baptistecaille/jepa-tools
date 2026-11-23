# Next steps
# Let the user define between bf16 and float32 for the regularizer, but default to float32 for now

from typing import Callable, Type
import torch.nn.functional as F

# Global register which maps regularizer names (str) to their classes
REGULARIZER: Dict[str, Type["BaseRegularizer"]] = {}

# Create a decorator to register regularizer classes
def register_regularizer(name: str) -> Callable[[Type["BaseRegularizer"]], Type["BaseRegularizer"]]:

    """Decorator to register a regularizer class with a given name."""
    
    def decorator(cls: Type["BaseRegularizer"]) -> Type["BaseRegularizer"]:
        REGULARIZER[name] = cls
        return cls
    return decorator

def _flatten_emb(emb) -> torch.Tensor:

    """ Data shape (B, N, D) -> (B*N, D) in float 32 to avoid bf16 problems but slower """
    return emb.reshape(-1, emb.shape[-1]).float()

def _off_diagonal(x) -> torch.Tensor:

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

class BaseRegularizer:

    """Base class for regularizers."""

    # Id of the regularizer
    name:str
    active:bool


    # Callable interface returning a float regularization term for emb
    def __call__(self, emb) -> float:
        raise NotImplementedError("Regularizers must implement the __call__ method.")


@register_regularizer("none")
class NoRegularizer:

    """No regularization applied."""
    name = "none"
    active = False

    def __call__(self, emb) -> float:
        return out.emb.new_zeros(())

@register_regularizer("vicreg")
class VicReg(BaseRegularizer):

    """VicReg regularization term."""
    name = "vicreg"
    active = True

    def __init__(self, mu_var: float = 25.0, nu_cov: float = 1.0, gamma: float = 1.0, eps: float = 1e-4):
        self.mu_var = mu_var
        self.nu_cov = nu_cov
        self.gamma = gamma
        self.eps = eps

    def __call__(self, emb) -> float:
        emb = _flatten_emb(emb)
        # Compute the mean and standard deviation of the emb
        mean = emb.mean(dim=0)
        std = emb.std(dim=0)

        # Compute the variance loss
        var_loss = torch.mean(F.relu(1 - std))

        # Compute the covariance loss
        cov_matrix = (emb - mean).T @ (emb - mean) / (emb.shape[0] - 1)
        cov_loss = torch.sum(_off_diagonal(cov_matrix) ** 2)

        # Compute the similarity loss
        sim_loss = F.mse_loss(emb, torch.zeros_like(emb))

        # Combine the losses with their respective coefficients
        total_loss = self.mu_var * var_loss + self.nu_cov * cov_loss + self.gamma * sim_loss
        return total_loss.item()

