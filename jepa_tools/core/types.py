"""Data contract shared by every slot (see the design spec, section 5.1).

Indices designate token positions in the *full* grid of the input.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import LongTensor, Tensor


@dataclass
class Plan:
    """Produced by a Sampler."""

    ctx_inputs: Tensor  # what the context encoder sees
    tgt_inputs: Tensor  # what the target sees (may be identical to ctx_inputs)
    ctx_idx: LongTensor | None  # (B, Nc) visible context tokens, None = all
    tgt_idx: LongTensor | None  # (B, Nt) tokens to predict, None = all


@dataclass
class StepOutputs:
    """Passed to the Objective."""

    pred: Tensor  # (B, Nt, D)
    target: Tensor  # (B, Nt, D)
    emb: Tensor  # (B, Nc, D) online context embeddings
    progress: float  # 0 to 1, for schedules
