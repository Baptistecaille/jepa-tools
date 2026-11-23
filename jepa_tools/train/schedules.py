""" Pure scheduling functions """

from __future__ import annotations

import math

def momentum_at(progress: float, start: float, end: float) -> float:
    """Linear interpolation from ``start`` to ``end`` as progress goes 0 -> 1."""
    progress = min(max(progress, 0.0), 1.0)
    return start + (end - start) * progress

print(momentum_at(2, 0.0, 1.0))