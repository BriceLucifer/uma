from __future__ import annotations
import mlx.core as mx
from .module import Module

class Dropout(Module):
    """Inverted dropout regularization.

    Randomly zeroes elements with probability ``p`` during training and
    scales surviving elements by ``1/(1-p)``. Identity during eval.

    Args:
        p: Drop probability. Default ``0.5``.
    """

    def __init__(self, p: float = 0.5) -> None:
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout p must be in [0, 1), got {p}")
        self.p = p
        self.training: bool = True

    def forward(self, x: mx.array) -> mx.array:
        if not self.training:
            return x
        mask = mx.random.uniform(shape=x.shape) > self.p
        return x * mask / (1 - self.p)