from __future__ import annotations
import mlx.core as mx
from .base import Optimizer

class SGD(Optimizer):
    """Stochastic Gradient Descent.

    Update rule: ``θ ← θ - lr * ∇θ``

    Args:
        params: Parameter dict from ``model.parameters()``.
        lr: Learning rate.
    """

    def step(self, grads: dict[str, mx.array]) -> dict[str, mx.array]:
        return {
            name: param - self.lr * grads[name]
            for name, param in self.params.items()
            if name in grads
        }