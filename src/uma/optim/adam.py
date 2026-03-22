from __future__ import annotations
import mlx.core as mx
from .base import Optimizer

class Adam(Optimizer):
    """Adam optimiser with bias-corrected moment estimates.

    Update rule::

        m ← β₁·m + (1-β₁)·g
        v ← β₂·v + (1-β₂)·g²
        m̂ = m / (1 - β₁ᵗ)
        v̂ = v / (1 - β₂ᵗ)
        θ ← θ - lr · m̂ / (√v̂ + ε)

    Args:
        params: Parameter dict from ``model.parameters()``.
        lr: Learning rate. Default ``1e-3``.
        betas: Coefficients for first and second moment estimates. Default ``(0.9, 0.999)``.
        eps: Numerical stability term. Default ``1e-8``.
    """

    def __init__(
        self,
        params: dict[str, mx.array],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.t: int = 0
        self.m: dict[str, mx.array] = {
            k: mx.zeros_like(v) for k, v in params.items()
        }
        self.v: dict[str, mx.array] = {
            k: mx.zeros_like(v) for k, v in params.items()
        }

    def step(self, grads: dict[str, mx.array]) -> dict[str, mx.array]:
        self.t += 1
        b1, b2 = self.betas
        updates: dict[str, mx.array] = {}

        for name, param in self.params.items():
            if name not in grads:
                continue
            g = grads[name]
            self.m[name] = b1 * self.m[name] + (1 - b1) * g
            self.v[name] = b2 * self.v[name] + (1 - b2) * g ** 2
            m_hat = self.m[name] / (1 - b1 ** self.t)
            v_hat = self.v[name] / (1 - b2 ** self.t)
            updates[name] = param - self.lr * m_hat / (mx.sqrt(v_hat) + self.eps)

        return updates