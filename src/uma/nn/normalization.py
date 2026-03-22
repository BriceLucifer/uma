from __future__ import annotations
import mlx.core as mx
from .module import Module

class LayerNorm(Module):
    """Layer normalisation over the last dimension.

    Normalises each sample independently across features, then applies
    learnable scale (``weight`` / γ) and shift (``bias`` / β).

    Formula: ``output = weight * (x - mean) / sqrt(var + ε) + bias``

    Args:
        dims: Number of features (size of the last dimension).
        eps: Stability term added to the variance. Default ``1e-5``.

    Example:
        ```python
        norm = nn.LayerNorm(256)
        y = norm(x)  # x: (N, 256)
        ```
    """

    def __init__(self, dims: int, eps: float = 1e-5) -> None:
        self.weight: mx.array = mx.ones((dims,))   # gamma
        self.bias: mx.array = mx.zeros((dims,))    # beta
        self.eps = eps

    def forward(self, x: mx.array) -> mx.array:
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / mx.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias
'''
公式就是：
x_norm = (x - mean) / sqrt(var + eps)
output = gamma * x_norm + beta
'''