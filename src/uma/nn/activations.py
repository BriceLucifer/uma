import mlx.core as mx
from .module import Module


class ReLU(Module):
    """Rectified Linear Unit: ``max(x, 0)``."""

    def forward(self, x: mx.array) -> mx.array:
        return mx.maximum(x, 0)


class GELU(Module):
    """Gaussian Error Linear Unit (tanh approximation).

    Formula: ``0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))``
    """

    def forward(self, x: mx.array) -> mx.array:
        return mx.array(0.5) * x * (1 + mx.tanh(
            mx.array(0.7978845608) * (x + mx.array(0.044715) * x ** 3)
        ))


class Softmax(Module):
    """Softmax normalisation along a given axis.

    Args:
        axis: Dimension to normalise over. Default ``-1`` (last dim).
    """

    def __init__(self, axis: int = -1) -> None:
        self.axis = axis

    def forward(self, x: mx.array) -> mx.array:
        return mx.softmax(x, axis=self.axis)