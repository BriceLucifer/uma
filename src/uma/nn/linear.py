import mlx.core as mx
from .module import Module


class Linear(Module):
    """Fully-connected linear layer: ``y = x @ W + b``.

    Weight shape is ``(in_features, out_features)`` — note this is the
    transpose of PyTorch's convention.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If ``False``, no bias term is added. Default ``True``.

    Example:
        ```python
        fc = nn.Linear(128, 64)
        y = fc(x)  # x: (N, 128) → y: (N, 64)
        ```
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        scale = 1.0 / (in_features ** 0.5)  # Kaiming uniform
        self.weight: mx.array = mx.random.uniform(
            low=-scale, high=scale,
            shape=(in_features, out_features),
        )
        self.bias: mx.array | None = (
            mx.zeros((out_features,)) if bias else None
        )

    def forward(self, x: mx.array) -> mx.array:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out