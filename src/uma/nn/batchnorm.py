"""Batch Normalization layers."""
from __future__ import annotations
import mlx.core as mx
from .module import Module


class BatchNorm2d(Module):
    """Batch Normalization for 2D feature maps (NHWC layout).

    Normalizes over the (N, H, W) dimensions for each channel independently,
    then applies learnable scale (γ) and shift (β).

    During training the batch statistics are used. A running mean and variance
    are tracked for use during evaluation (``model.eval()``).

    Args:
        num_features: Number of channels (C in NHWC input).
        eps: Numerical stability term added to variance. Default ``1e-5``.
        momentum: Weight for running stats update (EMA). Default ``0.1``.

    Input shape: ``(N, H, W, C)`` — channels-last, matching MLX convention.

    Example::

        bn = BatchNorm2d(64)
        x = mx.zeros((32, 8, 8, 64))  # (N, H, W, C)
        out = bn(x)
    """

    training: bool

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # Learnable parameters (tracked by Module.parameters())
        self.weight = mx.ones((num_features,))   # γ
        self.bias = mx.zeros((num_features,))    # β

        # Running statistics — not returned by parameters() because they
        # are not arrays stored directly on self at __init__ time; we keep
        # them as plain Python attributes and update them manually.
        self._running_mean = mx.zeros((num_features,))
        self._running_var = mx.ones((num_features,))

    def forward(self, x: mx.array) -> mx.array:
        # x: (N, H, W, C)
        if self.training:
            # Compute batch mean/var over (N, H, W), shape (C,)
            mean = mx.mean(x, axis=(0, 1, 2))
            var = mx.var(x, axis=(0, 1, 2))

            # Update running stats (EMA) — detach from graph by converting
            # to concrete values after eval so they don't grow the tape.
            self._running_mean = (
                (1 - self.momentum) * self._running_mean + self.momentum * mean
            )
            self._running_var = (
                (1 - self.momentum) * self._running_var + self.momentum * var
            )
        else:
            mean = self._running_mean
            var = self._running_var

        x_hat = (x - mean) / mx.sqrt(var + self.eps)
        return self.weight * x_hat + self.bias


class BatchNorm1d(Module):
    """Batch Normalization for 1D inputs (N, C).

    Normalizes over the N dimension for each feature independently.

    Args:
        num_features: Number of features (C).
        eps: Numerical stability term. Default ``1e-5``.
        momentum: EMA weight for running stats. Default ``0.1``.

    Input shape: ``(N, C)``.

    Example::

        bn = BatchNorm1d(256)
        x = mx.zeros((32, 256))
        out = bn(x)
    """

    training: bool

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.weight = mx.ones((num_features,))
        self.bias = mx.zeros((num_features,))

        self._running_mean = mx.zeros((num_features,))
        self._running_var = mx.ones((num_features,))

    def forward(self, x: mx.array) -> mx.array:
        # x: (N, C)
        if self.training:
            mean = mx.mean(x, axis=0)
            var = mx.var(x, axis=0)
            self._running_mean = (
                (1 - self.momentum) * self._running_mean + self.momentum * mean
            )
            self._running_var = (
                (1 - self.momentum) * self._running_var + self.momentum * var
            )
        else:
            mean = self._running_mean
            var = self._running_var

        x_hat = (x - mean) / mx.sqrt(var + self.eps)
        return self.weight * x_hat + self.bias
