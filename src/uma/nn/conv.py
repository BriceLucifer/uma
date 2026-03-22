from __future__ import annotations
import math
import mlx.core as mx
from .module import Module


class Conv2d(Module):
    """2D convolution.

    Follows MLX's channels-last convention: input must be (N, H, W, C).
    This is the opposite of PyTorch's (N, C, H, W) — the most likely
    source of shape errors when porting models.

    Weight shape: (out_channels, kernel_size, kernel_size, in_channels)

    Failure modes:
        - Passing (N, C, H, W) input will silently produce wrong results
          because mlx.conv2d will treat the channel dim as spatial.
        - If in_channels doesn't match x.shape[-1] at runtime, MLX raises
          a shape error deep inside the kernel.
        - bias=False stores None, which parameters() skips — that's correct,
          but callers enumerating params should not assume bias always exists.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        self.stride = stride
        self.padding = padding
        # Kaiming uniform: fan_in = in_channels * kH * kW
        fan_in = in_channels * kernel_size * kernel_size
        scale = 1.0 / math.sqrt(fan_in)
        # MLX weight layout: (C_out, kH, kW, C_in)
        self.weight: mx.array = mx.random.uniform(
            low=-scale, high=scale,
            shape=(out_channels, kernel_size, kernel_size, in_channels),
        )
        self.bias: mx.array | None = (
            mx.zeros((out_channels,)) if bias else None
        )

    def forward(self, x: mx.array) -> mx.array:
        # x must be (N, H, W, C) — channels last
        out = mx.conv2d(x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            out = out + self.bias
        return out


class MaxPool2d(Module):
    """Non-overlapping max pooling (stride == kernel_size).

    Failure modes:
        - Only works when H and W are both divisible by kernel_size.
          Odd spatial dims (e.g. 27x27 with pool=2) will raise a reshape
          error at runtime — there is no automatic padding.
        - stride != kernel_size is not supported; behaviour is undefined.
        - Input must be (N, H, W, C) channels-last, same as Conv2d.
    """

    def __init__(self, kernel_size: int = 2) -> None:
        self.kernel_size = kernel_size

    def forward(self, x: mx.array) -> mx.array:
        k = self.kernel_size
        n, h, w, c = x.shape
        if h % k != 0 or w % k != 0:
            raise ValueError(
                f"MaxPool2d({k}): spatial dims ({h}, {w}) must be divisible by {k}."
            )
        # reshape trick: (N, H/k, k, W/k, k, C) → max over the k-windows
        x = x.reshape(n, h // k, k, w // k, k, c)
        return x.max(axis=(2, 4))
