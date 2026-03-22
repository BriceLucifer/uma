"""Stateless loss and regularization functions.

All functions are pure — they take arrays and return arrays with no side
effects, making them safe to use inside ``mx.value_and_grad`` closures.
"""
from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn
import functools

# loss functions
def cross_entropy(
    logits: mx.array,
    targets: mx.array,
    reduction: str = "mean",
) -> mx.array:
    """Softmax cross-entropy loss for multi-class classification.

    Args:
        logits: Raw (unnormalised) model outputs, shape ``(N, C)``.
        targets: Integer class indices, shape ``(N,)``.
        reduction: ``"mean"`` (default) or ``"sum"``.

    Returns:
        Scalar loss value.
    """
    loss = nn.losses.cross_entropy(logits=logits, targets=targets)
    return mx.mean(loss) if reduction == "mean" else mx.sum(loss)

def mse(
    predictions: mx.array,
    targets: mx.array,
    reduction: str = "mean",
) -> mx.array:
    """Mean squared error loss for regression.

    Args:
        predictions: Model outputs, any shape.
        targets: Ground-truth values, same shape as ``predictions``.
        reduction: ``"mean"`` (default) or ``"sum"``.

    Returns:
        Scalar loss value.
    """
    loss = (predictions - targets) ** 2
    return mx.mean(loss) if reduction == "mean" else mx.sum(loss)

def binary_cross_entropy(
    logits: mx.array,
    targets: mx.array,
    reduction: str = "mean",
) -> mx.array:
    """Binary cross-entropy loss from raw logits (sigmoid applied internally).

    Uses the numerically stable form: ``max(x,0) - x*y + log(1+exp(-|x|))``.

    Args:
        logits: Raw model outputs, shape ``(N,)`` or ``(N, 1)``.
        targets: Binary labels (0 or 1), same shape as ``logits``.
        reduction: ``"mean"`` (default) or ``"sum"``.

    Returns:
        Scalar loss value.
    """
    loss = mx.maximum(logits, 0) - logits * targets + mx.log(1 + mx.exp(-mx.abs(logits)))
    return mx.mean(loss) if reduction == "mean" else mx.sum(loss)


def huber(
    predictions: mx.array,
    targets: mx.array,
    delta: float = 1.0,
    reduction: str = "mean",
) -> mx.array:
    """Huber (smooth L1) loss — less sensitive to outliers than MSE.

    For ``|error| <= delta`` behaves like MSE; for larger errors like MAE.

    Args:
        predictions: Model outputs, any shape.
        targets: Ground-truth values, same shape as ``predictions``.
        delta: Threshold between quadratic and linear regions. Default ``1.0``.
        reduction: ``"mean"`` (default) or ``"sum"``.

    Returns:
        Scalar loss value.
    """
    err = mx.abs(predictions - targets)
    loss = mx.where(err <= delta, 0.5 * err ** 2, delta * (err - 0.5 * delta))
    return mx.mean(loss) if reduction == "mean" else mx.sum(loss)


# regularization
def l2_regularization(
    params: dict[str, mx.array],
    lam: float,
) -> mx.array:
    """L2 (weight decay) regularization penalty.

    Computes ``lam * Σ ||θ||²`` over all parameters.

    Args:
        params: Parameter dict from ``model.parameters()``.
        lam: Regularization strength.

    Returns:
        Scalar penalty to add to the task loss.
    """
    terms = [mx.sum(p ** 2) for p in params.values()]
    return lam * functools.reduce(mx.add, terms)

def l1_regularization(
    params: dict[str, mx.array],
    lam: float,
) -> mx.array:
    """L1 (sparsity) regularization penalty.

    Computes ``lam * Σ ||θ||₁`` over all parameters.

    Args:
        params: Parameter dict from ``model.parameters()``.
        lam: Regularization strength.

    Returns:
        Scalar penalty to add to the task loss.
    """
    terms = [mx.sum(mx.abs(p)) for p in params.values()]
    return lam * functools.reduce(mx.add, terms)