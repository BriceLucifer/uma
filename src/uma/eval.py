from __future__ import annotations
import mlx.core as mx
from .nn.module import Module


def accuracy(model: Module, x: mx.array, y: mx.array, batch_size: int = 512) -> float:
    """Compute top-1 accuracy, evaluating in batches to avoid OOM.

    Failure modes:
        - Running the whole test set in one shot on large models will OOM
          on Apple Silicon unified memory. batch_size guards against this.
        - model.eval() must be called before inference — this function does
          it, but callers that bypass it (calling model.forward directly)
          will leave Dropout active and get noisy/wrong metrics.
        - y must be integer class indices (uint32/int32), not one-hot.
          Passing float labels silently produces nonsense comparisons.
    """
    model.eval()
    n = x.shape[0]
    correct = 0
    for start in range(0, n, batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        logits = model(xb)
        mx.eval(logits)
        preds = mx.argmax(logits, axis=-1)
        correct += int(mx.sum(preds == yb).item())
    return correct / n


def top_k_accuracy(
    model: Module, x: mx.array, y: mx.array, k: int = 5, batch_size: int = 512
) -> float:
    """Fraction of samples where the true label is in the top-k predictions."""
    model.eval()
    n = x.shape[0]
    correct = 0
    for start in range(0, n, batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        logits = model(xb)
        mx.eval(logits)
        # argsort descending: take last k indices of ascending sort
        sorted_idx = mx.argsort(logits, axis=-1)          # (B, C) ascending
        topk = sorted_idx[:, -k:]                          # (B, k)
        yb_col = yb.reshape(-1, 1)                         # (B, 1)
        correct += int(mx.sum(mx.any(topk == yb_col, axis=-1)).item())
    return correct / n


def confusion_matrix(
    model: Module, x: mx.array, y: mx.array, num_classes: int, batch_size: int = 512
) -> mx.array:
    """Return a (num_classes, num_classes) confusion matrix (int32).

    Entry [i, j] = number of samples with true label i predicted as j.

    Failure modes:
        - num_classes must match the model's output dimension exactly.
          A mismatch won't error here but will produce an under/over-sized matrix.
        - This materialises (N,) predictions in Python — fine for MNIST,
          but for large datasets consider streaming or numpy aggregation.
    """
    model.eval()
    n = x.shape[0]
    cm = [[0] * num_classes for _ in range(num_classes)]
    for start in range(0, n, batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        logits = model(xb)
        mx.eval(logits)
        preds = mx.argmax(logits, axis=-1)
        for true, pred in zip(yb.tolist(), preds.tolist()):
            cm[int(true)][int(pred)] += 1
    return mx.array(cm, dtype=mx.int32)
