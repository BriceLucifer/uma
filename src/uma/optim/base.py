from abc import ABC, abstractmethod
import mlx.core as mx


class Optimizer(ABC):
    """Abstract base class for all optimizers.

    Args:
        params: Initial parameter dict from ``model.parameters()``.
        lr: Learning rate.
    """

    def __init__(self, params: dict[str, mx.array], lr: float) -> None:
        self.params = params
        self.lr = lr

    @abstractmethod
    def step(self, grads: dict[str, mx.array]) -> dict[str, mx.array]:
        """Compute and return updated parameters.

        Args:
            grads: Gradient dict with the same keys as ``self.params``.

        Returns:
            Dict of updated parameter arrays.
        """
        ...