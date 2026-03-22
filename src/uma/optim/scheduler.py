"""Learning rate schedulers.

All schedulers wrap an existing optimizer and adjust its ``lr`` attribute
each time ``step()`` is called (once per epoch, typically).

Example::

    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    for epoch in range(10):
        trainer.fit_epoch(loader)
        scheduler.step()
"""
from __future__ import annotations
import math
from .base import Optimizer


class StepLR:
    """Decays the learning rate by ``gamma`` every ``step_size`` epochs.

    Args:
        optimizer: Optimizer whose ``lr`` will be modified.
        step_size: Number of epochs between each decay step.
        gamma: Multiplicative decay factor. Default ``0.1``.

    Example::

        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        # lr halves every 5 epochs
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
    ) -> None:
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self._epoch = 0
        self._base_lr = optimizer.lr

    def step(self) -> None:
        """Advance one epoch and update the optimizer's learning rate."""
        self._epoch += 1
        if self._epoch % self.step_size == 0:
            self.optimizer.lr = self.optimizer.lr * self.gamma

    @property
    def last_lr(self) -> float:
        return self.optimizer.lr


class CosineAnnealingLR:
    """Cosine annealing schedule from ``lr`` down to ``eta_min`` over ``T_max`` epochs.

    After ``T_max`` epochs the schedule restarts from the initial lr.

    Args:
        optimizer: Optimizer whose ``lr`` will be modified.
        T_max: Number of epochs for one cosine half-cycle.
        eta_min: Minimum learning rate. Default ``0.0``.

    Example::

        scheduler = CosineAnnealingLR(optimizer, T_max=20)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self._epoch = 0
        self._base_lr = optimizer.lr

    def step(self) -> None:
        self._epoch += 1
        t = self._epoch % self.T_max
        cos_val = math.cos(math.pi * t / self.T_max)
        self.optimizer.lr = self.eta_min + 0.5 * (self._base_lr - self.eta_min) * (1 + cos_val)

    @property
    def last_lr(self) -> float:
        return self.optimizer.lr


class ReduceLROnPlateau:
    """Reduces the learning rate when a metric stops improving.

    Useful when you don't know how many epochs are needed — it adapts to
    the validation loss/metric automatically.

    Args:
        optimizer: Optimizer whose ``lr`` will be modified.
        mode: ``"min"`` (loss, lower is better) or ``"max"`` (accuracy).
        factor: Factor to reduce lr by. Default ``0.1``.
        patience: Epochs to wait before reducing. Default ``10``.
        min_lr: Lower bound on the learning rate. Default ``0.0``.

    Example::

        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5)
        for epoch in range(100):
            val_loss = ...
            scheduler.step(val_loss)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 0.0,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self._best: float = float("inf") if mode == "min" else float("-inf")
        self._wait = 0

    def step(self, metric: float) -> None:
        """Call once per epoch with the current validation metric."""
        improved = (
            metric < self._best if self.mode == "min" else metric > self._best
        )
        if improved:
            self._best = metric
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
                self.optimizer.lr = new_lr
                self._wait = 0

    @property
    def last_lr(self) -> float:
        return self.optimizer.lr
