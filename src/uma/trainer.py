from __future__ import annotations
from typing import Callable
import mlx.core as mx
from .nn.module import Module
from .optim.base import Optimizer
from .data.dataloader import DataLoader

LossFn = Callable[[Module, mx.array, mx.array], mx.array]
# Optional metric: (model, x, y) -> scalar float shown after each epoch
MetricFn = Callable[[Module, mx.array, mx.array], float]


class Trainer:
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: LossFn,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, x: mx.array, y: mx.array) -> float:
        current_params = self.model.parameters()

        def loss_fn(params: dict[str, mx.array]) -> mx.array:
            self.model.update(params)
            return self.loss_fn(self.model, x, y)

        loss, grads = mx.value_and_grad(loss_fn)(current_params)

        # 每步都同步 optimizer 的 params
        self.optimizer.params = current_params
        updates = self.optimizer.step(grads)
        self.model.update(updates)
        mx.eval(loss, updates)
        return loss.item()

    def fit(
        self,
        loader: DataLoader,
        epochs: int = 10,
        val_data: tuple[mx.array, mx.array] | None = None,
        metric_fn: MetricFn | None = None,
        metric_name: str = "metric",
    ) -> dict[str, list[float]]:
        """Train for `epochs` epochs.

        Args:
            loader:      Training DataLoader.
            epochs:      Number of epochs.
            val_data:    Optional (x_val, y_val) arrays evaluated after each epoch.
            metric_fn:   Optional callable (model, x, y) -> float. Called on
                         val_data if provided, else skipped.
            metric_name: Label shown in the progress line (e.g. "acc").

        Returns:
            history dict with keys "loss" and optionally metric_name.

        Failure modes:
            - val_data is evaluated in a single call — pass large datasets
              through uma.eval functions that batch internally.
            - metric_fn must call model.eval() itself (or use uma.eval helpers
              which do so). If it doesn't, Dropout remains active during eval.
        """
        history: dict[str, list[float]] = {"loss": []}
        if val_data is not None and metric_fn is not None:
            history[metric_name] = []

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            steps = 0

            for x, y in loader:
                epoch_loss += self.train_step(x, y)
                steps += 1

            avg = epoch_loss / steps
            history["loss"].append(avg)
            line = f"epoch {epoch + 1}/{epochs}  loss: {avg:.4f}"

            if val_data is not None and metric_fn is not None:
                val_score = metric_fn(self.model, val_data[0], val_data[1])
                history[metric_name].append(val_score)
                line += f"  {metric_name}: {val_score:.4f}"
                self.model.train()  # restore training mode after eval

            print(line)

        return history