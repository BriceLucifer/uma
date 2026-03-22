from __future__ import annotations
from typing import Callable
import mlx.core as mx
from .nn.module import Module
from .optim.base import Optimizer
from .data.dataloader import DataLoader

LossFn = Callable[[Module, mx.array, mx.array], mx.array]
# Optional metric: (model, x, y) -> scalar float shown after each epoch
MetricFn = Callable[[Module, mx.array, mx.array], float]


def _clip_grad_norm(
    grads: dict[str, mx.array], max_norm: float
) -> dict[str, mx.array]:
    """Scale gradients so that their global L2 norm does not exceed max_norm."""
    total_sq = sum(float(mx.sum(g ** 2).item()) for g in grads.values())
    global_norm = total_sq ** 0.5
    if global_norm > max_norm:
        scale = max_norm / (global_norm + 1e-6)
        return {k: g * scale for k, g in grads.items()}
    return grads


class Trainer:
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: LossFn,
        clip_grad_norm: float | None = None,
    ) -> None:
        """
        Args:
            model:          The neural network module to train.
            optimizer:      Optimizer instance.
            loss_fn:        Callable ``(model, x, y) -> scalar loss``.
            clip_grad_norm: If set, clip gradient global L2 norm to this value
                            before each optimizer step. ``None`` disables clipping.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.clip_grad_norm = clip_grad_norm

    def train_step(self, x: mx.array, y: mx.array) -> float:
        current_params = self.model.parameters()

        def loss_fn(params: dict[str, mx.array]) -> mx.array:
            self.model.update(params)
            return self.loss_fn(self.model, x, y)

        loss, grads = mx.value_and_grad(loss_fn)(current_params)

        if self.clip_grad_norm is not None:
            grads = _clip_grad_norm(grads, self.clip_grad_norm)

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
        early_stopping_patience: int | None = None,
        early_stopping_mode: str = "min",
        scheduler=None,
    ) -> dict[str, list[float]]:
        """Train for `epochs` epochs.

        Args:
            loader:                   Training DataLoader.
            epochs:                   Number of epochs.
            val_data:                 Optional (x_val, y_val) arrays evaluated after each epoch.
            metric_fn:                Optional callable (model, x, y) -> float. Called on
                                      val_data if provided, else skipped.
            metric_name:              Label shown in the progress line (e.g. "acc").
            early_stopping_patience:  Stop training if the monitored metric/loss does
                                      not improve for this many epochs. ``None`` disables.
            early_stopping_mode:      ``"min"`` (lower is better, e.g. loss) or
                                      ``"max"`` (higher is better, e.g. accuracy).
                                      Determines what counts as "improvement".
            scheduler:                Optional LR scheduler with a ``.step()`` method.
                                      Called at the end of each epoch.
                                      ``ReduceLROnPlateau`` schedulers receive the
                                      monitored value; others receive no argument.

        Returns:
            history dict with keys "loss" and optionally metric_name.

        Failure modes:
            - val_data is evaluated in a single call — pass large datasets
              through uma.eval functions that batch internally.
            - metric_fn must call model.eval() itself (or use uma.eval helpers
              which do so). If it doesn't, Dropout remains active during eval.
            - Early stopping requires val_data + metric_fn when mode="max",
              otherwise it monitors training loss.
        """
        if early_stopping_patience is not None and early_stopping_mode not in ("min", "max"):
            raise ValueError(f"early_stopping_mode must be 'min' or 'max', got '{early_stopping_mode}'")

        history: dict[str, list[float]] = {"loss": []}
        if val_data is not None and metric_fn is not None:
            history[metric_name] = []

        # Early stopping state
        _best = float("inf") if early_stopping_mode == "min" else float("-inf")
        _wait = 0
        _best_params: dict[str, mx.array] | None = None

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            steps = 0

            for x, y in loader:
                epoch_loss += self.train_step(x, y)
                steps += 1

            avg = epoch_loss / steps if steps > 0 else 0.0
            history["loss"].append(avg)
            line = f"epoch {epoch + 1}/{epochs}  loss: {avg:.4f}"

            val_score: float | None = None
            if val_data is not None and metric_fn is not None:
                val_score = metric_fn(self.model, val_data[0], val_data[1])
                history[metric_name].append(val_score)
                line += f"  {metric_name}: {val_score:.4f}"
                self.model.train()  # restore training mode after eval

            # LR scheduler step
            if scheduler is not None:
                from .optim.scheduler import ReduceLROnPlateau
                monitored = val_score if val_score is not None else avg
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(monitored)
                else:
                    scheduler.step()
                line += f"  lr: {scheduler.last_lr:.2e}"

            print(line)

            # Early stopping
            if early_stopping_patience is not None:
                monitored = val_score if val_score is not None else avg
                improved = (
                    monitored < _best
                    if early_stopping_mode == "min"
                    else monitored > _best
                )
                if improved:
                    _best = monitored
                    _wait = 0
                    _best_params = {k: mx.array(v) for k, v in self.model.state_dict().items()}
                else:
                    _wait += 1
                    if _wait >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch + 1} (no improvement for {early_stopping_patience} epochs)")
                        if _best_params is not None:
                            self.model.update(_best_params)
                        break

        return history