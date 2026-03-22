# Training

## The Trainer

`Trainer` wraps the train loop so you only need to supply a model, an optimizer, and a loss function.

```python
from uma.trainer import Trainer
import uma

def loss_fn(model, x, y):
    return uma.cross_entropy(model(x), y)

trainer = Trainer(model, optimizer, loss_fn)

history = trainer.fit(
    loader,
    epochs=20,
    val_data=(test_x, test_y),      # optional
    metric_fn=uma.eval.accuracy,    # optional — any (model, x, y) -> float
    metric_name="acc",
)
# history = {"loss": [...], "acc": [...]}
```

Each epoch prints:
```
epoch 3/20  loss: 0.0821  acc: 0.9812
```

---

## What happens inside `train_step`

```
params = model.parameters()
loss, grads = mx.value_and_grad(loss_closure)(params)
updates = optimizer.step(grads)
model.update(updates)
mx.eval(loss, updates)
```

MLX's `value_and_grad` re-executes `forward` as part of the backward pass. Avoid side effects (printing, random seeds, mutable state) inside `forward`.

---

## Optimizers

### SGD

```python
import uma.optim as optim
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

Update rule: `θ ← θ − lr · ∇θ`

### Adam

```python
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
```

Update rule:
```
m ← β₁·m + (1−β₁)·g
v ← β₂·v + (1−β₂)·g²
θ ← θ − lr · (m/(1−β₁ᵗ)) / (√(v/(1−β₂ᵗ)) + ε)
```

### AdamW

Adam with decoupled weight decay — applies weight decay directly to parameters rather than through gradients.

```python
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

---

## LR Schedulers

Wrap an optimizer and call `.step()` once per epoch. Pass to `trainer.fit` via the `scheduler=` argument.

```python
# Decay lr by gamma every step_size epochs
scheduler = optim.StepLR(optimizer, step_size=5, gamma=0.5)

# Cosine annealing from lr down to eta_min over T_max epochs
scheduler = optim.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

# Reduce lr when a metric stops improving
scheduler = optim.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
```

`ReduceLROnPlateau.step(metric)` is called automatically with the monitored value; other schedulers call `.step()` with no argument.

---

## Gradient clipping

Pass `clip_grad_norm` to `Trainer` to clip the global L2 norm of all gradients before each optimizer step:

```python
trainer = Trainer(model, optimizer, loss_fn, clip_grad_norm=1.0)
```

---

## Early stopping

Stop training when the monitored metric stops improving and restore the best weights seen so far:

```python
history = trainer.fit(
    loader, epochs=100,
    val_data=(val_x, val_y),
    metric_fn=uma.eval.accuracy,
    metric_name="acc",
    early_stopping_patience=10,   # stop after 10 epochs without improvement
    early_stopping_mode="max",    # "max" for accuracy, "min" for loss
)
```

- When `val_data` + `metric_fn` are provided, the validation metric is monitored; otherwise training loss is used.
- `early_stopping_mode` must be `"min"` or `"max"` — any other value raises `ValueError`.
- Best weights (parameters **and** buffers) are restored automatically on early stop.

---

## Loss functions

```python
import uma

# Classification (integer targets, not one-hot)
loss = uma.cross_entropy(logits, targets)
loss = uma.cross_entropy(logits, targets, reduction="sum")

# Binary classification (from raw logits)
loss = uma.binary_cross_entropy(logits, targets)   # targets: 0.0 or 1.0

# Regression
loss = uma.mse(predictions, targets)

# Huber — robust to outliers
loss = uma.huber(predictions, targets, delta=1.0)
```

All loss functions accept `reduction="mean"` (default) or `reduction="sum"`. Any other value raises `ValueError`.

---

## Regularization

Combine with any loss:

```python
def loss_fn(model, x, y):
    params = model.parameters()
    return (
        uma.cross_entropy(model(x), y)
        + uma.l2_regularization(params, lam=1e-4)
    )
```

| Function | Penalty |
|----------|---------|
| `uma.l2_regularization(params, lam)` | `lam * Σ ‖θ‖²` |
| `uma.l1_regularization(params, lam)` | `lam * Σ ‖θ‖₁` |

---

## Manual training loop

If you need more control than `Trainer.fit` provides:

```python
model.train()
for x, y in loader:
    params = model.parameters()

    def loss_closure(params):
        model.update(params)
        return uma.cross_entropy(model(x), y)

    loss, grads = mx.value_and_grad(loss_closure)(params)
    optimizer.params = params
    updates = optimizer.step(grads)
    model.update(updates)
    mx.eval(loss, updates)
```
