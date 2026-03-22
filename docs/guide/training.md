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

---

## Loss functions

```python
import uma

# Classification (integer targets, not one-hot)
loss = uma.cross_entropy(logits, targets)
loss = uma.cross_entropy(logits, targets, reduction="sum")

# Regression
loss = uma.mse(predictions, targets)
```

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
