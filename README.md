# uma

A lightweight neural network framework built on [MLX](https://github.com/ml-explore/mlx) for Apple Silicon.

uma gives you PyTorch-style module composition, automatic differentiation, and training utilities — all running natively on the Apple GPU via MLX's unified memory model.

---

## Table of Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Core concepts](#core-concepts)
- [Modules — `uma.nn`](#modules--umann)
  - [Module base class](#module-base-class)
  - [Sequential & ModuleList](#sequential--modulelist)
  - [Linear](#linear)
  - [Conv2d](#conv2d)
  - [MaxPool2d](#maxpool2d)
  - [Activations](#activations)
  - [Dropout](#dropout)
  - [LayerNorm](#layernorm)
  - [BatchNorm](#batchnorm)
  - [Embedding](#embedding)
- [Optimizers — `uma.optim`](#optimizers--umaoptim)
  - [SGD](#sgd)
  - [Adam](#adam)
  - [AdamW](#adamw)
  - [LR Schedulers](#lr-schedulers)
- [Loss functions — `uma.functional`](#loss-functions--umafunctional)
- [Regularization](#regularization)
- [Data — `uma.data`](#data--umadata)
  - [Dataset](#dataset)
  - [DataLoader](#dataloader)
- [Trainer](#trainer)
  - [Gradient clipping](#gradient-clipping)
  - [Early stopping](#early-stopping)
- [Evaluation — `uma.eval`](#evaluation--umaeval)
- [Saving and loading weights](#saving-and-loading-weights)
- [Known limitations](#known-limitations)

---

## Installation

Requires Python ≥ 3.13 and an Apple Silicon Mac.

```bash
git clone https://github.com/you/uma
cd uma
uv sync          # or: pip install -e .
```

The only runtime dependency is `mlx >= 0.31.1`.

---

## Quick start

```python
import mlx.core as mx
import uma
import uma.nn as nn
import uma.optim as optim
from uma.data.dataset import Dataset
from uma.data.dataloader import DataLoader
from uma.trainer import Trainer

mx.set_default_device(mx.Device(mx.gpu))

# 1. define a model
class MLP(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# 2. dataset + loader
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

loader = DataLoader(MyDataset(train_x, train_y), batch_size=256, shuffle=True)

# 3. train
model = MLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = lambda model, x, y: uma.cross_entropy(model(x), y)

trainer = Trainer(model, optimizer, loss_fn)
history = trainer.fit(loader, epochs=10)

# 4. save
model.save("my_model")
```

See [`example/mnist.py`](example/mnist.py) for a complete CNN training example and [`example/test_saving_loading.py`](example/test_saving_loading.py) for a save/load round-trip test.

---

## Core concepts

### Lazy evaluation

uma runs on MLX, which uses **lazy evaluation**: operations build a compute graph and are not executed until `mx.eval()` is called. The `Trainer` calls `mx.eval` at the end of each step. If you run inference outside the trainer, call `mx.eval(output)` yourself before reading values.

### Channels-last

MLX uses **channels-last** layout for images: `(N, H, W, C)`.
This is the opposite of PyTorch's default `(N, C, H, W)`.
All uma conv and pooling layers expect channels-last input.

### Parameter tracking

uma tracks parameters by inspecting `self.__dict__` recursively. Every `mx.array` attribute on a `Module`, and every `mx.array` inside a child `Module` attribute, is a parameter. **Plain Python lists and dicts of modules are not traversed** — use `nn.ModuleList` or direct attributes instead:

```python
# correct — direct attribute
self.conv1 = nn.Conv2d(...)
self.conv2 = nn.Conv2d(...)

# correct — ModuleList registers sub-modules properly
self.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(4)])

# wrong — parameters inside a plain list are NOT tracked
self.layers = [nn.Conv2d(...), nn.Conv2d(...)]
```

---

## Modules — `uma.nn`

### Sequential & ModuleList

**`nn.Sequential`** chains modules in order — the output of each layer is the input to the next:

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
out = model(x)   # calls each layer in sequence
```

**`nn.ModuleList`** stores a list of modules and registers their parameters for tracking. Use it when you need to iterate manually (e.g. residual connections):

```python
class ResBlock(nn.Module):
    def __init__(self, dim, n_layers):
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.act = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers)):
            x = x + self.act(self.layers[i](x))   # residual
        return x
```

---

### Module base class

All layers inherit from `nn.Module`.

```python
class MyLayer(nn.Module):
    def __init__(self):
        self.weight = mx.ones((4, 4))  # automatically tracked

    def forward(self, x: mx.array) -> mx.array:
        return x @ self.weight
```

**Key methods:**

| Method | Description |
|--------|-------------|
| `forward(x)` | Define the computation. Must be overridden. |
| `__call__(*args)` | Calls `forward`. Use this at call sites. |
| `parameters()` | Returns `{dotted.name: mx.array}` flat dict of all parameters. |
| `update(dict)` | Applies a flat dict of parameter updates in-place. |
| `train()` | Sets all submodules to training mode (activates Dropout). |
| `eval()` | Sets all submodules to eval mode (disables Dropout). |
| `named_modules()` | Yields `(name, module)` for self and all descendants. |
| `state_dict()` | Returns a copy of all parameters (same as `parameters()`). |
| `load_state_dict(sd, strict=True)` | Load a flat parameter dict. |
| `save(path)` | Save parameters to `path.npz`. |
| `load(path)` | Load parameters from a `.npz` file. |

---

### Linear

Fully-connected layer: `y = x @ W + b`.

```python
nn.Linear(in_features, out_features, bias=True)
```

- Weights initialised with Kaiming uniform: `scale = 1 / sqrt(in_features)`
- Weight shape: `(in_features, out_features)` — transposed relative to PyTorch

```python
fc = nn.Linear(128, 64)
y = fc(x)   # x: (N, 128) → y: (N, 64)
```

---

### Conv2d

2D convolution over a channels-last feature map.

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
```

- Weight shape: `(out_channels, kernel_size, kernel_size, in_channels)`
- Input/output shape: `(N, H, W, C)` — **channels last**
- Initialised with Kaiming uniform: `scale = 1 / sqrt(in_channels * kH * kW)`

```python
conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
# (N, 28, 28, 1) → (N, 28, 28, 32)
y = conv(x)
```

> **Coming from PyTorch?** Reshape your input from `(N, C, H, W)` to `(N, H, W, C)` before passing to `Conv2d`.

---

### MaxPool2d

Non-overlapping max pooling (stride equals kernel size).

```python
nn.MaxPool2d(kernel_size=2)
```

```python
pool = nn.MaxPool2d(2)
# (N, 28, 28, 32) → (N, 14, 14, 32)
y = pool(x)
```

> Spatial dimensions must be evenly divisible by `kernel_size`. A `ValueError` is raised at runtime otherwise.

---

### Activations

| Class | Formula | Notes |
|-------|---------|-------|
| `nn.ReLU()` | `max(x, 0)` | |
| `nn.GELU()` | `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))` | |
| `nn.Softmax(axis=-1)` | `exp(xᵢ) / Σexp(xⱼ)` along `axis` | |
| `nn.Sigmoid()` | `1 / (1 + exp(-x))` | Binary classification output |
| `nn.Tanh()` | `tanh(x)` | Output range (-1, 1) |
| `nn.LeakyReLU(negative_slope=0.01)` | `x if x>0 else slope·x` | Avoids dying ReLU |

```python
act = nn.LeakyReLU(negative_slope=0.2)
y = act(x)
```

---

### Dropout

Stochastic regularization. Active only when the module is in training mode.

```python
nn.Dropout(p=0.5)
```

- **Training:** randomly zeroes elements with probability `p`, then scales by `1/(1-p)` (inverted dropout).
- **Eval:** identity pass-through.

```python
drop = nn.Dropout(p=0.2)
model.train()   # dropout active
model.eval()    # dropout off
```

---

### LayerNorm

Normalises over the last dimension with learnable scale and shift.

```python
nn.LayerNorm(dims, eps=1e-5)
```

Formula: `output = weight * (x - mean) / sqrt(var + ε) + bias`

- `weight` (γ) initialised to ones, `bias` (β) initialised to zeros.

```python
norm = nn.LayerNorm(256)
y = norm(x)   # x: (N, 256)
```

---

### BatchNorm

**`nn.BatchNorm2d`** normalises over the spatial dimensions (N, H, W) for each channel in a feature map. Use after `Conv2d` layers.

```python
nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1)
```

Input shape: `(N, H, W, C)` — channels-last, matching MLX's convention.

```python
self.bn = nn.BatchNorm2d(32)
# in forward:
x = self.bn(x)   # shape (N, H, W, 32)
```

**`nn.BatchNorm1d`** normalises over the batch dimension for 1D features:

```python
nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.1)
# input (N, C) → output (N, C)
```

Both layers switch automatically between batch statistics (training) and running statistics (eval) when you call `model.train()` / `model.eval()`.

---

### Embedding

Lookup table for integer token indices.

```python
nn.Embedding(num_embeddings, embedding_dim)
```

```python
embed = nn.Embedding(10000, 128)
tokens = mx.array([[1, 42, 7]])    # (1, 3) — integer indices
out = embed(tokens)                # (1, 3, 128)
```

Useful for word embeddings, positional encodings, or any categorical input.

---

## Optimizers — `uma.optim`

All optimizers are constructed with the model's parameter dict and a learning rate, then used via `trainer.fit` or manually:

```python
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# manual usage:
updates = optimizer.step(grads)
model.update(updates)
```

### SGD

```python
optim.SGD(params, lr=0.01)
```

Update rule: `θ ← θ - lr · ∇θ`

### Adam

```python
optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
```

Update rule:
```
m ← β₁·m + (1-β₁)·g
v ← β₂·v + (1-β₂)·g²
m̂ = m / (1 - β₁ᵗ)
v̂ = v / (1 - β₂ᵗ)
θ ← θ - lr · m̂ / (√v̂ + ε)
```

### AdamW

Adam with **decoupled weight decay** — weight decay is applied directly to parameters rather than through the gradient, giving better regularization with adaptive methods.

```python
optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
```

Update rule (extra term vs Adam):
```
θ ← θ - lr · m̂/(√v̂ + ε) - lr · weight_decay · θ
```

---

### LR Schedulers

Schedulers wrap an optimizer and call `.step()` once per epoch to adjust the learning rate.

**`StepLR`** — multiply lr by `gamma` every `step_size` epochs:

```python
scheduler = optim.StepLR(optimizer, step_size=5, gamma=0.5)
```

**`CosineAnnealingLR`** — decay lr from initial down to `eta_min` following a cosine curve over `T_max` epochs:

```python
scheduler = optim.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
```

**`ReduceLROnPlateau`** — reduce lr when a metric stops improving (no need to know the epoch count in advance):

```python
scheduler = optim.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
```

Pass the scheduler to `trainer.fit`:

```python
history = trainer.fit(
    loader, epochs=50,
    val_data=(val_x, val_y),
    metric_fn=uma.eval.accuracy,
    metric_name="acc",
    scheduler=scheduler,
)
```

`ReduceLROnPlateau.step(metric)` is called automatically with the validation metric; other schedulers call `.step()` with no argument.

---

## Loss functions — `uma.functional`

```python
import uma

# multi-class classification — targets are integer class indices, not one-hot
loss = uma.cross_entropy(logits, targets)
loss = uma.cross_entropy(logits, targets, reduction="sum")

# binary classification — from raw logits (sigmoid applied internally)
loss = uma.binary_cross_entropy(logits, targets)   # targets: 0 or 1

# regression
loss = uma.mse(predictions, targets)

# Huber loss — robust to outliers (MSE for small errors, MAE for large)
loss = uma.huber(predictions, targets, delta=1.0)
```

---

## Regularization

```python
params = model.parameters()

loss = uma.cross_entropy(model(x), y) \
     + uma.l2_regularization(params, lam=1e-4) \
     + uma.l1_regularization(params, lam=1e-5)
```

Both `l1_regularization` and `l2_regularization` take the parameter dict and a scalar `lam` (lambda weight).

---

## Data — `uma.data`

### Dataset

Abstract base class — implement `__len__` and `__getitem__`.

```python
from uma.data.dataset import Dataset

class MyDataset(Dataset):
    def __init__(self, x: mx.array, y: mx.array):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
```

### DataLoader

```python
from uma.data.dataloader import DataLoader

loader = DataLoader(dataset, batch_size=256, shuffle=True)

for x_batch, y_batch in loader:
    # x_batch: (batch_size, ...), y_batch: (batch_size, ...)
    ...
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | `32` | Samples per batch |
| `shuffle` | `True` | Shuffle indices at the start of each iteration |

The last batch may be smaller than `batch_size`.

---

## Trainer

```python
from uma.trainer import Trainer

def loss_fn(model, x, y):
    return uma.cross_entropy(model(x), y)

trainer = Trainer(model, optimizer, loss_fn)

history = trainer.fit(
    loader,
    epochs=10,
    val_data=(test_x, test_y),      # optional
    metric_fn=uma.eval.accuracy,    # any (model, x, y) -> float callable
    metric_name="acc",
)
# → {"loss": [0.45, 0.32, ...], "acc": [0.87, 0.91, ...]}
```

Progress output per epoch:
```
epoch 1/10  loss: 0.4521  acc: 0.8734
epoch 2/10  loss: 0.3198  acc: 0.9102
...
```

**`fit` returns** a dict with `"loss"` and the `metric_name` key (if `val_data` was provided).

### Gradient clipping

Pass `clip_grad_norm` to `Trainer` to clip the global L2 norm of gradients before each optimizer step. Useful when training RNNs or deep networks prone to exploding gradients.

```python
trainer = Trainer(model, optimizer, loss_fn, clip_grad_norm=1.0)
```

### Early stopping

Stop training automatically when the monitored metric stops improving, and restore the best weights seen so far.

```python
history = trainer.fit(
    loader,
    epochs=100,
    val_data=(val_x, val_y),
    metric_fn=uma.eval.accuracy,
    metric_name="acc",
    early_stopping_patience=10,      # stop after 10 epochs without improvement
    early_stopping_mode="max",       # "max" for accuracy, "min" for loss
)
```

When `val_data` + `metric_fn` are provided, early stopping monitors the validation metric; otherwise it monitors training loss.

---

## Evaluation — `uma.eval`

All functions call `model.eval()` automatically before running inference and process data in batches to avoid OOM errors.

```python
import uma.eval as ueval

# top-1 accuracy
acc = ueval.accuracy(model, test_x, test_y)
# → 0.9912

# top-k accuracy
top5 = ueval.top_k_accuracy(model, test_x, test_y, k=5)
# → 0.9987

# confusion matrix: entry [i,j] = samples with true label i predicted as j
cm = ueval.confusion_matrix(model, test_x, test_y, num_classes=10)
# → mx.array shape (10, 10), dtype int32

# precision, recall, F1
p, r, f1 = ueval.precision_recall_f1(model, test_x, test_y, num_classes=10)
p, r, f1 = ueval.precision_recall_f1(model, test_x, test_y, num_classes=10, average="micro")
```

Optional `batch_size` parameter on all functions (default `512`).

> `y` must be integer class indices (`uint32` or `int32`), not one-hot vectors.

---

## Saving and loading weights

Weights are stored in `.npz` format — a plain numpy archive with no framework dependency.

```python
# save
model.save("checkpoint")      # writes checkpoint.npz

# load
model2 = MyModel()
model2.load("checkpoint.npz")
```

**`state_dict` / `load_state_dict`** for manual control:

```python
sd = model.state_dict()                       # dict[str, mx.array]
model2.load_state_dict(sd)                    # strict: all keys must match
model2.load_state_dict(sd, strict=False)      # partial load, missing keys skipped
```

Keys use dot notation (`conv1.weight`, `fc1.bias`). Inside the file, dots are stored as `/` and converted back on load.

---

## Known limitations

| Area | Limitation |
|------|------------|
| Module containers | Plain Python `list`/`dict` of modules are not tracked. Use `nn.Sequential` or `nn.ModuleList` instead. |
| `MaxPool2d` | Stride equals kernel size only. Spatial dims must be divisible by `kernel_size`. |
| Lazy shape errors | MLX validates shapes at eval time, not construction. A wrong-shape `load_state_dict` silently succeeds and crashes on the first forward pass. |
| BatchNorm running stats | `_running_mean` / `_running_var` in `BatchNorm1d/2d` are not saved by `model.save()` — they are plain Python attributes, not tracked `mx.array`s. For deployment, convert to `LayerNorm` or fold BN into the preceding linear layer. |
| `value_and_grad` double-runs forward | The gradient closure re-executes `forward` during backprop. Avoid side effects inside `forward`. |
| Platform | MLX is Apple Silicon only. uma will not run on CUDA or x86. |
