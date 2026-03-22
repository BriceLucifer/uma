# Building Models

All neural network components in uma inherit from `nn.Module`. This page explains how to build models, compose layers, and understand how parameters are tracked.

---

## Subclassing Module

```python
import mlx.core as mx
import uma.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x: mx.array) -> mx.array:
        return self.fc2(self.relu(self.fc1(x)))

model = MyModel()
output = model(x)   # calls forward()
```

---

## Parameter tracking

uma discovers parameters by recursively inspecting `self.__dict__`. Any `mx.array` attribute — and any `mx.array` inside a child `Module` — is automatically tracked.

```python
params = model.parameters()
# {"fc1.weight": mx.array(...), "fc1.bias": mx.array(...), ...}
```

!!! warning "Lists of modules are not tracked"
    All sub-layers must be **direct attributes**:

    ```python
    # correct
    self.conv1 = nn.Conv2d(...)
    self.conv2 = nn.Conv2d(...)

    # wrong — params inside the list are invisible
    self.layers = [nn.Conv2d(...), nn.Conv2d(...)]
    ```

---

## Training and eval mode

`model.train()` and `model.eval()` recursively toggle the `training` flag on all submodules that have one (`Dropout` and `BatchNorm1d`/`BatchNorm2d`).

```python
model.train()   # Dropout active
model.eval()    # Dropout disabled — always call this before inference
```

!!! tip
    `uma.eval.*` metric functions call `model.eval()` for you. If you run inference manually, remember to call it yourself.

---

## Available layers

### Linear

Fully-connected layer. Weight shape: `(in_features, out_features)`.

```python
nn.Linear(in_features=128, out_features=64, bias=True)
```

### Conv2d

2D convolution. Input must be **channels-last**: `(N, H, W, C)`.

```python
nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
```

### MaxPool2d

Non-overlapping max pooling. Spatial dims must be divisible by `kernel_size`.

```python
nn.MaxPool2d(kernel_size=2)
# (N, 28, 28, C) → (N, 14, 14, C)
```

### Activations

```python
nn.ReLU()                          # max(x, 0)
nn.GELU()                          # Gaussian Error Linear Unit
nn.Softmax(axis=-1)                # softmax along last dim
nn.Sigmoid()                       # 1 / (1 + exp(-x)), binary output
nn.Tanh()                          # tanh(x), range (-1, 1)
nn.LeakyReLU(negative_slope=0.01)  # slope * x for x < 0
```

### Dropout

```python
nn.Dropout(p=0.3)   # drop 30% of activations during training; p must be in [0, 1)
```

### LayerNorm

```python
nn.LayerNorm(dims=256)   # normalise over last dimension
```

### BatchNorm1d / BatchNorm2d

Normalise over the batch dimension with learnable scale and shift. Tracks running statistics for use at eval time.

```python
nn.BatchNorm1d(num_features=256)          # input (N, C)
nn.BatchNorm2d(num_features=32)           # input (N, H, W, C) — channels-last
```

Running stats (`_running_mean`, `_running_var`) are **persistent buffers**: saved/loaded by `save()`/`load()` but excluded from gradient computation and optimizer updates.

### Embedding

Lookup table mapping integer token indices to dense vectors.

```python
nn.Embedding(num_embeddings=10000, embedding_dim=128)
tokens = mx.array([[1, 42, 7]])   # (1, 3) int
out = embed(tokens)               # (1, 3, 128)
```

### Sequential

Chains modules in order — output of each layer feeds the next.

```python
model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 10),
)
out = model(x)
```

### ModuleList

Stores a list of modules and registers their parameters. Use when you need manual iteration (e.g. residual connections).

```python
self.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(4)])
# in forward:
for i in range(len(self.layers)):
    x = x + self.layers[i](x)
```

---

## Channels-last convention

MLX uses `(N, H, W, C)` — the opposite of PyTorch's `(N, C, H, W)`. Reshape inside `forward` to keep the `Dataset` and `DataLoader` unaware of the layout:

```python
def forward(self, x):
    n = x.shape[0]
    x = x.reshape(n, 28, 28, 1)  # flat → channels-last
    ...
```
