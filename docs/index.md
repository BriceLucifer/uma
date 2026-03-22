# uma

**A lightweight neural network framework for Apple Silicon.**

uma is built on top of [MLX](https://github.com/ml-explore/mlx) and gives you PyTorch-style module composition, automatic differentiation, and training utilities — all running natively on the Apple GPU through MLX's unified memory model.

---

## Why uma?

| Feature | Detail |
|---------|--------|
| **Apple Silicon native** | Built on MLX — runs on the M-series GPU with zero device management overhead. |
| **PyTorch-like API** | Familiar `Module`, `forward`, `parameters()`, `train()`/`eval()` patterns. |
| **No heavy dependencies** | Only requires `mlx`. No numpy, no torch at runtime. |
| **Portable weights** | Save and load checkpoints (parameters + BatchNorm running stats) in `.npz` format. |
| **Built-in eval metrics** | Accuracy, top-k, confusion matrix, precision/recall/F1 out of the box. |

---

## Install

```bash
git clone https://github.com/BriceLucifer/uma
cd uma
uv sync
```

Requires **Python ≥ 3.13** and an **Apple Silicon Mac**.

---

## 30-second example

```python
import mlx.core as mx
import uma
import uma.nn as nn
import uma.optim as optim
from uma.data.dataset import Dataset
from uma.data.dataloader import DataLoader
from uma.trainer import Trainer

mx.set_default_device(mx.Device(mx.gpu))

class MLP(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = MLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
trainer = Trainer(model, optimizer, lambda m, x, y: uma.cross_entropy(m(x), y))
trainer.fit(loader, epochs=10)

model.save("checkpoint")
```

Continue to the [Quick Start](quickstart.md) for a full walkthrough.

---

## Examples

| File | Description |
|------|-------------|
| [`example/mnist.py`](https://github.com/BriceLucifer/uma/blob/main/example/mnist.py) | Full CNN training on MNIST |
| [`example/new_features_demo.py`](https://github.com/BriceLucifer/uma/blob/main/example/new_features_demo.py) | Demonstrates all layers, optimizers, schedulers, and losses |
| [`example/test_saving_loading.py`](https://github.com/BriceLucifer/uma/blob/main/example/test_saving_loading.py) | Save/load round-trip verification |
| [`example/test_uma.py`](https://github.com/BriceLucifer/uma/blob/main/example/test_uma.py) | Unit tests (67 tests, run with `uv run python example/test_uma.py`) |
