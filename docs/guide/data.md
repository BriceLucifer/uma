# Data Pipeline

uma's data utilities mirror PyTorch's `Dataset` / `DataLoader` pattern.

---

## Dataset

Subclass `Dataset` and implement two methods:

```python
from uma.data.dataset import Dataset
import mlx.core as mx

class MyDataset(Dataset):
    def __init__(self, x: mx.array, y: mx.array):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
```

`__getitem__` should return a `(input, label)` tuple. Both values are stacked by `DataLoader` using `mx.stack`.

---

## DataLoader

```python
from uma.data.dataloader import DataLoader

loader = DataLoader(dataset, batch_size=256, shuffle=True)

for x_batch, y_batch in loader:
    # x_batch: (batch_size, ...)
    # y_batch: (batch_size, ...)
    ...
```

| Argument | Default | Description |
|----------|---------|-------------|
| `batch_size` | `32` | Samples per batch |
| `shuffle` | `True` | Re-shuffle at the start of every iteration |

The last batch may be smaller than `batch_size` if the dataset length is not evenly divisible.

---

## Example — loading MNIST from raw bytes

```python
import urllib.request, gzip, os
import mlx.core as mx

def load_mnist():
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    os.makedirs("data/mnist", exist_ok=True)

    def fetch(fname):
        path = f"data/mnist/{fname}"
        if not os.path.exists(path):
            urllib.request.urlretrieve(base + fname, path)
        with gzip.open(path, "rb") as f:
            return f.read()

    train_x = mx.array(
        list(fetch("train-images-idx3-ubyte.gz")[16:]), dtype=mx.float32
    ).reshape(60000, 784) / 255.0

    train_y = mx.array(
        list(fetch("train-labels-idx1-ubyte.gz")[8:]), dtype=mx.uint32
    )
    return train_x, train_y
```

---

## Tips

- Keep data as `mx.array` throughout — avoid converting to Python lists in hot loops.
- For large datasets that don't fit in memory, override `__getitem__` to load from disk lazily.
- `DataLoader` shuffles via `mx.random.permutation`, which runs on the GPU.
