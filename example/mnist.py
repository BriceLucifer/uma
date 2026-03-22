from __future__ import annotations
import mlx.core as mx
import uma
import uma.nn as nn
import uma.optim as optim
import uma.eval as ueval
from uma.data.dataset import Dataset
from uma.data.dataloader import DataLoader
from uma.trainer import Trainer


# ── 1. 下载 & 加载数据 ──────────────────────────────────────────
def load_mnist() -> tuple[mx.array, mx.array, mx.array, mx.array]:
    import urllib.request
    import gzip
    import os

    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_x": "train-images-idx3-ubyte.gz",
        "train_y": "train-labels-idx1-ubyte.gz",
        "test_x":  "t10k-images-idx3-ubyte.gz",
        "test_y":  "t10k-labels-idx1-ubyte.gz",
    }

    os.makedirs("data/mnist", exist_ok=True)

    def fetch(filename: str) -> bytes:
        path = f"data/mnist/{filename}"
        if not os.path.exists(path):
            print(f"downloading {filename}...")
            urllib.request.urlretrieve(base + filename, path)
        with gzip.open(path, "rb") as f:
            return f.read()

    train_x = mx.array(
        list(fetch(files["train_x"])[16:]), dtype=mx.float32
    ).reshape(60000, 784) / 255.0

    train_y = mx.array(
        list(fetch(files["train_y"])[8:]), dtype=mx.uint32
    )

    test_x = mx.array(
        list(fetch(files["test_x"])[16:]), dtype=mx.float32
    ).reshape(10000, 784) / 255.0

    test_y = mx.array(
        list(fetch(files["test_y"])[8:]), dtype=mx.uint32
    )

    return train_x, train_y, test_x, test_y


# ── 2. Dataset ──────────────────────────────────────────────────
class MNISTDataset(Dataset):
    def __init__(self, x: mx.array, y: mx.array) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[mx.array, mx.array]:
        return self.x[idx], self.y[idx]


# ── 3a. MLP model (baseline) ────────────────────────────────────
class MLP(nn.Module):
    def __init__(self) -> None:
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: mx.array) -> mx.array:
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# ── 3b. CNN model ────────────────────────────────────────────────
# Input arrives flat (N, 784).  CNN needs (N, H, W, C) channels-last
# (MLX convention — not PyTorch's (N, C, H, W)).
# We reshape inside forward so the Dataset/DataLoader stay unchanged.
#
# Architecture:
#   784 → reshape (N,28,28,1)
#   Conv2d(1→32, k=3, pad=1) → ReLU → MaxPool2d(2)   # (N,14,14,32)
#   Conv2d(32→64, k=3, pad=1) → ReLU → MaxPool2d(2)  # (N, 7, 7,64)
#   flatten → (N, 3136)
#   Linear(3136→128) → ReLU → Dropout(0.3)
#   Linear(128→10)
#
# Where things can go wrong:
#   1. channels-last reshape — forgetting to reshape or using wrong order
#      (N,1,28,28) will silently convolve along the wrong axes.
#   2. MaxPool2d requires spatial dims divisible by kernel_size.
#      28 → 14 → 7: all fine.  If you crop to 27px, pool fails at runtime.
#   3. parameters() doesn't recurse into Python lists/dicts.
#      All conv/linear layers must be direct Module attributes (self.conv1,
#      not self.layers = [Conv2d(...), ...]).
#   4. value_and_grad re-runs forward inside the closure; don't put
#      side-effectful ops (printing, random seeds) inside forward.
class CNN(nn.Module):
    def __init__(self) -> None:
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: mx.array) -> mx.array:
        # x: (N, 784) flat → (N, 28, 28, 1) channels-last
        n = x.shape[0]
        x = x.reshape(n, 28, 28, 1)

        x = self.relu(self.conv1(x))   # (N, 28, 28, 32)
        x = self.pool(x)               # (N, 14, 14, 32)

        x = self.relu(self.conv2(x))   # (N, 14, 14, 64)
        x = self.pool(x)               # (N,  7,  7, 64)

        x = x.reshape(n, -1)           # (N, 3136)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ── 4. Training ──────────────────────────────────────────────────
if __name__ == "__main__":
    mx.set_default_device(mx.Device(mx.gpu))
    print("loading mnist...")
    train_x, train_y, test_x, test_y = load_mnist()
    print(f"train: {train_x.shape}  test: {test_x.shape}")

    train_loader = DataLoader(
        MNISTDataset(train_x, train_y),
        batch_size=256,
        shuffle=True,
    )

    # swap MLP() for CNN() to try the conv model
    model = CNN()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    def loss_fn(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
        return uma.cross_entropy(model(x), y)

    trainer = Trainer(model, optimizer, loss_fn)
    history = trainer.fit(
        train_loader,
        epochs=20,
        val_data=(test_x, test_y),
        metric_fn=ueval.accuracy,
        metric_name="acc",
    )

    # final evaluation with full metrics
    acc = ueval.accuracy(model, test_x, test_y)
    top5 = ueval.top_k_accuracy(model, test_x, test_y, k=5)
    cm = ueval.confusion_matrix(model, test_x, test_y, num_classes=10)
    print(f"\ntest accuracy : {acc * 100:.2f}%")
    print(f"top-5 accuracy: {top5 * 100:.2f}%")
    print("confusion matrix:")
    print(cm)

    # ── save weights ────────────────────────────────────────────────
    model.save("cnn_mnist")
    print("\nweights saved to cnn_mnist.npz")

    # ── reload and verify ───────────────────────────────────────────
    model2 = CNN()
    model2.load("cnn_mnist.npz")
    acc2 = ueval.accuracy(model2, test_x, test_y)
    print(f"reloaded model accuracy: {acc2 * 100:.2f}%  (should match {acc * 100:.2f}%)")
