"""test_saving_loading.py — verify that save/load preserves weights exactly.

Run after mnist.py has produced cnn_mnist.npz:
    python example/mnist.py
    python example/test_saving_loading.py
"""
from __future__ import annotations
import mlx.core as mx
import uma.nn as nn
import uma.eval as ueval
from uma.data.dataloader import DataLoader
from uma.data.dataset import Dataset


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
        n = x.shape[0]
        x = x.reshape(n, 28, 28, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(n, -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def load_test_data() -> tuple[mx.array, mx.array]:
    import urllib.request, gzip, os
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    os.makedirs("data/mnist", exist_ok=True)

    def fetch(fname: str) -> bytes:
        path = f"data/mnist/{fname}"
        if not os.path.exists(path):
            print(f"downloading {fname}...")
            urllib.request.urlretrieve(base + fname, path)
        with gzip.open(path, "rb") as f:
            return f.read()

    test_x = mx.array(
        list(fetch("t10k-images-idx3-ubyte.gz")[16:]), dtype=mx.float32
    ).reshape(10000, 784) / 255.0
    test_y = mx.array(
        list(fetch("t10k-labels-idx1-ubyte.gz")[8:]), dtype=mx.uint32
    )
    return test_x, test_y


if __name__ == "__main__":
    mx.set_default_device(mx.Device(mx.gpu))

    print("loading test data...")
    test_x, test_y = load_test_data()

    # ── load trained model ───────────────────────────────────────────
    original = CNN()
    original.load("cnn_mnist.npz")
    acc_original = ueval.accuracy(original, test_x, test_y)
    print(f"original model accuracy : {acc_original * 100:.2f}%")

    # ── save to a new file, load into a fresh model ──────────────────
    original.save("cnn_mnist_copy")
    reloaded = CNN()
    reloaded.load("cnn_mnist_copy.npz")
    acc_reloaded = ueval.accuracy(reloaded, test_x, test_y)
    print(f"reloaded model accuracy : {acc_reloaded * 100:.2f}%")

    # ── verify weights are bit-for-bit identical ─────────────────────
    orig_sd = original.state_dict()
    rel_sd = reloaded.state_dict()
    all_match = all(
        mx.array_equal(orig_sd[k], rel_sd[k]).item()
        for k in orig_sd
    )
    print(f"weights identical       : {all_match}")

    # ── partial load (strict=False) ──────────────────────────────────
    partial = CNN()
    partial.load_state_dict({"fc2.weight": orig_sd["fc2.weight"]}, strict=False)
    print("partial load (strict=False): ok")

    if all_match and abs(acc_original - acc_reloaded) < 1e-6:
        print("\n[pass] save/load round-trip is exact.")
    else:
        print("\n[fail] mismatch after save/load.")
