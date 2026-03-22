# Quick Start

This page walks through a complete MNIST training run using uma's CNN and eval modules. The full script lives at [`example/mnist.py`](https://github.com/BriceLucifer/uma/blob/main/example/mnist.py).

---

## 1. Set the device

```python
import mlx.core as mx
mx.set_default_device(mx.Device(mx.gpu))
```

MLX defaults to CPU if you skip this. Always set GPU for training.

---

## 2. Define a model

### MLP (baseline)

```python
import uma.nn as nn

class MLP(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.fc3(x)
```

### CNN

uma uses **channels-last** layout `(N, H, W, C)` — different from PyTorch's `(N, C, H, W)`.
Reshape inside `forward`:

```python
class CNN(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(7 * 7 * 64, 128)
        self.fc2   = nn.Linear(128, 10)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        n = x.shape[0]
        x = x.reshape(n, 28, 28, 1)        # channels-last!
        x = self.relu(self.conv1(x))        # (N, 28, 28, 32)
        x = self.pool(x)                    # (N, 14, 14, 32)
        x = self.relu(self.conv2(x))        # (N, 14, 14, 64)
        x = self.pool(x)                    # (N,  7,  7, 64)
        x = x.reshape(n, -1)               # (N, 3136)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
```

---

## 3. Data pipeline

```python
from uma.data.dataset import Dataset
from uma.data.dataloader import DataLoader

class MNISTDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_loader = DataLoader(MNISTDataset(train_x, train_y), batch_size=256, shuffle=True)
```

---

## 4. Train

```python
import uma
import uma.optim as optim
import uma.eval as ueval
from uma.trainer import Trainer

model = CNN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_fn(model, x, y):
    return uma.cross_entropy(model(x), y)

trainer = Trainer(model, optimizer, loss_fn)

history = trainer.fit(
    train_loader,
    epochs=20,
    val_data=(test_x, test_y),
    metric_fn=ueval.accuracy,
    metric_name="acc",
)
```

Output:
```
epoch 1/20  loss: 0.3821  acc: 0.9134
epoch 2/20  loss: 0.1042  acc: 0.9701
...
```

---

## 5. Evaluate

```python
import uma.eval as ueval

acc  = ueval.accuracy(model, test_x, test_y)
top5 = ueval.top_k_accuracy(model, test_x, test_y, k=5)
cm   = ueval.confusion_matrix(model, test_x, test_y, num_classes=10)

print(f"accuracy : {acc * 100:.2f}%")
print(f"top-5    : {top5 * 100:.2f}%")
print(cm)
```

---

## 6. Save and reload

```python
model.save("cnn_mnist")       # writes cnn_mnist.npz

model2 = CNN()
model2.load("cnn_mnist.npz")  # exact same weights
```

See [Saving & Loading](guide/weights.md) for partial loads and `state_dict` usage.
