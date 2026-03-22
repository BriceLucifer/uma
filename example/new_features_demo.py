"""Demo of all new uma features on a simple synthetic dataset.

This script exercises:
  - nn.Sequential
  - nn.ModuleList
  - nn.BatchNorm1d
  - nn.Sigmoid, nn.Tanh, nn.LeakyReLU
  - nn.Embedding
  - optim.AdamW
  - optim.CosineAnnealingLR, StepLR, ReduceLROnPlateau
  - uma.binary_cross_entropy, uma.huber
  - Trainer clip_grad_norm
  - Trainer early_stopping_patience
  - uma.eval.precision_recall_f1
"""

import mlx.core as mx
import uma
import uma.nn as nn
import uma.optim as optim
from uma.data.dataset import Dataset
from uma.data.dataloader import DataLoader
from uma.trainer import Trainer

mx.set_default_device(mx.Device(mx.gpu))


# ── helpers ────────────────────────────────────────────────────────────────

class ArrayDataset(Dataset):
    def __init__(self, x: mx.array, y: mx.array):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


# ── 1. Sequential ──────────────────────────────────────────────────────────

print_section("1. nn.Sequential — multi-class classification (10 classes)")

N, D, C = 1000, 32, 10
x_train = mx.random.normal((N, D))
y_train = mx.array(mx.random.randint(low=0, high=C, shape=(N,)).tolist(), dtype=mx.uint32)
x_val   = mx.random.normal((200, D))
y_val   = mx.array(mx.random.randint(low=0, high=C, shape=(200,)).tolist(), dtype=mx.uint32)

model = nn.Sequential(
    nn.Linear(D, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, C),
)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = optim.CosineAnnealingLR(optimizer, T_max=5)
loss_fn = lambda m, x, y: uma.cross_entropy(m(x), y)

trainer = Trainer(model, optimizer, loss_fn, clip_grad_norm=1.0)
history = trainer.fit(
    DataLoader(ArrayDataset(x_train, y_train), batch_size=128, shuffle=True),
    epochs=5,
    val_data=(x_val, y_val),
    metric_fn=uma.eval.accuracy,
    metric_name="acc",
    scheduler=scheduler,
)

acc = uma.eval.accuracy(model, x_val, y_val)
p, r, f1 = uma.eval.precision_recall_f1(model, x_val, y_val, num_classes=C)
print(f"\nFinal val accuracy : {acc:.4f}")
print(f"Precision / Recall / F1 (macro): {p:.4f} / {r:.4f} / {f1:.4f}")


# ── 2. ModuleList — residual MLP ───────────────────────────────────────────

print_section("2. nn.ModuleList — residual MLP")

class ResidualMLP(nn.Module):
    def __init__(self, dim: int, n_layers: int):
        self.proj = nn.Linear(D, dim)
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.norms  = nn.ModuleList([nn.BatchNorm1d(dim) for _ in range(n_layers)])
        self.act    = nn.LeakyReLU(negative_slope=0.1)
        self.head   = nn.Linear(dim, C)

    def forward(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        for i in range(len(self.layers)):
            x = x + self.act(self.norms[i](self.layers[i](x)))
        return self.head(x)

res_model = ResidualMLP(dim=64, n_layers=3)
res_opt   = optim.Adam(res_model.parameters(), lr=5e-4)
res_sched = optim.StepLR(res_opt, step_size=2, gamma=0.5)

res_trainer = Trainer(res_model, res_opt, loss_fn, clip_grad_norm=2.0)
res_trainer.fit(
    DataLoader(ArrayDataset(x_train, y_train), batch_size=128, shuffle=True),
    epochs=6,
    val_data=(x_val, y_val),
    metric_fn=uma.eval.accuracy,
    metric_name="acc",
    scheduler=res_sched,
)
print(f"\nResidualMLP val accuracy: {uma.eval.accuracy(res_model, x_val, y_val):.4f}")


# ── 3. Early stopping ──────────────────────────────────────────────────────

print_section("3. Early stopping (patience=3)")

early_model = nn.Sequential(
    nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, C)
)
early_opt = optim.SGD(early_model.parameters(), lr=1e-2)
early_trainer = Trainer(early_model, early_opt, loss_fn)

early_trainer.fit(
    DataLoader(ArrayDataset(x_train, y_train), batch_size=128, shuffle=True),
    epochs=50,
    val_data=(x_val, y_val),
    metric_fn=uma.eval.accuracy,
    metric_name="acc",
    early_stopping_patience=3,
    early_stopping_mode="max",
)


# ── 4. Binary classification with Sigmoid + BCE ────────────────────────────

print_section("4. Binary classification — Sigmoid + binary_cross_entropy")

x_bin = mx.random.normal((800, 16))
y_bin = mx.array(
    mx.random.randint(low=0, high=2, shape=(800,)).tolist(), dtype=mx.float32
)
x_val_bin = mx.random.normal((200, 16))
y_val_bin = mx.array(
    mx.random.randint(low=0, high=2, shape=(200,)).tolist(), dtype=mx.float32
)

bin_model = nn.Sequential(
    nn.Linear(16, 32),
    nn.Tanh(),
    nn.Linear(32, 1),
)

def bce_loss(m, x, y):
    logits = m(x).squeeze(-1)   # (N,)
    return uma.binary_cross_entropy(logits, y)

bin_opt = optim.Adam(bin_model.parameters(), lr=1e-3)
bin_trainer = Trainer(bin_model, bin_opt, bce_loss)
bin_trainer.fit(
    DataLoader(ArrayDataset(x_bin, y_bin), batch_size=64, shuffle=True),
    epochs=5,
)
print("Binary model trained successfully.")


# ── 5. Regression with Huber loss ─────────────────────────────────────────

print_section("5. Regression — Huber loss")

x_reg = mx.random.normal((500, 8))
y_reg = mx.sum(x_reg, axis=-1, keepdims=True) + mx.random.normal((500, 1)) * 0.1

reg_model = nn.Sequential(
    nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1)
)

def huber_loss(m, x, y):
    return uma.huber(m(x), y, delta=1.0)

reg_opt = optim.AdamW(reg_model.parameters(), lr=1e-3)
reg_trainer = Trainer(reg_model, reg_opt, huber_loss)
reg_trainer.fit(
    DataLoader(ArrayDataset(x_reg, y_reg), batch_size=64, shuffle=True),
    epochs=5,
)
print("Regression model trained successfully.")


# ── 6. Embedding layer ─────────────────────────────────────────────────────

print_section("6. nn.Embedding — token classification")

VOCAB, SEQ, EMB = 100, 10, 32

class TokenClassifier(nn.Module):
    def __init__(self):
        self.embed = nn.Embedding(VOCAB, EMB)
        self.fc    = nn.Linear(EMB, 5)

    def forward(self, x: mx.array) -> mx.array:
        # x: (N, SEQ) int → embed → (N, SEQ, EMB) → mean → (N, EMB)
        emb = self.embed(x)           # (N, SEQ, EMB)
        pooled = mx.mean(emb, axis=1) # (N, EMB)
        return self.fc(pooled)        # (N, 5)

x_tok = mx.array(
    mx.random.randint(low=0, high=VOCAB, shape=(400, SEQ)).tolist(), dtype=mx.uint32
)
y_tok = mx.array(
    mx.random.randint(low=0, high=5, shape=(400,)).tolist(), dtype=mx.uint32
)

tok_model = TokenClassifier()
tok_opt   = optim.Adam(tok_model.parameters(), lr=1e-3)
tok_trainer = Trainer(tok_model, tok_opt, loss_fn)
tok_trainer.fit(
    DataLoader(ArrayDataset(x_tok, y_tok), batch_size=64, shuffle=True),
    epochs=5,
)
print("Token classifier trained successfully.")


# ── 7. ReduceLROnPlateau ──────────────────────────────────────────────────

print_section("7. ReduceLROnPlateau scheduler")

plat_model = nn.Sequential(nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, C))
plat_opt   = optim.Adam(plat_model.parameters(), lr=1e-2)
plat_sched = optim.ReduceLROnPlateau(plat_opt, mode="max", factor=0.5, patience=2)

plat_trainer = Trainer(plat_model, plat_opt, loss_fn)
plat_trainer.fit(
    DataLoader(ArrayDataset(x_train, y_train), batch_size=128, shuffle=True),
    epochs=8,
    val_data=(x_val, y_val),
    metric_fn=uma.eval.accuracy,
    metric_name="acc",
    scheduler=plat_sched,
)

print("\nAll demos completed successfully.")
