# Evaluation

`uma.eval` provides metric functions for classification models. All functions:

- Call `model.eval()` automatically before inference.
- Process data in batches (default `batch_size=512`) to avoid OOM on large test sets.
- Expect `y` as **integer class indices** (`uint32` or `int32`), not one-hot.

---

## Accuracy

```python
import uma.eval as ueval

acc = ueval.accuracy(model, test_x, test_y)
print(f"{acc * 100:.2f}%")   # e.g. 99.21%
```

Optional `batch_size` argument:

```python
acc = ueval.accuracy(model, test_x, test_y, batch_size=1024)
```

---

## Top-k accuracy

Fraction of samples where the true label appears in the top-k predictions:

```python
top5 = ueval.top_k_accuracy(model, test_x, test_y, k=5)
```

---

## Confusion matrix

Returns an `mx.array` of shape `(num_classes, num_classes)` with `int32` dtype.
Entry `[i, j]` = number of samples with true label `i` predicted as `j`.

```python
cm = ueval.confusion_matrix(model, test_x, test_y, num_classes=10)
print(cm)
```

```
array([[978,   0,   1,   0,   0,   0,   0,   1,   0,   0],
       [  0,1133,   2,   0,   0,   0,   0,   0,   0,   0],
       ...], dtype=int32)
```

---

## Using metrics in the Trainer

Pass any `(model, x, y) -> float` callable as `metric_fn`:

```python
trainer.fit(
    loader,
    epochs=20,
    val_data=(test_x, test_y),
    metric_fn=ueval.accuracy,
    metric_name="acc",
)
```

After each epoch the trainer calls `metric_fn`, prints the score, then restores `model.train()`.

---

## Restoring training mode

After calling any eval function directly, remember to call `model.train()` before resuming training:

```python
acc = ueval.accuracy(model, test_x, test_y)   # sets model to eval
model.train()                                  # restore for next epoch
```

The `Trainer` handles this automatically inside `fit`.
