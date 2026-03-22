# Saving & Loading Weights

uma serialises model weights to `.npz` format using MLX's native `mx.savez` / `mx.load`. No numpy or torch required.

---

## Save

```python
model.save("checkpoint")       # writes checkpoint.npz
model.save("checkpoints/ep10") # directories must exist
```

All parameters are stored under their dotted key names (dots replaced with `/` inside the zip).

---

## Load

```python
model2 = MyModel()
model2.load("checkpoint.npz")
```

The model architecture must match — key names and shapes must be identical to those used when saving.

---

## state_dict / load_state_dict

For manual control over what gets saved or loaded:

```python
# get the parameter dict
sd = model.state_dict()
# {"fc1.weight": mx.array(...), "fc1.bias": mx.array(...), ...}

# load into another model (strict by default)
model2.load_state_dict(sd)
```

### Strict mode

`strict=True` (default) raises `KeyError` if there are missing or unexpected keys:

```python
model2.load_state_dict(sd)           # raises if any key mismatches
model2.load_state_dict(sd, strict=False)  # skips missing/extra keys silently
```

### Partial load (transfer learning)

Load only the backbone weights and skip the classifier head:

```python
pretrained = mx.load("backbone.npz")
sd = {k.replace("/", "."): v for k, v in pretrained.items()}
model.load_state_dict(sd, strict=False)   # head initialised randomly
```

---

## Round-trip test

Run [`example/test_saving_loading.py`](https://github.com/BriceLucifer/uma/blob/main/example/test_saving_loading.py) after training to verify weights are bit-for-bit identical after save/load:

```bash
python example/mnist.py               # trains, writes cnn_mnist.npz
python example/test_saving_loading.py # verifies round-trip
```

Output:
```
original model accuracy : 99.21%
reloaded model accuracy : 99.21%
weights identical       : True
partial load (strict=False): ok

[pass] save/load round-trip is exact.
```

---

## Notes

- Shapes are **not validated** at load time — MLX validates lazily when the graph is evaluated. A wrong-shape load will crash on the first `forward` call, not at `load_state_dict`.
- Key names use dot notation: `conv1.weight`, `fc2.bias`, etc. Renaming a layer attribute between saving and loading breaks the load.
