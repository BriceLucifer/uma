from __future__ import annotations
from typing import Any, Iterator
import mlx.core as mx


class Module:
    """Base class for all neural network modules."""

    def __call__(self, *args: Any, **kwargs: Any) -> mx.array:
        return self.forward(*args, **kwargs)

    def forward(self, x: mx.array) -> Any:
        raise NotImplementedError

    def parameters(self) -> dict[str, mx.array]:
        params: dict[str, mx.array] = {}
        for name, val in self.__dict__.items():
            if isinstance(val, mx.array):
                params[name] = val
            elif isinstance(val, Module):
                for k, v in val.parameters().items():
                    params[f"{name}.{k}"] = v
        return params

    def update(self, updates: dict[str, mx.array]) -> None:
        """Apply in-place parameter updates (MLX style)."""
        for name, val in updates.items():
            parts = name.split(".", 1)
            if len(parts) == 1:
                setattr(self, name, val)
            else:
                child = getattr(self, parts[0])
                child.update({parts[1]: val})

    def named_modules(self) -> Iterator[tuple[str, Module]]:
        yield "", self
        for name, val in self.__dict__.items():
            if isinstance(val, Module):
                for subname, mod in val.named_modules():
                    yield (f"{name}.{subname}" if subname else name, mod)

    def train(self) -> None:
        for _, mod in self.named_modules():
            if hasattr(mod, "training"):
                setattr(mod, "training", True)

    def eval(self) -> None:
        for _, mod in self.named_modules():
            if hasattr(mod, "training"):
                setattr(mod, "training", False)

    # ── serialisation ─────────────────────────────────────────────

    def state_dict(self) -> dict[str, mx.array]:
        """Return a flat copy of all parameters (same as parameters())."""
        return dict(self.parameters())

    def load_state_dict(self, sd: dict[str, mx.array], strict: bool = True) -> None:
        """Load a state dict produced by state_dict() or uma.bridge.

        Args:
            sd:     Flat dict of {dotted.key: mx.array}.
            strict: If True (default), raise KeyError on missing/extra keys.
                    Set False to do a partial load (useful when loading a
                    pre-trained backbone and ignoring head weights).

        Failure modes:
            - Shape mismatches are not checked here — MLX will raise inside
              update() only when the array is actually evaluated.
            - strict=False silently skips keys that don't exist on the model;
              typos in key names will not be caught.
        """
        own = set(self.parameters().keys())
        incoming = set(sd.keys())
        if strict:
            missing = own - incoming
            unexpected = incoming - own
            if missing:
                raise KeyError(f"load_state_dict: missing keys: {sorted(missing)}")
            if unexpected:
                raise KeyError(f"load_state_dict: unexpected keys: {sorted(unexpected)}")
        filtered = {k: v for k, v in sd.items() if k in own}
        self.update(filtered)

    def save(self, path: str) -> None:
        """Save parameters to a .npz file.

        Uses MLX's native mx.savez — no numpy or torch required.
        Dots in key names are stored as '/' (MLX npz convention).

        Usage:
            model.save("weights")    # writes weights.npz
        """
        mx.eval(list(self.parameters().values()))
        arrays = {k.replace(".", "/"): v for k, v in self.parameters().items()}
        mx.savez(path, **arrays)

    def load(self, path: str) -> None:
        """Load parameters from a .npz file saved by save().

        Usage:
            model.load("weights.npz")

        Failure modes:
            - Keys in the file must exactly match the model's current
              parameter names (after replacing '/' back to '.').
              A renamed layer will cause a KeyError in load_state_dict.
            - Shapes are not validated until MLX evaluates the graph.
        """
        file = path if path.endswith(".npz") else path + ".npz"
        data = mx.load(file)
        sd = {k.replace("/", "."): v for k, v in data.items()}
        self.load_state_dict(sd)