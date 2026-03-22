"""Module containers: Sequential and ModuleList."""
from __future__ import annotations
import mlx.core as mx
from .module import Module


class Sequential(Module):
    """A sequential container that chains modules in order.

    Modules are called in the order they are passed, with the output of
    each forwarded to the next.

    Args:
        *layers: Modules to chain in order.

    Example::

        model = Sequential(
            Linear(784, 256),
            ReLU(),
            Linear(256, 10),
        )
        out = model(x)
    """

    def __init__(self, *layers: Module) -> None:
        # Store as _layer_0, _layer_1, ... so Module.parameters() recurses
        for i, layer in enumerate(layers):
            setattr(self, f"_layer_{i}", layer)
        self._num_layers = len(layers)

    def forward(self, x: mx.array) -> mx.array:
        for i in range(self._num_layers):
            x = getattr(self, f"_layer_{i}")(x)
        return x

    def __getitem__(self, idx: int) -> Module:
        return getattr(self, f"_layer_{idx}")

    def __len__(self) -> int:
        return self._num_layers


class ModuleList(Module):
    """A list container that registers sub-modules so their parameters are tracked.

    Unlike a plain Python list, sub-modules stored here are visible to
    ``parameters()``, ``train()``, ``eval()``, and ``save()``/``load()``.

    Args:
        modules: Iterable of Modules to register.

    Example::

        self.layers = ModuleList([Linear(64, 64) for _ in range(4)])
        # in forward:
        for i in range(len(self.layers)):
            x = self.layers[i](x)
    """

    def __init__(self, modules: list[Module]) -> None:
        for i, mod in enumerate(modules):
            setattr(self, f"_item_{i}", mod)
        self._num_items = len(modules)

    def forward(self, x: mx.array) -> mx.array:
        raise NotImplementedError(
            "ModuleList has no default forward. Iterate over it manually."
        )

    def __getitem__(self, idx: int) -> Module:
        return getattr(self, f"_item_{idx}")

    def __len__(self) -> int:
        return self._num_items

    def __iter__(self):
        for i in range(self._num_items):
            yield getattr(self, f"_item_{i}")
