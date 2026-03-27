"""Recurrent modules: RNN, LSTM, GRU (single- and multi-layer)."""
from __future__ import annotations
import mlx.core as mx
from .module import Module
from .linear import Linear


# ── internal single-layer cells ───────────────────────────────────────────────

class _RNNCell(Module):
    """Single Elman RNN layer."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool) -> None:
        self.ih = Linear(input_size, hidden_size, bias=bias)
        self.hh = Linear(hidden_size, hidden_size, bias=bias)


class _LSTMCell(Module):
    """Single LSTM layer (fused 4-gate projection)."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool) -> None:
        self.ih = Linear(input_size, 4 * hidden_size, bias=bias)
        self.hh = Linear(hidden_size, 4 * hidden_size, bias=bias)


class _GRUCell(Module):
    """Single GRU layer."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool) -> None:
        self.ih_rz = Linear(input_size, 2 * hidden_size, bias=bias)
        self.hh_rz = Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.ih_n  = Linear(input_size, hidden_size, bias=bias)
        self.hh_n  = Linear(hidden_size, hidden_size, bias=bias)


# ── public modules ─────────────────────────────────────────────────────────────

class RNN(Module):
    """Elman recurrent neural network (tanh or ReLU), optionally stacked.

    Processes a sequence of shape ``(batch, seq_len, input_size)`` and returns
    the full sequence of hidden states from the **last** layer,
    shape ``(batch, seq_len, hidden_size)``.

    Args:
        input_size:   Number of expected features in the input.
        hidden_size:  Number of features in the hidden state.
        num_layers:   Number of stacked RNN layers (default 1).
        nonlinearity: ``"tanh"`` (default) or ``"relu"``.
        bias:         If ``False``, no bias terms are used.

    Initial hidden state ``h0``:
        - ``None`` — zeros for all layers.
        - ``mx.array`` of shape ``(batch, hidden_size)`` — used for every layer.
        - ``mx.array`` of shape ``(num_layers, batch, hidden_size)`` — per-layer.

    Example::

        rnn = RNN(input_size=32, hidden_size=64, num_layers=2)
        x   = mx.random.normal((8, 10, 32))
        out = rnn(x)   # (8, 10, 64)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
    ) -> None:
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError(f"nonlinearity must be 'tanh' or 'relu', got {nonlinearity!r}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        for i in range(num_layers):
            in_sz = input_size if i == 0 else hidden_size
            setattr(self, f"_layer_{i}", _RNNCell(in_sz, hidden_size, bias))

    def forward(self, x: mx.array, h0: mx.array | None = None) -> mx.array:
        batch, seq_len, _ = x.shape
        H, L = self.hidden_size, self.num_layers

        # Build per-layer initial hidden states
        if h0 is None:
            hs = [mx.zeros((batch, H)) for _ in range(L)]
        elif h0.ndim == 2:                  # (batch, H) — broadcast to all layers
            hs = [h0] * L
        else:                               # (num_layers, batch, H)
            hs = [h0[i] for i in range(L)]

        # Run each layer
        inp = x
        for i in range(L):
            cell = getattr(self, f"_layer_{i}")
            h = hs[i]
            outputs = []
            for t in range(seq_len):
                xt = inp[:, t, :]
                pre = cell.ih(xt) + cell.hh(h)
                h = mx.tanh(pre) if self.nonlinearity == "tanh" else mx.maximum(pre, 0)
                outputs.append(h[:, None, :])
            inp = mx.concatenate(outputs, axis=1)   # feed into the next layer
        return inp


class LSTM(Module):
    """Long Short-Term Memory, optionally stacked.

    Processes ``(batch, seq_len, input_size)`` and returns the full sequence
    of hidden states from the **last** layer,
    shape ``(batch, seq_len, hidden_size)``.

    Args:
        input_size:  Number of expected features in the input.
        hidden_size: Number of features in the hidden state.
        num_layers:  Number of stacked LSTM layers (default 1).
        bias:        If ``False``, no bias terms are used.

    Initial state ``state``:
        - ``None`` — zeros for all layers.
        - Tuple ``(h0, c0)`` where each is ``(batch, H)`` (broadcast) or
          ``(num_layers, batch, H)`` (per-layer).

    Example::

        lstm = LSTM(input_size=32, hidden_size=128, num_layers=2)
        x    = mx.random.normal((4, 20, 32))
        out  = lstm(x)   # (4, 20, 128)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
    ) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        for i in range(num_layers):
            in_sz = input_size if i == 0 else hidden_size
            setattr(self, f"_layer_{i}", _LSTMCell(in_sz, hidden_size, bias))

    def forward(
        self,
        x: mx.array,
        state: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        batch, seq_len, _ = x.shape
        H, L = self.hidden_size, self.num_layers

        if state is None:
            hs = [mx.zeros((batch, H)) for _ in range(L)]
            cs = [mx.zeros((batch, H)) for _ in range(L)]
        else:
            h0, c0 = state
            if h0.ndim == 2:
                hs = [h0] * L
                cs = [c0] * L
            else:
                hs = [h0[i] for i in range(L)]
                cs = [c0[i] for i in range(L)]

        inp = x
        for i in range(L):
            cell = getattr(self, f"_layer_{i}")
            h, c = hs[i], cs[i]
            outputs = []
            for t in range(seq_len):
                xt = inp[:, t, :]
                gates = cell.ih(xt) + cell.hh(h)
                ig = mx.sigmoid(gates[:, :H])
                fg = mx.sigmoid(gates[:, H:2 * H])
                g  = mx.tanh(gates[:, 2 * H:3 * H])
                og = mx.sigmoid(gates[:, 3 * H:])
                c  = fg * c + ig * g
                h  = og * mx.tanh(c)
                outputs.append(h[:, None, :])
            inp = mx.concatenate(outputs, axis=1)
        return inp


class GRU(Module):
    """Gated Recurrent Unit, optionally stacked.

    Processes ``(batch, seq_len, input_size)`` and returns the full sequence
    of hidden states from the **last** layer,
    shape ``(batch, seq_len, hidden_size)``.

    Args:
        input_size:  Number of expected features in the input.
        hidden_size: Number of features in the hidden state.
        num_layers:  Number of stacked GRU layers (default 1).
        bias:        If ``False``, no bias terms are used.

    Initial hidden state ``h0``:
        - ``None`` — zeros for all layers.
        - ``mx.array`` of shape ``(batch, hidden_size)`` — broadcast to all layers.
        - ``mx.array`` of shape ``(num_layers, batch, hidden_size)`` — per-layer.

    Example::

        gru = GRU(input_size=32, hidden_size=64, num_layers=2)
        x   = mx.random.normal((4, 15, 32))
        out = gru(x)   # (4, 15, 64)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
    ) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        for i in range(num_layers):
            in_sz = input_size if i == 0 else hidden_size
            setattr(self, f"_layer_{i}", _GRUCell(in_sz, hidden_size, bias))

    def forward(self, x: mx.array, h0: mx.array | None = None) -> mx.array:
        batch, seq_len, _ = x.shape
        H, L = self.hidden_size, self.num_layers

        if h0 is None:
            hs = [mx.zeros((batch, H)) for _ in range(L)]
        elif h0.ndim == 2:
            hs = [h0] * L
        else:
            hs = [h0[i] for i in range(L)]

        inp = x
        for i in range(L):
            cell = getattr(self, f"_layer_{i}")
            h = hs[i]
            outputs = []
            for t in range(seq_len):
                xt = inp[:, t, :]
                rz = mx.sigmoid(cell.ih_rz(xt) + cell.hh_rz(h))
                r, z = rz[:, :H], rz[:, H:]
                n = mx.tanh(cell.ih_n(xt) + r * cell.hh_n(h))
                h = (1 - z) * n + z * h
                outputs.append(h[:, None, :])
            inp = mx.concatenate(outputs, axis=1)
        return inp
