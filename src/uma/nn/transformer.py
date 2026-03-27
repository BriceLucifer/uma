"""Transformer modules: MultiheadAttention, TransformerEncoderLayer, TransformerEncoder."""
from __future__ import annotations
import math
import mlx.core as mx
from .module import Module
from .linear import Linear
from .normalization import LayerNorm
from .dropout import Dropout
from .activations import ReLU, GELU


class MultiheadAttention(Module):
    """Multi-head scaled dot-product attention.

    Args:
        embed_dim:   Total embedding dimension.
        num_heads:   Number of parallel attention heads.
                     ``embed_dim`` must be divisible by ``num_heads``.
        dropout:     Dropout probability applied to attention weights (default 0.0).
        bias:        Add bias to projection layers (default ``True``).

    Example::

        attn = MultiheadAttention(embed_dim=128, num_heads=8)
        x    = mx.random.normal((4, 20, 128))   # (batch, seq, embed_dim)
        out  = attn(x, x, x)                    # (4, 20, 128)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.attn_drop = Dropout(dropout) if dropout > 0.0 else None

    def forward(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Compute multi-head attention.

        Args:
            query: ``(batch, tgt_len, embed_dim)``
            key:   ``(batch, src_len, embed_dim)``
            value: ``(batch, src_len, embed_dim)``
            mask:  Optional additive mask of shape ``(tgt_len, src_len)`` or
                   ``(batch, tgt_len, src_len)``. Large negative values (e.g.
                   ``-1e9``) block positions before softmax.

        Returns:
            Output of shape ``(batch, tgt_len, embed_dim)``.
        """
        B, T, _ = query.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(query).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, -1, H, D).transpose(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, -1, H, D).transpose(0, 2, 1, 3)

        # (B, H, T, S)
        attn = (q @ k.transpose(0, 1, 3, 2)) / self.scale
        if mask is not None:
            attn = attn + mask
        attn = mx.softmax(attn, axis=-1)
        if self.attn_drop is not None:
            attn = self.attn_drop(attn)

        # (B, H, T, D) → (B, T, embed_dim)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        return self.out_proj(out)


class TransformerEncoderLayer(Module):
    """A single Transformer encoder layer.

    Follows the Pre-LN or Post-LN formulation:
    ``FFN(Norm(x + Attention(x)))`` (post-norm, the classic "vanilla" transformer).

    Args:
        d_model:     Input/output embedding dimension.
        nhead:       Number of attention heads.
        dim_feedforward: Width of the feedforward sublayer (default ``4 * d_model``).
        dropout:     Dropout probability (default 0.0).
        activation:  ``"relu"`` (default) or ``"gelu"``.

    Example::

        layer = TransformerEncoderLayer(d_model=128, nhead=8)
        x     = mx.random.normal((4, 20, 128))
        out   = layer(x)                        # (4, 20, 128)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        if activation not in ("relu", "gelu"):
            raise ValueError(f"activation must be 'relu' or 'gelu', got {activation!r}")
        dim_feedforward = dim_feedforward or 4 * d_model

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ff1 = Linear(d_model, dim_feedforward)
        self.ff2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.drop1 = Dropout(dropout) if dropout > 0.0 else None
        self.drop2 = Dropout(dropout) if dropout > 0.0 else None
        self.act = ReLU() if activation == "relu" else GELU()

    def forward(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        # Self-attention sublayer
        attn_out = self.self_attn(x, x, x, mask=mask)
        if self.drop1 is not None:
            attn_out = self.drop1(attn_out)
        x = self.norm1(x + attn_out)

        # Feed-forward sublayer
        ff_out = self.ff2(self.act(self.ff1(x)))
        if self.drop2 is not None:
            ff_out = self.drop2(ff_out)
        x = self.norm2(x + ff_out)
        return x


class TransformerEncoder(Module):
    """Stack of N TransformerEncoderLayer modules.

    Args:
        encoder_layer: A single ``TransformerEncoderLayer`` instance that is
                       cloned ``num_layers`` times (each clone has its own
                       independent weights).
        num_layers:    Number of sub-encoder-layers.

    .. note::
        Layers are stored as ``_layer_0``, ``_layer_1``, … so that
        ``Module.parameters()`` recurses correctly.

    Example::

        layer   = TransformerEncoderLayer(d_model=128, nhead=8, dropout=0.1)
        encoder = TransformerEncoder(layer, num_layers=4)
        x       = mx.random.normal((4, 20, 128))
        out     = encoder(x)                    # (4, 20, 128)
    """

    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int) -> None:
        import copy
        self._num_layers = num_layers
        for i in range(num_layers):
            setattr(self, f"_layer_{i}", copy.deepcopy(encoder_layer))

    def forward(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        for i in range(self._num_layers):
            x = getattr(self, f"_layer_{i}")(x, mask=mask)
        return x
