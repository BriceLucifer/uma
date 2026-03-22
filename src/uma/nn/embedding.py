"""Embedding lookup layer."""
from __future__ import annotations
import mlx.core as mx
from .module import Module


class Embedding(Module):
    """A lookup table mapping integer indices to dense vectors.

    Useful for word embeddings, token embeddings, or any categorical input.

    Args:
        num_embeddings: Vocabulary size (number of distinct indices).
        embedding_dim: Dimension of each embedding vector.

    Input:  Integer array of shape ``(*)`` with values in ``[0, num_embeddings)``.
    Output: Float array of shape ``(*,  embedding_dim)``.

    Example::

        embed = Embedding(10000, 128)
        tokens = mx.array([[1, 42, 7]])   # (1, 3)
        out = embed(tokens)               # (1, 3, 128)
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        scale = (1.0 / embedding_dim) ** 0.5
        self.weight = mx.random.normal((num_embeddings, embedding_dim)) * scale

    def forward(self, x: mx.array) -> mx.array:
        return self.weight[x]
