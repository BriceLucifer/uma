from __future__ import annotations
from typing import Iterator, cast
import mlx.core as mx
from .dataset import Dataset

class DataLoader:
    """Batches a :class:`Dataset` and optionally shuffles each epoch.

    Args:
        dataset: Any object implementing the ``Dataset`` interface.
        batch_size: Number of samples per batch. Default ``32``.
        shuffle: If ``True``, indices are shuffled at the start of each
            iteration. Default ``True``.

    Example:
        ```python
        loader = DataLoader(MyDataset(x, y), batch_size=256, shuffle=True)
        for x_batch, y_batch in loader:
            ...
        ```
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        """Number of batches per epoch (ceiling division)."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    from typing import Iterator, cast

    def __iter__(self) -> Iterator[tuple[mx.array, mx.array]]:
        n = len(self.dataset)
        indices: list[int] = (
            cast(list[int], mx.random.permutation(n).tolist()) if self.shuffle
            else list(range(n))
        )

        for i in range(0, n, self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            batch = [self.dataset[int(j)] for j in batch_idx]
            xs = mx.stack([item[0] for item in batch])
            ys = mx.stack([item[1] for item in batch])
            yield xs, ys