from __future__ import annotations
from typing import Any
import mlx.core as mx


class Dataset:
    """Abstract base class for all datasets.

    Subclasses must implement ``__len__`` and ``__getitem__``.

    Example:
        ```python
        class MyDataset(Dataset):
            def __init__(self, x, y):
                self.x, self.y = x, y

            def __len__(self):
                return self.x.shape[0]

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]
        ```
    """

    def __len__(self) -> int:
        """Return the total number of samples."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Any:
        """Return the sample at position ``idx``."""
        raise NotImplementedError