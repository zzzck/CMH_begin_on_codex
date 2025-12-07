"""Base dataset interface for cross-modal hashing retrieval."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from torch.utils.data import Dataset


class BaseRetrievalDataset(Dataset, ABC):
    """Abstract base class for retrieval datasets.

    Each sample should include an image tensor, tokenized text inputs, and a
    label used to determine positive pairs for hashing supervision. The label is
    typically an integer representing the underlying image id so that captions
    referring to the same image share the label.
    """

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:

        """Return a dictionary with keys: pixel_values, input_ids, attention_mask, label."""

        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:  # pragma: no cover - dataset length is trivial
        raise NotImplementedError
