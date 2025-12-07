"""COCO captions dataset wrapper for cross-modal hashing."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CocoCaptions
from transformers import AutoTokenizer

from .base import BaseRetrievalDataset


class CocoHashingDataset(BaseRetrievalDataset):
    """COCO dataset returning image tensors and tokenized captions.

    Args:
        img_root: Directory containing COCO images (train2017/val2017).
        ann_file: Path to COCO captions annotation file (captions_train2017.json).
        tokenizer_name: Name of a HuggingFace tokenizer to encode captions.
        max_length: Maximum number of tokens for captions.
        random_caption: If True, pick a random caption per image; otherwise use the first.
        image_size: Square resize for images.
    """

    def __init__(
        self,
        img_root: str | Path,
        ann_file: str | Path,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 32,
        random_caption: bool = True,
        image_size: int = 224,
    ) -> None:
        self.img_root = Path(img_root)
        self.ann_file = Path(ann_file)
        self.random_caption = random_caption

        self.dataset: Dataset = CocoCaptions(
            root=str(self.img_root),
            annFile=str(self.ann_file),
            transform=self._image_transform(image_size),
            target_transform=None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    @staticmethod
    def _image_transform(image_size: int) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _pick_caption(self, captions: list[str]) -> str:
        if self.random_caption:
            return random.choice(captions)
        return captions[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image, captions = self.dataset[index]
        caption = self._pick_caption(captions)
        tokenized = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # CocoCaptions returns annotation ids as index; map to image id via dataset.ids
        image_id = self.dataset.ids[index]
        sample = {
            "image": image,
            "input_ids": tokenized.input_ids.squeeze(0),
            "attention_mask": tokenized.attention_mask.squeeze(0),
            "label": torch.tensor(image_id, dtype=torch.long),
        }
        return sample

    def __len__(self) -> int:
        return len(self.dataset)


def build_dataset(name: str, **kwargs: Optional[Dict]) -> BaseRetrievalDataset:
    """Factory to create datasets by name.

    Args:
        name: Dataset identifier (e.g., "coco").
        **kwargs: Parameters forwarded to dataset constructors.
    """

    name = name.lower()
    if name == "coco":
        required_args = ["img_root", "ann_file"]
        for arg in required_args:
            if arg not in kwargs:
                raise ValueError(f"Missing required argument '{arg}' for COCO dataset")
        return CocoHashingDataset(**kwargs)

    raise ValueError(f"Unsupported dataset '{name}'. Extend build_dataset to add more.")
