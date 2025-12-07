"""Cross-modal hashing model built with PyTorch."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from transformers import AutoModel


@dataclass
class HashingOutputs:
    image_codes: torch.Tensor
    text_codes: torch.Tensor
    image_logits: torch.Tensor
    text_logits: torch.Tensor


class ImageEncoder(nn.Module):
    def __init__(
        self,
        bits: int,
        vision_model: str = "google/vit-base-patch16-224-in21k",
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.vision_model = AutoModel.from_pretrained(vision_model, cache_dir=cache_dir)
        hidden_size = self.vision_model.config.hidden_size
        self.proj = nn.Linear(hidden_size, bits)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.proj(pooled)

        return logits


class TextEncoder(nn.Module):
    def __init__(
        self,
        bits: int,
        model_name: str = "distilbert-base-uncased",
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        hidden_size = self.transformer.config.hidden_size
        self.proj = nn.Linear(hidden_size, bits)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token representation
        logits = self.proj(pooled)
        return logits


class HashingModel(nn.Module):
    """Encodes images and text into continuous hash logits and relaxed hash codes."""

    def __init__(
        self,
        bits: int = 32,
        text_model: str = "distilbert-base-uncased",
        image_model: str = "google/vit-base-patch16-224-in21k",
        image_cache_dir: str | Path | None = None,
        text_cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(
            bits=bits, vision_model=image_model, cache_dir=image_cache_dir
        )
        self.text_encoder = TextEncoder(
            bits=bits, model_name=text_model, cache_dir=text_cache_dir
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> HashingOutputs:
        image_logits = self.image_encoder(pixel_values=pixel_values)
        text_logits = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Continuous relaxation of sign to enable backpropagation
        image_codes = torch.tanh(image_logits)
        text_codes = torch.tanh(text_logits)

        return HashingOutputs(
            image_codes=image_codes,
            text_codes=text_codes,
            image_logits=image_logits,
            text_logits=text_logits,
        )

    @torch.inference_mode()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return torch.sign(torch.tanh(self.image_encoder(pixel_values)))

    @torch.inference_mode()
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return torch.sign(
            torch.tanh(self.text_encoder(input_ids=input_ids, attention_mask=attention_mask))
        )
