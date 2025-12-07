"""Losses for cross-modal hashing."""
from __future__ import annotations

import torch
from torch import nn


def compute_similarity_matrix(labels_a: torch.Tensor, labels_b: torch.Tensor) -> torch.Tensor:
    """Compute binary similarity matrix where entries are 1 if labels match."""
    return (labels_a.unsqueeze(1) == labels_b.unsqueeze(0)).float()


def cross_modal_hashing_loss(
    image_codes: torch.Tensor,
    text_codes: torch.Tensor,
    image_labels: torch.Tensor,
    text_labels: torch.Tensor,
    quantization_weight: float = 0.1,
) -> torch.Tensor:
    """Binary cross-entropy loss encouraging matched pairs to have similar codes.

    Args:
        image_codes: Relaxed image hash codes in [-1, 1].
        text_codes: Relaxed text hash codes in [-1, 1].
        image_labels: Labels for images (typically image ids).
        text_labels: Labels for text (match image ids of their captions).
        quantization_weight: Strength for pushing codes toward binary values.
    """

    similarity_logits = image_codes @ text_codes.t() / image_codes.size(1)
    targets = compute_similarity_matrix(image_labels, text_labels)
    bce = nn.functional.binary_cross_entropy_with_logits(similarity_logits, targets)

    quantization_loss = ((image_codes.abs() - 1).pow(2).mean() + (text_codes.abs() - 1).pow(2).mean()) / 2

    return bce + quantization_weight * quantization_loss
