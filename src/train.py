"""Training loop for cross-modal hashing on COCO or other datasets."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.coco import build_dataset
from src.models.hash_net import HashingModel
from src.models.losses import cross_modal_hashing_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-modal hashing training")
    parser.add_argument("--dataset", type=str, default="coco", help="Dataset name (default: coco)")
    parser.add_argument("--img-root", type=str, required=True, help="Path to images root")
    parser.add_argument("--ann-file", type=str, required=True, help="Path to annotations json")
    parser.add_argument("--bits", type=int, default=32, help="Number of hash bits")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=32, help="Max caption tokens")
    parser.add_argument("--image-size", type=int, default=224, help="Image resolution")
    parser.add_argument("--device", type=str, default="cuda", help="Training device")
    parser.add_argument(
        "--text-model",
        type=str,
        default="distilbert-base-uncased",
        help="HuggingFace model name for text encoder",
    )
    return parser.parse_args()


def create_dataloader(args: argparse.Namespace) -> DataLoader:
    dataset = build_dataset(
        name=args.dataset,
        img_root=args.img_root,
        ann_file=args.ann_file,
        tokenizer_name=args.text_model,
        max_length=args.max_length,
        image_size=args.image_size,
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)


def train_one_epoch(
    model: HashingModel, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device
) -> float:
    model.train()
    epoch_loss = 0.0
    for batch in tqdm(loader, desc="train"):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images=images, input_ids=input_ids, attention_mask=attention_mask)
        loss = cross_modal_hashing_loss(outputs.image_codes, outputs.text_codes, labels, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = HashingModel(bits=args.bits, text_model=args.text_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loader = create_dataloader(args)

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, loader, optimizer, device)
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {loss:.4f}")

    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    ckpt_path = save_dir / f"hashing_model_{args.dataset}_{args.bits}bits.pt"
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
