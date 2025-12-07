# Cross-Modal Hashing Prototype (COCO)

This project provides a small PyTorch pipeline for cross-modal hashing between images and text. It targets COCO captions but exposes a dataset factory so new datasets can be plugged in easily.

## Setup

```bash
pip install -r requirements.txt
```

## Training on COCO

Download COCO images and captions annotations, then run:

```bash
python -m src.train \
  --dataset coco \
  --img-root /path/to/train2017 \
  --ann-file /path/to/annotations/captions_train2017.json \
  --bits 32 \
  --batch-size 32 \
  --epochs 5 \
  --lr 1e-4 \
  --device cuda
```

Outputs are saved under `checkpoints/`. The command falls back to CPU automatically if CUDA is unavailable.

## Extending to other datasets


Implement a subclass of `BaseRetrievalDataset` in `src/datasets` that returns the same keys as the COCO version (`pixel_values`, `input_ids`, `attention_mask`, `label`) and register it inside `build_dataset`.

## Model overview

* **Image encoder**: Vision Transformer backbone from Hugging Face followed by a projection to hash logits.

* **Text encoder**: Transformer encoder (default DistilBERT) projected to hash logits.
* **Hashing**: `tanh` relaxation during training; `sign` for inference-ready binary codes.
* **Loss**: Binary cross-entropy over the pairwise similarity matrix with a quantization regularizer pushing codes toward ±1.

## Model checkpoints and caching

Image and text weights are cached locally after the first download. By default, Hugging Face uses its standard
cache locations, but you can override them when constructing the model:

```python
from src.models.hash_net import HashingModel

model = HashingModel(
    bits=32,
    image_cache_dir="checkpoints/hf_vision",
    text_cache_dir="checkpoints/hf_text",
)
```

This prevents repeated downloads across runs and keeps checkpoints within the project directory if desired.

