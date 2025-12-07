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
  --img-root /data2/zhangchaoke/PythonProject/MyCMH/datasets/train2014 \
  --ann-file /data2/zhangchaoke/PythonProject/MyCMH/datasets/annotations/captions_train2014.json \
  --bits 32 \
  --batch-size 32 \
  --epochs 5 \
  --lr 1e-4 \
  --device cuda
```

Outputs are saved under `checkpoints/`. The command falls back to CPU automatically if CUDA is unavailable.

## Extending to other datasets

Implement a subclass of `BaseRetrievalDataset` in `src/datasets` that returns the same keys as the COCO version (`image`, `input_ids`, `attention_mask`, `label`) and register it inside `build_dataset`.

## Model overview

* **Image encoder**: ResNet-50 backbone followed by a projection to hash logits.
* **Text encoder**: Transformer encoder (default DistilBERT) projected to hash logits.
* **Hashing**: `tanh` relaxation during training; `sign` for inference-ready binary codes.
* **Loss**: Binary cross-entropy over the pairwise similarity matrix with a quantization regularizer pushing codes toward ±1.

