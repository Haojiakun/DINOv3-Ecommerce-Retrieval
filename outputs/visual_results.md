# Task A: pure-visual attention pooling

## Setup

- `models.py`: `AttentionPooling` (Linear(768,256) + Tanh + Linear(256,1)) and
  `VisualHead` (concat CLS + pooled patch -> MLP -> 256-d).
- `train_visual.py`: supervised contrastive (InfoNCE) over `item_key`, frozen
  DINOv3, trains only the head on cached `patch.npy` / `cls.npy`.
- 6:2:2 product-level split on the 5,003-image balanced subset.

## Test Recall@K (image -> image)

| Head | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| CLS token (no training) | 85.1 | 93.4 | 95.6 |
| VisualHead (AttentionPooling + CLS fusion) | 95.7 | 98.8 | 99.2 |

R@1: 85.1% -> 95.7%; R@10: 95.6% -> 99.2%.

## Artifacts

- `work/models.py`, `work/train_visual.py`.
- `work/features_5k/visual_head.pt`.
