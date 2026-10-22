# Task 1: weak-baseline results

## Setup

- Data: `Marqo/deepfashion-inshop`, first 5000 images.
- Ground truth: same `item_key` (product), leave-one-out retrieval.
- Feature backbones:
  - DINOv3: `vit_base_patch16_dinov3.lvd1689m`, CLS token.
  - CLIP: `openai/clip-vit-base-patch32`.

## Retrieval Recall@K

| Method | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| DINOv3 CLS, image -> image | 0.1538 | 0.3479 | 0.4316 |
| DINOv3 CLS + CLIP text (naive concat), image -> image | 0.4073 | 0.7461 | 0.8546 |
| CLIP native, image -> image | 0.2585 | 0.4947 | 0.5953 |
| CLIP native, text -> image | 0.0414 | 0.1270 | 0.2124 |

## Notes

- The DINOv3 CLS baseline is weak (R@10 43%), which makes the "fine-grained
  e-commerce retrieval is hard for a frozen DINOv3 CLS token" pain point concrete.
- The naive concat baseline looks strong only because the captions are item-specific
  (identity leakage). It is not usable for queries without captions and is exactly the
  "stitched-together" baseline to argue against.
- CLIP text -> image is low (R@10 21%), confirming the cross-modal semantic gap.

## Figures

- `tsne_cls_category2.png`: DINOv3 CLS colored by clothing category.
- `tsne_cls_items.png`: DINOv3 CLS with 8 frequent products highlighted.
