# Task 2: visual-head validation

## Setup

- Frozen `vit_base_patch16_dinov3.lvd1689m`; only the lightweight head is trained.
- Metric learning: supervised contrastive loss over `item_key` (same product views
  pulled together, different products pushed apart).
- 6:2:2 product-level split on the 5,003-image balanced subset.
- Inputs are the cached `patch.npy` / `cls.npy` features, so no backbone forward pass.

## Test Recall@K (image -> image, same item)

| Head | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| CLS token (no training) | 85.1 | 93.4 | 95.6 |
| Mean pooling (no training) | 60.7 | 76.3 | 81.2 |
| AttentionPool only | 95.5 | 99.1 | 99.5 |
| CLS + AttentionPool fusion + MLP | 95.7 | 98.8 | 99.2 |

## Conclusion

- Attention pooling on its own is much stronger than mean pooling and even
  surpasses the CLS token (R@1 95.5 vs 85.1), confirming that learned patch
  aggregation captures fine-grained product cues the CLS token misses.
- Fusing CLS with the attention-pooled patch tokens gives a small additional
  gain on R@1 (95.5 → 95.7); both trained heads far exceed no-training baselines.
- The visual branch is viable before any text/attribute alignment.
