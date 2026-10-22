# Flickr30K domain-boundary analysis

## Diagnosis (Flickr30K 1K test, text->image)

| Setting | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| CLIP native | 71.1 | 91.6 | 96.8 |
| Ours zero-shot (trained AttentionPooling) | 0.3 | 0.9 | 1.7 |
| Ours zero-shot (patch mean-pool, same MLP) | 0.3 | 1.0 | 1.8 |
| Ours adapted projections (probe, 700/300 split) | 56.3 | 83.0 | 90.7 |

Findings:
- Mean-pooling does not recover (0.3% vs 0.3%), so the collapse is NOT caused by
  attention specialization.
- It is caused by domain-specific projection layers: a few hundred steps of
  projection adaptation on Flickr30K recovers to 56.3% R@1.
- The framework is domain-specialized by design; general-domain transfer requires
  adapting the text/visual projections.

## Paper framing (Domain Boundary Analysis, separate section, not in Table 1)

State that the method is designed for fine-grained e-commerce retrieval, report
the zero-shot transfer collapse honestly, localize the cause to the projection
layers, and give the adapted-projection upper bound. Do not chase Flickr30K SOTA.

## Pure visual baselines (DeepFashion test, image->image)

| Method | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| DINOv3 patch mean-pool | 10.7 | 22.9 | 29.6 |
| DINOv3 patch GeM (p=3) | 13.0 | 27.5 | 37.2 |
| SCDA (simplified, ResNet-50) | 28.4 | 48.4 | 58.7 |
| DINOv3 CLS | 34.1 | 54.5 | 62.9 |
| Ours (attribute-guided local alignment) | 79.6 | - | 96.7 |

Pure visual aggregation (mean-pool, GeM, SCDA) is far below attribute-guided
alignment, supporting the core claim. Simplified SCDA (no PCA whitening) is used;
the full SCDA with whitening is left as future work.

## Scripts

- work/flickr30k_diag.py: zero-shot V1/V2 + CLIP native.
- work/flickr30k_finetune_probe.py: adapted-projection upper bound.
- work/visual_baselines.py: mean-pool / GeM / CLS on DeepFashion test.
