# Table 1: cross-backbone comparison (held-out test, 997 images / 223 unseen items)

| Method | Visual | Text | Text->Image R@1/R@10 | Image->Image R@1/R@10 |
|---|---|---|---:|---:|
| CLIP Zero-shot | CLIP-ViT | CLIP | 6.5 / 28.2 | - |
| CLIP + InfoNCE | CLIP-ViT | CLIP | 19.9 / 60.0 | 67.8 / 92.4 |
| DINOv2 + InfoNCE | DINOv2 | CLIP | 21.8 / 63.4 | 66.0 / 94.0 |
| SigLIP + InfoNCE | SigLIP | CLIP | 24.6 / 64.9 | 80.6 / 95.0 |
| DINOv3 + InfoNCE | DINOv3 | CLIP | 24.5 / 64.0 | 72.4 / 95.3 |
| DINOv3 + Ours (hard + all-attribute local) | DINOv3 | CLIP | 25.6 / 62.0 | 79.6 / 96.7 |

Notes:
- All "+ InfoNCE" rows use the same frozen CLIP text encoder, a fresh VisualHead
  and a plain symmetric InfoNCE loss (w=1), so the comparison isolates the visual
  backbone.
- The final "Ours" row additionally uses hard-negative weighting (w=3) and
  all-attribute local patch alignment. Its hard-only variant (w=3, no local)
  reaches text->image R@1 27.0 and image->image R@1 82.0.
- DINOv2 is patch14@518 (1369 patch tokens); others are patch16@224/256.
