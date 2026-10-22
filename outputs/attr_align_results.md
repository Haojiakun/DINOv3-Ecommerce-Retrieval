# Attribute-weighted alignment (refined, held-out test)

## Setup

- 5000 images; BLIP-VQA attributes, confidence-filtered (0.6) + dictionary-constrained.
- train/val/test split by item: 60/20/20. Test = 997 images / 223 unseen items.
- Weight warmup (w: 1 -> target over 4 epochs), grad clip max_norm=1.0.
- Local alignment over material/neckline/sleeve/pattern with per-dimension weights
  (material 1.0, others 0.5).

## Held-out test recall

| Method | Text->Image R@1/R@5/R@10 | Image->Image R@1/R@10 |
|---|---:|---:|
| CLIP native (attribute text) | 6.5 / 20.4 / 28.2 | - |
| DINOv3 CLS | - | 34.1 / 62.9 |
| Ours: InfoNCE (w=1) | 25.0 / 46.0 / 59.7 | 78.9 / 96.2 |
| Ours: + hard negatives (w=3) | 27.0 / 47.0 / 58.0 | 82.0 / 96.7 |
| Ours: + all-attribute local, tau=0.07 | 25.6 / 46.8 / 62.0 | 79.6 / 96.7 |
| Ours: + all-attribute local, tau=0.1 | 23.1 / 43.7 / 56.5 | 79.1 / 95.2 |

## Takeaways

- Attribute text -> image improves from 28.2% (CLIP) to 62.0% R@10 (+33.8).
- Hard negatives give the best R@1 (27.0) and best image->image R@1 (82.0).
- All-attribute local alignment gives the best R@10 (62.0) at tau=0.07.
- Raising tau to 0.1 hurt, so tau=0.07 is kept.

## Artifacts

- outputs/main_table.tex / main_table.md, tsne_compare.png, heatmap_sleeve.png.
- work/features/attr_emb_*.npy, attr_label_*.npy, attr_projs.pt,
  attr_align_visual_*.pt, attr_align_textproj_*.pt.
