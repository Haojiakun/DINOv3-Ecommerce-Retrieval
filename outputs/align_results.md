# Task 3: text / attribute alignment

## Setup

- Visual: frozen DINOv3 -> `FusionHead` (CLS + AttentionPool + MLP), warm-started
  from Task 2 weights.
- Text: frozen CLIP text encoder (`openai/clip-vit-base-patch32`) -> trainable
  `Linear(512, 768)` projections for captions and for attribute strings.
- Loss: symmetric InfoNCE for caption and attribute, plus a hard-negative margin
  (same category2, different product; prefers different color).
- Item-level 80/20 split, evaluated on the held-out 997-image val split.

## Val recall (same val split, fair comparison)

| Method | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| CLIP native image -> image | 0.4712 | 0.7195 | 0.8032 |
| CLIP native text -> image | 0.0100 | 0.0201 | 0.0371 |
| Ours visual image -> image | 0.8163 | 0.9576 | 0.9798 |
| Ours caption text -> image | 0.4363 | 0.6881 | 0.7964 |
| Ours attribute text -> image | 0.2768 | 0.5537 | 0.6871 |

## Conclusion

- Before alignment, CLIP text -> image is near chance (R@10 3.7%).
- After InfoNCE alignment + hard negatives, caption text -> image reaches
  R@10 79.6% and attribute text -> image reaches R@10 68.7%.
- Visual image -> image also improves to R@1 81.6%, showing the cross-modal
  supervision helps the visual head too.

## Artifacts

- `work/features/attr_text.npy`: CLIP text features for attribute strings.
- `work/features/align_visual.pt`, `align_text_proj.pt`, `align_attr_proj.pt`.
- Scripts: `work/extract_attr_text.py`, `work/train_align.py`.
