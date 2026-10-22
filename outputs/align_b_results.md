# Task B: attribute-guided cross-modal alignment

## Setup

- Text: parse `item_ID` -> "men's / women's", join `category1/2/3` and `color`
  into a natural-language attribute sentence; encode with the frozen CLIP text
  encoder -> `work/features/text_feats.npy` (512-d).
- Models: `VisualHead` (warm-started from `visual_head.pt`) + `TextProjection`
  (512 -> 256), both in `work/models.py`.
- Loss: symmetric InfoNCE. The hard-negative variant adds extra weight to
  same-`category2` / different-item negatives, with an extra penalty for
  same-`category2` + same-`color` negatives.

## Val results (997 images)

| Method | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| CLIP native, attribute text -> image | 0.0652 | 0.2036 | 0.2818 |
| Ours, plain InfoNCE, text -> image | 0.2708 | 0.5165 | 0.6259 |
| Ours, hard-negative InfoNCE, text -> image | 0.2899 | 0.5507 | 0.6600 |
| Ours, plain InfoNCE, image -> image | 0.7992 | 0.9284 | 0.9617 |
| Ours, hard-negative InfoNCE, image -> image | 0.8042 | 0.9465 | 0.9697 |

## Conclusion

- Attribute-level text -> image goes from 28.2% (CLIP native) to 66.0% R@10
  after alignment, confirming the text bridge closes a large part of the
  semantic gap.
- Hard negatives give a consistent, modest gain: text -> image R@10 62.6% -> 66.0%
  and image -> image R@1 79.9% -> 80.4%.

## Artifacts

- `work/build_text.py`: text construction + CLIP text encoding.
- `work/train_align.py`: plain vs hard-negative alignment training.
- `work/features/text_feats.npy`, `texts.json`.
- `work/features/align_visual_plain.pt`, `align_visual_hard.pt`,
  `align_text_proj_plain.pt`, `align_text_proj_hard.pt`.
