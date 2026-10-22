# Table 9: Cross-dataset validation of the visual head (image-to-image retrieval)

| Dataset | Method | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|
| DeepFashion | DINOv3 CLS | 85.1 | 93.4 | 95.6 |
| DeepFashion | VisualHead (Ours) | 92.9 | 98.1 | 98.8 |
| Fashion-Gen | DINOv3 CLS | 44.6 | 59.6 | 64.8 |
| Fashion-Gen | VisualHead (Ours, 3 seeds) | 83.8 ± 0.3 | 93.6 ± 0.4 | 95.8 ± 0.1 |

Note: Both datasets train the pure-visual head (Attention Pooling + CLS fusion)
with supervised contrastive learning on a held-out 20% item split. DeepFashion
reports the single-run result; Fashion-Gen reports the mean ± std over 3 seeds.
The DeepFashion R@1 (92.9) is the cross-modal InfoNCE-trained head (70 epochs).
Fashion-Gen numbers are unchanged from the prior run.
