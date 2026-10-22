# Main experiment table (held-out test: 1057 images / 144 unseen items, 5003 balanced subset)

| Method | Text->Image R@1/R@5/R@10 | Image->Image R@1/R@10 |
|---|---:|---:|
| CLIP native (attribute text) | 26.3 / 48.6 / 57.2 | 83.9 / 96.1 |
| DINOv3 CLS | - | 85.1 / 95.6 |
| GCA-Net (reimpl, 3 seeds) | 18.9 / 31.4 / 40.7 | 87.7 / 97.3 |
| Ours: InfoNCE (70 ep) | 25.7 / 42.0 / 53.0 | 92.9 / 98.8 |
| Ours: + hard negatives (w=3) | 10.8 / 27.2 / 51.4 | 94.0 / 98.3 |
| Ours: + all-attribute local (w=3) | 27.9 / 44.4 / 50.1 | 91.7 / 98.6 |
