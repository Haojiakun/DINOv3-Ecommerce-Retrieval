# Table 8: Comparison with pure visual baselines (Image->Image on 5003 test set)

All methods evaluated on the 5,003-image balanced subset (6:2:2 product split,
1057 test images / 144 unseen items). No-training rows use frozen DINOv3 features.

| Method | R@1 | R@10 |
|---|---:|---:|
| DINOv3 Patch Mean-pooling | 60.7 | 81.2 |
| DINOv3 Patch GeM (p = 3) | 66.0 | 85.8 |
| SCDA (Simplified, ResNet-50) | 64.4 | 85.1 |
| DINOv3 CLS | 85.1 | 95.6 |
| Ours (Attribute-Guided) | 92.9 | 98.8 |

Note: Ours uses the cross-modal InfoNCE-trained visual head (70 epochs, 3 seeds:
R@1=92.9±0.6, R@10=98.8±0.5). SCDA descriptors are extracted from a ResNet-50
pretrained on ImageNet with the simplified SCDA aggregation (channel-wise mean
thresholding).
