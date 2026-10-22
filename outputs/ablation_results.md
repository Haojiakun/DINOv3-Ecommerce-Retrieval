# Ablations: backbone generalization and dirty-data filtering

## Backbone ablation (plain InfoNCE and +local, fresh head, same held-out test, 3 seeds)

| Backbone | Config | Text->Image R@1/R@5/R@10 | Image->Image R@1/R@10 |
|---|---|---:|---:|
| DINOv3 (vit_base_patch16_dinov3) | plain | 26.5 / 44.3 / 57.3 | 94.4 / 99.0 |
| DINOv3 | + local | 25.4 / 41.4 / 50.8 | 94.4 / 99.0 |
| CLIP-ViT (vit_base_patch16_clip_224.openai) | plain | 19.3 / 42.2 / 60.5 | 91.9 / 98.4 |
| CLIP-ViT | + local | 20.3 / 42.2 / 60.2 | 92.1 / 98.4 |
| SigLIP (vit_base_patch16_siglip_224.webli) | plain | 16.7 / 43.5 / 52.1 | 87.2 / 97.6 |
| SigLIP | + local | 17.4 / 37.3 / 57.1 | 87.3 / 97.6 |
| DINOv2 (vit_base_patch14_dinov2.lvd142m) | plain | 6.6 / 8.6 / 10.4 | 15.6 / 17.0 |
| DINOv2 | + local | OOM | OOM |

DINOv3 achieves the best overall retrieval. CLIP-ViT and SigLIP are competitive on
text-to-image but weaker on image-to-image. DINOv2 (14x14 patches) underperforms
markedly; its local variant ran out of GPU memory due to the larger patch grid.

## Dirty vs filtered attributes (w=3 config)

| Config | Text->Image R@1/R@10 | Image->Image R@1/R@10 |
|---|---:|---:|
| filtered w=3 (hard neg only) | 10.8 / 51.4 | 94.0 / 98.3 |
| dirty w=3 (no filtering) | 29.8 / 53.7 | 93.1 / 98.5 |
| filtered w=3 + all local | 27.9 / 50.1 | 91.7 / 98.6 |
| dirty w=3 + all local | 29.6 / 52.6 | 93.3 / 99.0 |

On the 5003 balanced subset, dirty attributes yield a higher t2i R@1 for the
w=3 hard-negative config (29.8 vs 10.8), while the all-local config is less
sensitive to filtering (29.6 vs 27.9). Image-to-image retrieval is stable
across both conditions.

## Artifacts

- work/extract_backbone.py, work/run_backbone_global.py, work/dirty_ablation.py.
- work/features_5k/cls_{clip,siglip,dinov2}.npy, patch_{clip,siglip,dinov2}.npy,
  visual_head_{clip,siglip,dinov2}.pt, dirty_emb_*.npy, dirty_label_*.npy, dirty_attrs.json.
