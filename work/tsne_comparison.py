"""Side-by-side t-SNE: CLS baseline vs VisualHead features.

Left:  frozen DINOv3 CLS token features (scattered)
Right: VisualHead output after contrastive training (clustered)
"""
import json
import os

import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import VisualHead

FEAT_DIR = os.path.join(os.path.dirname(__file__), "features")
CKPT = os.path.join(FEAT_DIR, "visual_head.pt")
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "tsne_comparison.png")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    with open(os.path.join(FEAT_DIR, "meta.json")) as f:
        meta = json.load(f)

    cls = np.load(os.path.join(FEAT_DIR, "cls.npy")).astype(np.float32)
    patch = np.load(os.path.join(FEAT_DIR, "patch.npy")).astype(np.float32)
    print(f"cls: {cls.shape}, patch: {patch.shape}")

    # sample 2500 images for speed
    rng = np.random.RandomState(42)
    idx = rng.choice(len(cls), size=2500, replace=False)
    cls_s = cls[idx]
    patch_s = patch[idx]
    cats = np.array(meta["category2"])[idx]

    # VisualHead features
    model = VisualHead().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        cls_t = torch.from_numpy(cls_s).to(DEVICE)
        patch_t = torch.from_numpy(patch_s).to(DEVICE)
        vh = model(cls_t, patch_t).cpu().numpy()

    # normalize
    cls_n = cls_s / (np.linalg.norm(cls_s, axis=1, keepdims=True) + 1e-8)
    vh_n = vh / (np.linalg.norm(vh, axis=1, keepdims=True) + 1e-8)

    print("running t-SNE on CLS...")
    tsne_cls = TSNE(n_components=2, perplexity=50, random_state=42, init="pca").fit_transform(cls_n)

    print("running t-SNE on VisualHead...")
    tsne_vh = TSNE(n_components=2, perplexity=50, random_state=42, init="pca").fit_transform(vh_n)

    # color by category_2
    unique_cats = sorted(set(cats))
    cmap = plt.get_cmap("tab20")
    cat_to_color = {c: cmap(i % 20) for i, c in enumerate(unique_cats)}
    colors = [cat_to_color[c] for c in cats]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].scatter(tsne_cls[:, 0], tsne_cls[:, 1], c=colors, s=8, alpha=0.7)
    axes[0].set_title("(a) Baseline: DINOv3 CLS Token", fontsize=13)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].scatter(tsne_vh[:, 0], tsne_vh[:, 1], c=colors, s=8, alpha=0.7)
    axes[1].set_title("(b) Ours: VisualHead (CLS + AttentionPooling)", fontsize=13)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # legend
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cat_to_color[c], markersize=8, label=c)
               for c in unique_cats]
    fig.legend(handles=handles, loc="lower center", ncol=len(unique_cats), fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
