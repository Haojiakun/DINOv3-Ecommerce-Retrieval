"""Visualize AttentionPooling weights as 16x16 heatmaps over original images.

Shows that the learned attention focuses on discriminative clothing regions.
"""
import json
import io
import os

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import VisualHead

FEAT_DIR = os.path.join(os.path.dirname(__file__), "features")
PARQUET = os.path.join(os.path.dirname(__file__), "deepfashion_inshop.parquet")
CKPT = os.path.join(FEAT_DIR, "visual_head.pt")
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "attn_heatmap.png")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    with open(os.path.join(FEAT_DIR, "meta.json")) as f:
        meta = json.load(f)

    patch = np.load(os.path.join(FEAT_DIR, "patch.npy")).astype(np.float32)
    cls = np.load(os.path.join(FEAT_DIR, "cls.npy")).astype(np.float32)
    print(f"patch: {patch.shape}, cls: {cls.shape}")

    model = VisualHead().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()

    # pick a few images from different categories for variety
    item_ids = meta["item_id"]
    cats = meta["category2"]
    seen = set()
    picks = []
    for i in range(len(item_ids)):
        c = cats[i]
        if c not in seen:
            seen.add(c)
            picks.append(i)
        if len(picks) >= 6:
            break

    # get attention weights
    patch_t = torch.from_numpy(patch[picks]).to(DEVICE)
    with torch.no_grad():
        weights = model.attn_pool.attention(patch_t)  # (B, N, 1)
        weights = torch.softmax(weights, dim=1).cpu().numpy()[:, :, 0]  # (B, N)
    print(f"weights shape: {weights.shape}, range: [{weights.min():.4f}, {weights.max():.4f}]")

    # load original images from parquet
    table = pq.ParquetFile(PARQUET).read()
    images = table.column("image").to_pylist()

    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    for col, idx in enumerate(picks):
        # original image
        img_bytes = images[idx]
        if isinstance(img_bytes, dict):
            img_bytes = img_bytes["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((256, 256))

        # top row: original image
        axes[0, col].imshow(img)
        axes[0, col].set_title(cats[idx], fontsize=11)
        axes[0, col].axis("off")

        # bottom row: attention heatmap overlay
        attn = weights[col].reshape(16, 16)
        attn_resized = np.array(Image.fromarray((attn * 255).astype(np.uint8)).resize((256, 256), Image.BICUBIC)) / 255.0

        axes[1, col].imshow(img)
        axes[1, col].imshow(attn_resized, cmap="jet", alpha=0.5, vmin=attn.min(), vmax=attn.max())
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Image", fontsize=12, rotation=0, labelpad=40)
    axes[1, 0].set_ylabel("Attention", fontsize=12, rotation=0, labelpad=40)

    fig.suptitle("AttentionPooling Weights Mapped to 16x16 Patch Grid", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
