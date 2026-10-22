"""Visualize DINOv3 CLS features on DeepFashion In-shop images.

Generates three figures for the paper's motivation section:
  1. CLS cosine-similarity heatmap (block structure = same-item grouping).
  2. 2-D t-SNE / PCA scatter of CLS tokens, colored by item_ID.
  3. within-item vs cross-item cosine similarity bar chart.

Run from repo root:
    work\\venv\\Scripts\\python.exe outputs\\visualize_dinov3.py
"""

import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "work", "hf_home"))
for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_key, None)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import timm
from timm.data import create_transform, resolve_data_config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False


OUT_DIR = os.path.join(ROOT, "outputs", "figs")
os.makedirs(OUT_DIR, exist_ok=True)

N_IMAGES = 16  # first 16 rows of the parquet


def item_key(item_id):
    parts = item_id.split("_")
    return "_".join(parts[:-2]) if len(parts) >= 3 else item_id


def load_real_images(parquet_path, limit=N_IMAGES):
    import pyarrow.parquet as pq
    table = pq.ParquetFile(parquet_path).read_row_group(0).slice(0, limit)
    image_col = table.column("image")
    id_col = table.column("item_ID")
    images, ids = [], []
    for i in range(table.num_rows):
        img_struct = image_col[i].as_py()
        images.append(Image.open(io.BytesIO(img_struct["bytes"])).convert("RGB"))
        ids.append(id_col[i].as_py())
    return images, ids


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(
        "vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0
    )
    model = model.to(device).eval()

    data_cfg = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**data_cfg, is_training=False)

    parquet = os.path.join(ROOT, "work", "deepfashion_inshop.parquet")
    images, ids = load_real_images(parquet, limit=N_IMAGES)
    print(f"loaded {len(images)} images")

    batch = torch.stack([transform(img) for img in images]).to(device)
    with torch.inference_mode():
        feats = model.forward_features(batch)

    n_prefix = model.num_prefix_tokens
    cls = feats[:, 0]
    cls = F.normalize(cls, dim=-1).cpu().numpy()
    sim = cls @ cls.T  # (N, N) cosine similarity

    items = [item_key(i) for i in ids]
    unique_items = list(dict.fromkeys(items))  # preserve order
    n_items = len(unique_items)
    print(f"{n_items} unique items: {unique_items}")

    # ---- Reorder rows so same-item views are contiguous (for heatmap blocks) ----
    order = []
    for it in unique_items:
        order.extend([i for i, x in enumerate(items) if x == it])
    order = np.array(order)
    sim_ord = sim[np.ix_(order, order)]
    labels_ord = [ids[i] for i in order]
    short_labels = [l.split("_")[-1] for l in labels_ord]  # view index like "1_front"

    # ---- 1. Similarity heatmap ----
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_ord, cmap="YlOrRd", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(short_labels, fontsize=7)
    # draw block boundaries
    boundaries = np.cumsum([sum(1 for x in items if x == it) for it in unique_items])
    for b in boundaries[:-1]:
        ax.axhline(b - 0.5, color="black", lw=1.0)
        ax.axvline(b - 0.5, color="black", lw=1.0)
    plt.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title("DINOv3 CLS cosine similarity (rows grouped by item_ID)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cls_similarity_heatmap.png"), dpi=150)
    plt.close(fig)
    print("saved cls_similarity_heatmap.png")

    # ---- 2. 2-D embedding (t-SNE preferred, fallback PCA) ----
    if HAS_TSNE:
        emb = TSNE(n_components=2, perplexity=min(5, N_IMAGES - 1),
                   init="pca", random_state=42, metric="cosine").fit_transform(cls)
        method = "t-SNE"
    else:
        # PCA via SVD
        X = cls - cls.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        emb = X @ Vt[:2].T
        method = "PCA"

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 6))
    for idx, it in enumerate(unique_items):
        mask = np.array([x == it for x in items])
        ax.scatter(emb[mask, 0], emb[mask, 1], s=120, color=cmap(idx),
                   label=it.split("id_")[-1], edgecolors="black", linewidths=0.5)
        # label each point with its view index
        for i in np.where(mask)[0]:
            ax.annotate(ids[i].split("_")[-2] + ids[i].split("_")[-1][:1],
                        (emb[i, 0], emb[i, 1]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_title(f"DINOv3 CLS tokens — {method} (colored by item_ID)")
    ax.legend(title="item_ID", fontsize=8, title_fontsize=9, loc="best")
    ax.set_xlabel(f"{method} dim 1")
    ax.set_ylabel(f"{method} dim 2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"cls_{method.lower()}_scatter.png"), dpi=150)
    plt.close(fig)
    print(f"saved cls_{method.lower()}_scatter.png")

    # ---- 3. within vs cross similarity bar chart ----
    within_vals, cross_vals, names = [], [], []
    for it in unique_items:
        idxs = [i for i, x in enumerate(items) if x == it]
        w = [sim[a, b] for a in idxs for b in idxs if a != b]
        c = [sim[a, b] for a in idxs for b in range(len(ids)) if b not in idxs]
        within_vals.append(np.mean(w) if w else np.nan)
        cross_vals.append(np.mean(c) if c else np.nan)
        names.append(it.split("id_")[-1])

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w/2, within_vals, w, label="within-item", color="#4C72B0")
    bars2 = ax.bar(x + w/2, cross_vals, w, label="cross-item", color="#DD8452")
    for b in bars1:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{b.get_height():.3f}", ha="center", fontsize=8)
    for b in bars2:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{b.get_height():.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("mean cosine similarity")
    ax.set_title("Within-item vs Cross-item CLS similarity\n"
                 "(gap = fine-grained discrimination of frozen DINOv3)")
    ax.legend()
    ax.set_ylim(0.5, 0.85)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "within_vs_cross_bar.png"), dpi=150)
    plt.close(fig)
    print("saved within_vs_cross_bar.png")

    print(f"\nall figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
