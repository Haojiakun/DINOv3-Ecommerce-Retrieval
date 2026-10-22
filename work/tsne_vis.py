import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def main():
    feat_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features")
    out_dir = os.path.join(os.path.dirname(feat_dir), "..", "outputs")
    out_dir = os.path.abspath(out_dir)

    cls = np.load(os.path.join(feat_dir, "cls.npy")).astype(np.float32)
    with open(os.path.join(feat_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    n = 2500
    cls = cls[:n]
    cat2 = meta["category2"][:n]
    item_key = meta["item_key"][:n]

    x = StandardScaler().fit_transform(cls)
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0, max_iter=1000)
    xy = tsne.fit_transform(x)

    # Figure 1a: color by clothing category (category2).
    cats = sorted(set(c for c in cat2 if c), key=lambda c: -cat2.count(c))[:12]
    colors = plt.get_cmap("tab20").colors
    cmap = {c: colors[i % len(colors)] for i, c in enumerate(cats)}

    plt.figure(figsize=(10, 8))
    for c in cats:
        mask = [cc == c for cc in cat2]
        plt.scatter(xy[mask, 0], xy[mask, 1], s=4, alpha=0.55, color=cmap[c], label=c)
    plt.title("DINOv3 CLS t-SNE (colored by clothing category)")
    plt.legend(markerscale=4, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tsne_cls_category2.png"), dpi=150)
    plt.close()

    # Figure 1b: highlight the 8 most frequent items, everything else gray.
    top_items = sorted(set(item_key), key=lambda k: -item_key.count(k))[:8]
    plt.figure(figsize=(10, 8))
    others = [k not in top_items for k in item_key]
    plt.scatter(xy[others, 0], xy[others, 1], s=3, alpha=0.25, color="gray", label="others")
    for i, it in enumerate(top_items):
        mask = [k == it for k in item_key]
        plt.scatter(xy[mask, 0], xy[mask, 1], s=8, alpha=0.8, color=colors[i], label=it)
    plt.title("DINOv3 CLS t-SNE (8 frequent items highlighted)")
    plt.legend(markerscale=4, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tsne_cls_items.png"), dpi=150)
    plt.close()

    print("saved:", os.path.join(out_dir, "tsne_cls_category2.png"))
    print("saved:", os.path.join(out_dir, "tsne_cls_items.png"))


if __name__ == "__main__":
    main()
