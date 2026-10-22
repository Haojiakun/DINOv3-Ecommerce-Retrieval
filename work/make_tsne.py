import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

from models import VisualHead


FEAT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features")


def embed(model, patch, cls, device="cuda"):
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(patch), 128):
            p = torch.from_numpy(patch[i : i + 128]).to(device).float()
            c = torch.from_numpy(cls[i : i + 128]).to(device).float()
            out.append(F.normalize(model(c, p), dim=1).cpu())
    return torch.cat(out).numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(os.path.join(FEAT_DIR, "meta.json"), encoding="utf-8"))
    keys = np.asarray(meta["item_key"])
    cat2 = np.asarray(meta["category2"])
    uniques, labels = np.unique(keys, return_inverse=True)
    rng = np.random.RandomState(0)
    idx = np.arange(len(uniques))
    rng.shuffle(idx)
    n_train = int(len(idx) * 0.6)
    n_val = int(len(idx) * 0.2)
    test_keys = set(uniques[idx[n_train + n_val:]].tolist())
    test_idx = np.where(np.asarray([k in test_keys for k in keys]))[0]

    patch = np.load(os.path.join(FEAT_DIR, "patch.npy"), mmap_mode="r").astype(np.float32)[test_idx]
    cls = np.load(os.path.join(FEAT_DIR, "cls.npy"), mmap_mode="r").astype(np.float32)[test_idx]
    clip_img = F.normalize(torch.from_numpy(
        np.load(os.path.join(FEAT_DIR, "clip_img.npy")).astype(np.float32)[test_idx]
    ), dim=1).numpy()

    v_w1_model = VisualHead()
    v_w1_model.load_state_dict(torch.load(os.path.join(FEAT_DIR, "attr_align_visual_w1.pt"), map_location="cpu", weights_only=True))
    v_all_model = VisualHead()
    v_all_model.load_state_dict(torch.load(os.path.join(FEAT_DIR, "attr_align_visual_w3+all_t07.pt"), map_location="cpu", weights_only=True))
    v_w1 = embed(v_w1_model.to(device), patch, cls)
    v_all = embed(v_all_model.to(device), patch, cls)

    cats = sorted(set(c for c in cat2[test_idx] if c), key=lambda c: -list(cat2[test_idx]).count(c))[:10]
    cmap = {c: plt.get_cmap("tab10").colors[i] for i, c in enumerate(cats)}

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax, (name, data) in zip(axes, [("CLIP native", clip_img), ("Ours w=1", v_w1), ("Ours w=3 + all local", v_all)]):
        xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(data)
        for c in cats:
            m = np.asarray([cc == c for cc in cat2[test_idx]])
            ax.scatter(xy[m, 0], xy[m, 1], s=4, alpha=0.55, color=cmap[c], label=c)
        ax.set_title(name)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=10, markerscale=3)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = os.path.join(os.path.dirname(os.path.dirname(FEAT_DIR)), "outputs", "tsne_compare.png")
    fig.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
