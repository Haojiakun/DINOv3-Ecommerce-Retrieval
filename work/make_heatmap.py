import io
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from models import TextProjection, VisualHead
from train_attr_align import ATTR_DIMS, ATTR_LAMBDA, attr_infonce, hard_infonce, local_pool, make_codes


FEAT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patch = np.load(os.path.join(FEAT_DIR, "patch.npy"), mmap_mode="r").astype(np.float32)
    cls = np.load(os.path.join(FEAT_DIR, "cls.npy"), mmap_mode="r").astype(np.float32)
    text = np.load(os.path.join(FEAT_DIR, "text_feats.npy")).astype(np.float32)
    attr_emb = {d: np.load(os.path.join(FEAT_DIR, f"attr_emb_{d}.npy")).astype(np.float32) for d in ATTR_DIMS}
    attr_label = {d: np.load(os.path.join(FEAT_DIR, f"attr_label_{d}.npy"), allow_pickle=True) for d in ATTR_DIMS}
    attr_codes = {d: make_codes(attr_label[d]) for d in ATTR_DIMS}
    meta = json.load(open(os.path.join(FEAT_DIR, "meta.json"), encoding="utf-8"))
    attrs = json.load(open(os.path.join(FEAT_DIR, "filtered_attrs.json"), encoding="utf-8"))

    keys = np.asarray(meta["item_key"])
    uniques, labels = np.unique(keys, return_inverse=True)
    cat2 = np.asarray([a["category2"] for a in attrs])
    _, cat2_codes = np.unique(cat2, return_inverse=True)
    _, sleeve_codes = np.unique(attr_label["sleeve"], return_inverse=True)
    rng = np.random.RandomState(0)
    idx = np.arange(len(uniques))
    rng.shuffle(idx)
    n_train = int(len(idx) * 0.6)
    n_val = int(len(idx) * 0.2)
    train_keys = set(uniques[idx[:n_train]].tolist())
    val_keys = set(uniques[idx[n_train:n_train + n_val]].tolist())
    train_idx = np.where(np.asarray([k in train_keys for k in keys]))[0]

    model = VisualHead().to(device)
    model.load_state_dict(torch.load(os.path.join(FEAT_DIR, "visual_head.pt"), map_location="cpu", weights_only=True))
    text_proj = TextProjection().to(device)
    attr_projs = nn.ModuleDict({d: nn.Linear(512, 768).to(device) for d in ATTR_DIMS})
    params = list(model.parameters()) + list(text_proj.parameters()) + list(attr_projs.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

    for epoch in range(12):
        model.train()
        text_proj.train()
        attr_projs.train()
        eff_w = 1.0 + 2.0 * min(1.0, epoch / 4.0)
        perm = rng.permutation(train_idx)
        for i in range(0, len(perm), 128):
            bi = perm[i : i + 128]
            p = torch.from_numpy(patch[bi]).to(device).float()
            c = torch.from_numpy(cls[bi]).to(device).float()
            x = torch.from_numpy(text[bi]).to(device).float()
            lab = torch.from_numpy(labels[bi]).to(device)
            c2 = torch.from_numpy(cat2_codes[bi]).to(device)
            sl = torch.from_numpy(sleeve_codes[bi]).to(device)
            v = model(c, p)
            t = text_proj(x)
            loss = hard_infonce(v, t, lab, c2, sl, eff_w)
            for d in ATTR_DIMS:
                at = attr_projs[d](torch.from_numpy(attr_emb[d][bi]).to(device).float())
                local = local_pool(p, at)
                ac = torch.from_numpy(attr_codes[d][bi]).to(device)
                loss = loss + ATTR_LAMBDA[d] * attr_infonce(local, at, ac)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

    torch.save({d: attr_projs[d].state_dict() for d in ATTR_DIMS}, os.path.join(FEAT_DIR, "attr_projs.pt"))

    # Heatmap for a test image with a known sleeve label.
    rows = pq.read_table(os.path.join(os.path.dirname(FEAT_DIR), "deepfashion_inshop.parquet"), columns=["image"]).to_pandas()
    pick = None
    for i in range(3000, len(rows)):
        if str(attr_label["sleeve"][i]) not in ("unknown", "none"):
            pick = i
            break
    p = torch.from_numpy(patch[pick : pick + 1]).to(device).float()
    at = attr_projs["sleeve"](torch.from_numpy(attr_emb["sleeve"][pick : pick + 1]).to(device).float())
    score = (p[0] * at[0]).sum(dim=1) / (768 ** 0.5)
    hm = score.detach().softmax(dim=0).cpu().numpy().reshape(16, 16)

    img = Image.open(io.BytesIO(rows.iloc[pick]["image"]["bytes"])).convert("RGB")
    hm_up = np.asarray(Image.fromarray((hm / hm.max() * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)).astype(np.float32) / 255.0
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img)
    ax.imshow(hm_up, cmap="jet", alpha=0.55)
    ax.axis("off")
    ax.set_title(f"sleeve = {attr_label['sleeve'][pick]}")
    out = os.path.join(os.path.dirname(os.path.dirname(FEAT_DIR)), "outputs", "heatmap_sleeve.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out, "for image index", pick)


if __name__ == "__main__":
    main()
