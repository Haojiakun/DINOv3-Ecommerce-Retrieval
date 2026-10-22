import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import TextProjection, VisualHead


FEAT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features")


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def weighted_infonce(v, t, lab, cat2, color, temperature=0.07, alpha=1.0, beta=0.5):
    v = F.normalize(v, dim=1)
    t = F.normalize(t, dim=1)
    logits = (v @ t.T) / temperature
    same_cat = cat2[:, None] == cat2[None, :]
    same_color = color[:, None] == color[None, :]
    diff_item = lab[:, None] != lab[None, :]
    hard = (same_cat & diff_item).float()
    very_hard = (hard * same_color).float()
    logits = logits + alpha * hard + beta * very_hard
    gt = torch.arange(v.shape[0], device=v.device)
    return F.cross_entropy(logits, gt) + F.cross_entropy(logits.T, gt)


def recall(sim, labels, ks=(1, 5, 10), exclude_self=True):
    topk = sim.topk(max(ks) + 1, dim=1).indices.numpy()[:, 1:]
    hits = {k: [] for k in ks}
    n_q = 0
    for i, lab in enumerate(labels):
        pos = {j for j, l2 in enumerate(labels) if l2 == lab and (not exclude_self or j != i)}
        if not pos:
            continue
        n_q += 1
        for k in ks:
            hits[k].append(bool(pos.intersection(topk[i, :k].tolist())))
    return {k: float(np.mean(hits[k])) for k in ks}, n_q


def embed_both(model, text_proj, patch, cls, text, batch_size=256, device="cuda"):
    model.eval()
    text_proj.eval()
    Vs, Ts = [], []
    with torch.inference_mode():
        for i in range(0, len(patch), batch_size):
            p = torch.from_numpy(patch[i : i + batch_size]).to(device).float()
            c = torch.from_numpy(cls[i : i + batch_size]).to(device).float()
            x = torch.from_numpy(text[i : i + batch_size]).to(device).float()
            Vs.append(F.normalize(model(c, p), dim=1).cpu())
            Ts.append(F.normalize(text_proj(x), dim=1).cpu())
    return torch.cat(Vs), torch.cat(Ts)


def main():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    patch = np.load(os.path.join(FEAT_DIR, "patch.npy"), mmap_mode="r").astype(np.float32)
    cls = np.load(os.path.join(FEAT_DIR, "cls.npy"), mmap_mode="r").astype(np.float32)
    text = np.load(os.path.join(FEAT_DIR, "text_feats.npy")).astype(np.float32)
    with open(os.path.join(FEAT_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    item_keys = np.asarray(meta["item_key"])
    uniques, labels = np.unique(item_keys, return_inverse=True)
    _, cat2_codes = np.unique(np.asarray(meta["category2"]), return_inverse=True)
    _, color_codes = np.unique(np.asarray(meta["color"]), return_inverse=True)

    rng = np.random.RandomState(0)
    idx = np.arange(len(uniques))
    rng.shuffle(idx)
    n_train = int(len(idx) * 0.8)
    train_keys = set(uniques[idx[:n_train]].tolist())
    val_mask = np.asarray([k not in train_keys for k in item_keys])
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    print(f"images train/val: {len(train_idx)}/{len(val_idx)}")

    visual = torch.load(os.path.join(FEAT_DIR, "visual_head.pt"), map_location="cpu")

    def run(hard, tag, epochs=12):
        model = VisualHead().to(device)
        model.load_state_dict(visual)
        text_proj = TextProjection().to(device)
        opt = torch.optim.AdamW(
            list(model.parameters()) + list(text_proj.parameters()), lr=1e-3, weight_decay=1e-4
        )
        batch_size = 128
        for epoch in range(epochs):
            model.train()
            text_proj.train()
            perm = rng.permutation(train_idx)
            total, cnt = 0.0, 0
            for i in range(0, len(perm), batch_size):
                bi = perm[i : i + batch_size]
                p = torch.from_numpy(patch[bi]).to(device).float()
                c = torch.from_numpy(cls[bi]).to(device).float()
                x = torch.from_numpy(text[bi]).to(device).float()
                lab = torch.from_numpy(labels[bi]).to(device)
                cat2 = torch.from_numpy(cat2_codes[bi]).to(device)
                color = torch.from_numpy(color_codes[bi]).to(device)
                v = model(c, p)
                t = text_proj(x)
                if hard:
                    loss = weighted_infonce(v, t, lab, cat2, color)
                else:
                    loss = weighted_infonce(v, t, lab, cat2, color, alpha=0.0, beta=0.0)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(bi)
                cnt += len(bi)
        V, T = embed_both(model, text_proj, patch[val_idx], cls[val_idx], text[val_idx])
        r_t2i, nq_t = recall(T @ V.T, labels[val_idx], exclude_self=False)
        r_i2i, nq_i = recall(V @ V.T, labels[val_idx], exclude_self=True)
        torch.save(model.state_dict(), os.path.join(FEAT_DIR, f"align_visual_{tag}.pt"))
        torch.save(text_proj.state_dict(), os.path.join(FEAT_DIR, f"align_text_proj_{tag}.pt"))
        print(f"[{tag}] text->image: queries={nq_t} R@1={r_t2i[1]:.4f} R@5={r_t2i[5]:.4f} R@10={r_t2i[10]:.4f}")
        print(f"[{tag}] image->image: queries={nq_i} R@1={r_i2i[1]:.4f} R@5={r_i2i[5]:.4f} R@10={r_i2i[10]:.4f}")
        return r_t2i, r_i2i

    print("training plain InfoNCE ...")
    run(hard=False, tag="plain")
    print("training hard-negative InfoNCE ...")
    run(hard=True, tag="hard")


if __name__ == "__main__":
    main()
