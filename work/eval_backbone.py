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


def infonce(v, t, temperature=0.07):
    v = F.normalize(v, dim=1)
    t = F.normalize(t, dim=1)
    logits = (v @ t.T) / temperature
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


def main():
    set_seed(0)
    tag = os.environ.get("TAG", "dinov3")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls_file = os.path.join(FEAT_DIR, "cls.npy" if tag == "dinov3" else f"cls_{tag}.npy")
    patch_file = os.path.join(FEAT_DIR, "patch.npy" if tag == "dinov3" else f"patch_{tag}.npy")

    patch = np.load(patch_file, mmap_mode="r").astype(np.float32)
    cls = np.load(cls_file, mmap_mode="r").astype(np.float32)
    text = np.load(os.path.join(FEAT_DIR, "text_feats.npy")).astype(np.float32)
    meta = json.load(open(os.path.join(FEAT_DIR, "meta.json"), encoding="utf-8"))
    keys = np.asarray(meta["item_key"])
    uniques, labels = np.unique(keys, return_inverse=True)
    rng = np.random.RandomState(0)
    idx = np.arange(len(uniques))
    rng.shuffle(idx)
    n_train = int(len(idx) * 0.6)
    n_val = int(len(idx) * 0.2)
    train_keys = set(uniques[idx[:n_train]].tolist())
    val_keys = set(uniques[idx[n_train:n_train + n_val]].tolist())
    test_keys = set(uniques[idx[n_train + n_val:]].tolist())
    train_idx = np.where(np.asarray([k in train_keys for k in keys]))[0]
    test_idx = np.where(np.asarray([k in test_keys for k in keys]))[0]

    model = VisualHead().to(device)
    text_proj = TextProjection().to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(text_proj.parameters()), lr=1e-3, weight_decay=1e-4)
    for epoch in range(12):
        model.train()
        text_proj.train()
        perm = rng.permutation(train_idx)
        for i in range(0, len(perm), 128):
            bi = perm[i : i + 128]
            p = torch.from_numpy(patch[bi]).to(device).float()
            c = torch.from_numpy(cls[bi]).to(device).float()
            x = torch.from_numpy(text[bi]).to(device).float()
            loss = infonce(model(c, p), text_proj(x))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(text_proj.parameters()), 1.0)
            opt.step()

    model.eval()
    text_proj.eval()
    Vs, Ts = [], []
    with torch.inference_mode():
        for i in range(0, len(test_idx), 128):
            bi = test_idx[i : i + 128]
            p = torch.from_numpy(patch[bi]).to(device).float()
            c = torch.from_numpy(cls[bi]).to(device).float()
            x = torch.from_numpy(text[bi]).to(device).float()
            Vs.append(F.normalize(model(c, p), dim=1).cpu())
            Ts.append(F.normalize(text_proj(x), dim=1).cpu())
    V = torch.cat(Vs)
    T = torch.cat(Ts)
    r_t2i, _ = recall(T @ V.T, labels[test_idx], exclude_self=False)
    r_i2i, _ = recall(V @ V.T, labels[test_idx], exclude_self=True)
    print(f"[{tag}] text->image R@1={r_t2i[1]:.4f} R@5={r_t2i[5]:.4f} R@10={r_t2i[10]:.4f} | image->image R@1={r_i2i[1]:.4f} R@10={r_i2i[10]:.4f}")


if __name__ == "__main__":
    main()
