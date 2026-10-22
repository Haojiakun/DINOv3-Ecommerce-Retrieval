import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import TextProjection, VisualHead


ROOT = r"D:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper"
FEAT_DIR = os.path.join(ROOT, "work", "features")
OUT_DIR = os.path.join(ROOT, "outputs")
SEEDS = [0, 1, 2]
CONFIGS = [(1.0, False, "w1"), (3.0, False, "w3")]


def make_codes(vals):
    codes = np.empty(len(vals), dtype=np.int64)
    seen = {}
    nxt = 0
    for i, v in enumerate(vals):
        v = str(v)
        if v in ("unknown", "none", ""):
            codes[i] = -1 - i
        else:
            if v not in seen:
                seen[v] = nxt
                nxt += 1
            codes[i] = seen[v]
    return codes


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


def global_hard_infonce(v, t, hard_mask, w, temperature=0.07):
    v = F.normalize(v, dim=1)
    t = F.normalize(t, dim=1)
    logits = (v @ t.T) / temperature
    if hard_mask is not None and w > 1.0:
        logits = logits + (w - 1.0) * hard_mask
    gt = torch.arange(v.shape[0], device=v.device)
    return F.cross_entropy(logits, gt) + F.cross_entropy(logits.T, gt)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = int(os.environ.get("N_ATTR", "5000"))

    patch = np.load(os.path.join(FEAT_DIR, "patch.npy"), mmap_mode="r").astype(np.float32)[:n]
    cls = np.load(os.path.join(FEAT_DIR, "cls.npy"), mmap_mode="r").astype(np.float32)[:n]
    text = np.load(os.path.join(FEAT_DIR, "text_feats.npy")).astype(np.float32)[:n]
    with open(os.path.join(FEAT_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(FEAT_DIR, "filtered_attrs.json"), encoding="utf-8") as f:
        attrs = json.load(f)

    item_keys = np.asarray(meta["item_key"])[:n]
    uniques, labels = np.unique(item_keys, return_inverse=True)
    cat2 = np.asarray([a["category2"] for a in attrs])
    _, cat2_codes = np.unique(cat2, return_inverse=True)
    sleeve_codes = make_codes(np.load(os.path.join(FEAT_DIR, "attr_label_sleeve.npy"), allow_pickle=True))

    # Fixed 60/20/20 split by item (identical to the paper).
    mrng = np.random.RandomState(0)
    midx = np.arange(len(uniques))
    mrng.shuffle(midx)
    m_n_train = int(len(midx) * 0.6)
    m_n_val = int(len(midx) * 0.2)
    m_train_keys = set(uniques[midx[:m_n_train]].tolist())
    m_test_keys = set(uniques[midx[m_n_train + m_n_val:]].tolist())
    m_train_idx = np.where(np.asarray([k in m_train_keys for k in item_keys]))[0]
    m_test_idx = np.where(np.asarray([k in m_test_keys for k in item_keys]))[0]
    print(f"items train/test: {len(m_train_keys)}/{len(m_test_keys)}, images: {len(m_train_idx)}/{len(m_test_idx)}")

    # Offline hard-negative mask over training images: same category, different item,
    # different sleeve. Computed once, before any training.
    tr_labels = labels[m_train_idx]
    tr_cat2 = cat2_codes[m_train_idx]
    tr_sleeve = sleeve_codes[m_train_idx]
    same_cat = tr_cat2[:, None] == tr_cat2[None, :]
    diff_item = tr_labels[:, None] != tr_labels[None, :]
    diff_sleeve = tr_sleeve[:, None] != tr_sleeve[None, :]
    hard_mask = (same_cat & diff_item & diff_sleeve).astype(np.float32)
    n_hard_pairs = int(hard_mask.sum())
    print(f"offline hard pairs: {n_hard_pairs}, density={hard_mask.mean():.4f}")
    hard_mask_t = torch.from_numpy(hard_mask).to(device)

    visual = torch.load(os.path.join(FEAT_DIR, "visual_head.pt"), map_location="cpu")

    results = {tag: [] for _, _, tag in CONFIGS}
    for seed in SEEDS:
        for w, use_local, tag in CONFIGS:
            torch.manual_seed(seed)
            model = VisualHead().to(device)
            model.load_state_dict(visual)
            text_proj = TextProjection().to(device)
            params = list(model.parameters()) + list(text_proj.parameters())
            opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

            epochs = 50
            for epoch in range(epochs):
                model.train()
                text_proj.train()
                eff_w = 1.0 + (w - 1.0) * min(1.0, epoch / 4.0)
                Vs, Ts = [], []
                for i in range(0, len(m_train_idx), 128):
                    bi = m_train_idx[i : i + 128]
                    p = torch.from_numpy(patch[bi]).to(device).float()
                    c = torch.from_numpy(cls[bi]).to(device).float()
                    x = torch.from_numpy(text[bi]).to(device).float()
                    Vs.append(model(c, p))
                    Ts.append(text_proj(x))
                V = torch.cat(Vs)
                T = torch.cat(Ts)
                loss = global_hard_infonce(V, T, hard_mask_t, eff_w, 0.07)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

            model.eval()
            text_proj.eval()
            Vs, Ts = [], []
            with torch.inference_mode():
                for i in range(0, len(m_test_idx), 128):
                    bi = m_test_idx[i : i + 128]
                    p = torch.from_numpy(patch[bi]).to(device).float()
                    c = torch.from_numpy(cls[bi]).to(device).float()
                    x = torch.from_numpy(text[bi]).to(device).float()
                    Vs.append(F.normalize(model(c, p), dim=1).cpu())
                    Ts.append(F.normalize(text_proj(x), dim=1).cpu())
            V = torch.cat(Vs)
            T = torch.cat(Ts)
            r_t2i, _ = recall(T @ V.T, labels[m_test_idx], exclude_self=False)
            r_i2i, _ = recall(V @ V.T, labels[m_test_idx], exclude_self=True)
            row = (r_t2i[1], r_t2i[5], r_t2i[10], r_i2i[1], r_i2i[10])
            results[tag].append(row)
            print(f"[seed={seed} {tag}] t2i R@1/R@5/R@10={r_t2i[1]:.4f}/{r_t2i[5]:.4f}/{r_t2i[10]:.4f} | i2i R@1/R@10={r_i2i[1]:.4f}/{r_i2i[10]:.4f}")

    lines = ["# Offline hard-negative mining, 3-seed mean±std (global InfoNCE, fixed visual head)", ""]
    for _, _, tag in CONFIGS:
        arr = np.asarray(results[tag]) * 100.0
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        line = (
            f"{tag}: t2i R@1/R@5/R@10 = {mean[0]:.1f}±{std[0]:.1f} / {mean[1]:.1f}±{std[1]:.1f} / "
            f"{mean[2]:.1f}±{std[2]:.1f} | i2i R@1/R@10 = {mean[3]:.1f}±{std[3]:.1f} / {mean[4]:.1f}±{std[4]:.1f}"
        )
        print(line)
        lines.append(line)
    with open(os.path.join(OUT_DIR, "seeds_offline_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("saved", os.path.join(OUT_DIR, "seeds_offline_results.md"))


if __name__ == "__main__":
    main()
