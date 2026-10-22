import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import TextProjection, VisualHead


ROOT = r"D:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper"
FEAT_DIR = os.path.join(ROOT, "work", "features")
ATTR_DIMS = ["material", "neckline", "sleeve", "pattern"]


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


def hard_infonce(v, t, lab, cat2, sleeve, w, temperature=0.07):
    v = F.normalize(v, dim=1)
    t = F.normalize(t, dim=1)
    logits = (v @ t.T) / temperature
    same_cat = cat2[:, None] == cat2[None, :]
    diff_item = lab[:, None] != lab[None, :]
    diff_sleeve = sleeve[:, None] != sleeve[None, :]
    hard = (same_cat & diff_item & diff_sleeve).float()
    logits = logits + (w - 1.0) * hard
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patch = np.load(os.path.join(FEAT_DIR, "patch.npy"), mmap_mode="r").astype(np.float32)
    cls = np.load(os.path.join(FEAT_DIR, "cls.npy"), mmap_mode="r").astype(np.float32)
    text = np.load(os.path.join(FEAT_DIR, "text_feats.npy")).astype(np.float32)
    with open(os.path.join(FEAT_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(FEAT_DIR, "filtered_attrs.json"), encoding="utf-8") as f:
        attrs = json.load(f)

    item_keys = np.asarray(meta["item_key"])
    uniques, labels = np.unique(item_keys, return_inverse=True)
    cat2 = np.asarray([a["category2"] for a in attrs])
    _, cat2_codes = np.unique(cat2, return_inverse=True)
    sleeve_codes = make_codes(np.load(os.path.join(FEAT_DIR, "attr_label_sleeve.npy"), allow_pickle=True))

    mrng = np.random.RandomState(0)
    midx = np.arange(len(uniques))
    mrng.shuffle(midx)
    m_n_train = int(len(midx) * 0.6)
    m_n_val = int(len(midx) * 0.2)
    m_train_keys = set(uniques[midx[:m_n_train]].tolist())
    m_test_keys = set(uniques[midx[m_n_train + m_n_val:]].tolist())
    m_train_idx = np.where(np.asarray([k in m_train_keys for k in item_keys]))[0]
    m_test_idx = np.where(np.asarray([k in m_test_keys for k in item_keys]))[0]

    visual = torch.load(os.path.join(FEAT_DIR, "visual_head.pt"), map_location="cpu")

    def run(w, tag):
        torch.manual_seed(0)
        trng = np.random.RandomState(0)
        model = VisualHead().to(device)
        model.load_state_dict(visual)
        text_proj = TextProjection().to(device)
        attr_projs = nn.ModuleDict({d: nn.Linear(512, 768).to(device) for d in ATTR_DIMS})
        params = list(model.parameters()) + list(text_proj.parameters()) + list(attr_projs.parameters())
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
        for epoch in range(12):
            model.train()
            text_proj.train()
            attr_projs.train()
            eff_w = 1.0 + (w - 1.0) * min(1.0, epoch / 4.0)
            perm = trng.permutation(m_train_idx)
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
                loss = hard_infonce(v, t, lab, c2, sl, eff_w, 0.07)
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
        print(f"[{tag}] t2i R@1/R@5/R@10={r_t2i[1]:.4f}/{r_t2i[5]:.4f}/{r_t2i[10]:.4f} | i2i R@1={r_i2i[1]:.4f} R@10={r_i2i[10]:.4f}")

    run(1.0, "w1")
    run(3.0, "w3")


if __name__ == "__main__":
    main()
