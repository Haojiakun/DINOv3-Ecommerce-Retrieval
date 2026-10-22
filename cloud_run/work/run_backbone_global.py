import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import TextProjection, VisualHead


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(ROOT, "work", os.environ.get("FEAT_DIR", "features"))
OUT_DIR = os.path.join(ROOT, "outputs")
TAG = os.environ.get("TAG", "siglip")
# DINOv3 default features use unsuffixed filenames
_suffix = f"_{TAG}" if TAG and TAG not in ("dinov3", "") else ""
SEEDS = [0, 1, 2]
ATTR_DIMS = ["material", "neckline", "sleeve", "pattern"]
ATTR_LAMBDA = {"material": 1.0, "neckline": 0.5, "sleeve": 0.5, "pattern": 0.5}


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
    topk = sim.topk(max(ks) + 1, dim=1).indices.numpy()
    if exclude_self:
        topk = topk[:, 1:]  # i2i: drop self (rank-0 is always the query image)
    else:
        topk = topk[:, :max(ks)]  # t2i: keep rank-0 (text_i -> image_i is the correct answer)
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


def global_infonce(v, t, temperature=0.07):
    v = F.normalize(v, dim=1)
    t = F.normalize(t, dim=1)
    logits = (v @ t.T) / temperature
    gt = torch.arange(v.shape[0], device=v.device)
    return F.cross_entropy(logits, gt) + F.cross_entropy(logits.T, gt)


def local_pool(patch, attr_text, dim=768, chunk=512):
    outs = []
    for i in range(0, patch.shape[0], chunk):
        p = patch[i:i + chunk]
        score = (p * attr_text.unsqueeze(1)).sum(dim=2) / (dim ** 0.5)
        attn = score.softmax(dim=1)
        outs.append((p * attn.unsqueeze(2)).sum(dim=1))
    return torch.cat(outs, dim=0)


def attr_infonce_global(a, b, codes, temperature=0.07):
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    sim = (a @ b.T) / temperature
    eye = torch.eye(a.shape[0], dtype=torch.bool, device=a.device)
    pos = (codes[:, None] == codes[None, :]) & (~eye) & (codes[:, None] >= 0)
    has = pos.sum(dim=1) > 0
    if not has.any():
        return torch.tensor(0.0, device=a.device, requires_grad=True)
    exp_sim = torch.exp(sim)
    denom = (exp_sim * (~eye)).sum(dim=1, keepdim=True)
    num = (exp_sim * pos).sum(dim=1, keepdim=True)
    return -torch.log((num + 1e-8) / (denom + 1e-8))[has].mean()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(FEAT_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    n = int(os.environ.get("N_ATTR", str(len(meta["item_key"]))))

    patch = np.load(os.path.join(FEAT_DIR, f"patch{_suffix}.npy"), mmap_mode="r").astype(np.float32)[:n]
    cls = np.load(os.path.join(FEAT_DIR, f"cls{_suffix}.npy"), mmap_mode="r").astype(np.float32)[:n]
    text = np.load(os.path.join(FEAT_DIR, "text_feats.npy")).astype(np.float32)[:n]
    attr_emb = {d: np.load(os.path.join(FEAT_DIR, f"attr_emb_{d}.npy")).astype(np.float32)[:n] for d in ATTR_DIMS}
    attr_codes = {
        d: make_codes(np.load(os.path.join(FEAT_DIR, f"attr_label_{d}.npy"), allow_pickle=True))
        for d in ATTR_DIMS
    }

    item_keys = np.asarray(meta["item_key"])[:n]
    uniques, labels = np.unique(item_keys, return_inverse=True)

    mrng = np.random.RandomState(0)
    midx = np.arange(len(uniques))
    mrng.shuffle(midx)
    m_n_train = int(len(midx) * 0.6)
    m_n_val = int(len(midx) * 0.2)
    m_train_keys = set(uniques[midx[:m_n_train]].tolist())
    m_test_keys = set(uniques[midx[m_n_train + m_n_val:]].tolist())
    m_train_idx = np.where(np.asarray([k in m_train_keys for k in item_keys]))[0]
    m_test_idx = np.where(np.asarray([k in m_test_keys for k in item_keys]))[0]

    head_path = os.path.join(FEAT_DIR, f"visual_head{_suffix}.pt")
    visual = torch.load(head_path, map_location="cpu")
    print(f"[{TAG}] head={head_path}, items train/test: {len(m_train_keys)}/{len(m_test_keys)}")

    for seed in SEEDS:
        for use_local, tag in [(False, "plain"), (True, "local")]:
            torch.manual_seed(seed)
            model = VisualHead().to(device)
            model.load_state_dict(visual)
            text_proj = TextProjection().to(device)
            attr_projs = nn.ModuleDict({d: nn.Linear(512, 768).to(device) for d in ATTR_DIMS})
            params = list(model.parameters()) + list(text_proj.parameters()) + list(attr_projs.parameters())
            opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

            for epoch in range(50):
                model.train()
                text_proj.train()
                attr_projs.train()
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
                loss = global_infonce(V, T)
                if use_local:
                    for d in ATTR_DIMS:
                        Ls, As = [], []
                        for i in range(0, len(m_train_idx), 128):
                            bi = m_train_idx[i : i + 128]
                            p = torch.from_numpy(patch[bi]).to(device).float()
                            at = attr_projs[d](torch.from_numpy(attr_emb[d][bi]).to(device).float())
                            Ls.append(local_pool(p, at))
                            As.append(at)
                        L = torch.cat(Ls)
                        A = torch.cat(As)
                        ac = torch.from_numpy(attr_codes[d][m_train_idx]).to(device)
                        loss = loss + ATTR_LAMBDA[d] * attr_infonce_global(L, A, ac, 0.07)
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
            print(f"[{TAG} seed={seed} {tag}] t2i R@1/R@5/R@10={r_t2i[1]:.4f}/{r_t2i[5]:.4f}/{r_t2i[10]:.4f} | i2i R@1/R@10={r_i2i[1]:.4f}/{r_i2i[10]:.4f}")


if __name__ == "__main__":
    main()
