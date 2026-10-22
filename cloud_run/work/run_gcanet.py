import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(ROOT, "work", os.environ.get("FEAT_DIR", "features"))
OUT_DIR = os.path.join(ROOT, "outputs")
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


class GlobalContextualAttention(nn.Module):
    """Self-attention over patches (spatial context) + channel attention (redundancy removal)."""

    def __init__(self, dim=768, heads=4, bottleneck=128):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.channel = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, dim),
            nn.Sigmoid(),
        )

    def forward(self, patches):
        x = self.norm1(patches)
        x, _ = self.attn(x, x, x)
        x = patches + x
        gap = x.mean(dim=1)
        scale = self.channel(gap)
        return x * scale.unsqueeze(1)


class GCAVisualHead(nn.Module):
    """GCA module + learned patch pooling + CLS fusion -> 256-d global feature."""

    def __init__(self, dim=768, output_dim=256):
        super().__init__()
        self.gca = GlobalContextualAttention(dim)
        self.score = nn.Linear(dim, 1)
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, output_dim),
        )

    def forward(self, cls_token, patches):
        ctx = checkpoint(self.gca, patches, use_reentrant=False)
        w = self.score(ctx).softmax(dim=1)
        pooled = (ctx * w).sum(dim=1)
        fused = torch.cat([cls_token, pooled], dim=-1)
        return self.proj(fused), ctx


class TextProjection(nn.Module):
    def __init__(self, input_dim=512, output_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        return self.proj(x)


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


def supcon_loss(emb, labels, temperature=0.07):
    emb = F.normalize(emb, dim=1)
    sim = (emb @ emb.T) / temperature
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).fill_diagonal_(False)
    has_pos = pos_mask.sum(dim=1) > 0
    if not has_pos.any():
        return torch.tensor(0.0, device=emb.device, requires_grad=True)
    exp_sim = torch.exp(sim)
    eye = torch.eye(emb.shape[0], dtype=torch.bool, device=emb.device)
    denom = (exp_sim * (~eye)).sum(dim=1, keepdim=True)
    num = (exp_sim * pos_mask).sum(dim=1, keepdim=True)
    return -torch.log((num + 1e-8) / (denom + 1e-8))[has_pos].mean()


def local_pool(patch, attr_text, dim=768):
    score = (patch * attr_text.unsqueeze(1)).sum(dim=2) / (dim ** 0.5)
    attn = score.softmax(dim=1)
    return (patch * attn.unsqueeze(2)).sum(dim=1)


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


def pretrain_visual(patch, cls, labels, train_idx, val_idx, device, epochs=20):
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    model = GCAVisualHead().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batch_size = 128
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(train_idx)
        for i in range(0, len(perm), batch_size):
            bi = perm[i : i + batch_size]
            p = torch.from_numpy(patch[bi]).to(device).float()
            c = torch.from_numpy(cls[bi]).to(device).float()
            lab = torch.from_numpy(labels[bi]).to(device)
            z, _ = model(c, p)
            loss = supcon_loss(z, lab)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 5 == 0:
            model.eval()
            zs = []
            with torch.inference_mode():
                for i in range(0, len(val_idx), batch_size):
                    bi = val_idx[i : i + batch_size]
                    p = torch.from_numpy(patch[bi]).to(device).float()
                    c = torch.from_numpy(cls[bi]).to(device).float()
                    z, _ = model(c, p)
                    zs.append(F.normalize(z, dim=1).cpu())
            Z = torch.cat(zs)
            r, _ = recall(Z @ Z.T, labels[val_idx])
            print(f"[gca pretrain epoch {epoch + 1}] val i2i R@1={r[1]:.4f} R@10={r[10]:.4f}")
    return model.state_dict()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(FEAT_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    n = int(os.environ.get("N_ATTR", str(len(meta["item_key"]))))
    pre_epochs = int(os.environ.get("PRE_EPOCHS", "20"))
    ft_epochs = int(os.environ.get("FT_EPOCHS", "50"))
    smoke = os.environ.get("SMOKE", "0") == "1"
    seeds = SEEDS[:1] if smoke else SEEDS

    patch = np.load(os.path.join(FEAT_DIR, "patch.npy"), mmap_mode="r").astype(np.float32)[:n]
    cls = np.load(os.path.join(FEAT_DIR, "cls.npy"), mmap_mode="r").astype(np.float32)[:n]
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
    m_val_keys = set(uniques[midx[m_n_train : m_n_train + m_n_val]].tolist())
    m_test_keys = set(uniques[midx[m_n_train + m_n_val :]].tolist())
    m_train_idx = np.where(np.asarray([k in m_train_keys for k in item_keys]))[0]
    m_test_idx = np.where(np.asarray([k in m_test_keys for k in item_keys]))[0]
    # Visual pre-training uses the 80/20 split (train+val vs test), matching train_visual.py.
    pre_train_keys = m_train_keys | m_val_keys
    pre_train_idx = np.where(np.asarray([k in pre_train_keys for k in item_keys]))[0]
    pre_val_idx = np.where(np.asarray([k not in pre_train_keys for k in item_keys]))[0]
    print(f"[gcanet] items train/test: {len(m_train_keys)}/{len(m_test_keys)}, "
          f"images train/test: {len(m_train_idx)}/{len(m_test_idx)}, "
          f"pretrain images train/val: {len(pre_train_idx)}/{len(pre_val_idx)}")

    print(f"[gcanet] visual pretraining ({pre_epochs} epochs)")
    pre_state = pretrain_visual(patch, cls, labels, pre_train_idx, pre_val_idx, device, epochs=pre_epochs)

    all_rows = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = GCAVisualHead().to(device)
        model.load_state_dict(pre_state)
        text_proj = TextProjection().to(device)
        attr_projs = nn.ModuleDict({d: nn.Linear(512, 768).to(device) for d in ATTR_DIMS})
        params = list(model.parameters()) + list(text_proj.parameters()) + list(attr_projs.parameters())
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

        for epoch in range(ft_epochs):
            model.train()
            text_proj.train()
            attr_projs.train()
            Vs, Ts = [], []
            for i in range(0, len(m_train_idx), 128):
                bi = m_train_idx[i : i + 128]
                p = torch.from_numpy(patch[bi]).to(device).float()
                c = torch.from_numpy(cls[bi]).to(device).float()
                x = torch.from_numpy(text[bi]).to(device).float()
                v, _ = model(c, p)
                Vs.append(v)
                Ts.append(text_proj(x))
            V = torch.cat(Vs)
            T = torch.cat(Ts)
            loss = global_infonce(V, T)
            for d in ATTR_DIMS:
                As, Ls = [], []
                for i in range(0, len(m_train_idx), 128):
                    bi = m_train_idx[i : i + 128]
                    p = torch.from_numpy(patch[bi]).to(device).float()
                    at = attr_projs[d](torch.from_numpy(attr_emb[d][bi]).to(device).float())
                    As.append(at)
                    Ls.append(local_pool(p, at))
                A = torch.cat(As)
                L = torch.cat(Ls)
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
                v, _ = model(c, p)
                Vs.append(F.normalize(v, dim=1).cpu())
                Ts.append(F.normalize(text_proj(x), dim=1).cpu())
        V = torch.cat(Vs)
        T = torch.cat(Ts)
        r_t2i, _ = recall(T @ V.T, labels[m_test_idx], exclude_self=False)
        r_i2i, _ = recall(V @ V.T, labels[m_test_idx], exclude_self=True)
        all_rows.append((r_t2i[1], r_t2i[5], r_t2i[10], r_i2i[1], r_i2i[10]))
        print(f"[gcanet seed={seed}] t2i R@1/R@5/R@10={r_t2i[1]:.4f}/{r_t2i[5]:.4f}/{r_t2i[10]:.4f} | "
              f"i2i R@1/R@10={r_i2i[1]:.4f}/{r_i2i[10]:.4f}")

    arr = np.asarray(all_rows) * 100.0
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    line = (
        f"GCA-Net-style (reimpl): t2i R@1/R@5/R@10 = {mean[0]:.1f} +/- {std[0]:.1f} / "
        f"{mean[1]:.1f} +/- {std[1]:.1f} / {mean[2]:.1f} +/- {std[2]:.1f} | "
        f"i2i R@1/R@10 = {mean[3]:.1f} +/- {std[3]:.1f} / {mean[4]:.1f} +/- {std[4]:.1f}"
    )
    print(line)
    with open(os.path.join(OUT_DIR, "gcanet_results.md"), "w", encoding="utf-8") as f:
        f.write("# GCA-Net-style baseline (unified protocol, 3 seeds)\n\n" + line + "\n")
    print("saved", os.path.join(OUT_DIR, "gcanet_results.md"))


if __name__ == "__main__":
    main()
