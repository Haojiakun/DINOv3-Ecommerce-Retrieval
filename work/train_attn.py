import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


class AttentionPool(nn.Module):
    """Softmax attention over patch tokens -> (B, D)."""

    def __init__(self, dim):
        super().__init__()
        self.scorer = nn.Linear(dim, 1)

    def forward(self, patches):
        tokens = F.normalize(patches, dim=-1)
        score = self.scorer(tokens).squeeze(-1)
        attn = score.softmax(dim=1)
        return (attn.unsqueeze(-1) * patches).sum(dim=1)


class FusionHead(nn.Module):
    """concat(CLS, AttentionPool(patches)) -> MLP -> (B, D)."""

    def __init__(self, dim):
        super().__init__()
        self.pool = AttentionPool(dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, patches, cls):
        pooled = self.pool(patches)
        return self.mlp(torch.cat([cls, pooled], dim=1))


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
    loss = -torch.log((num + 1e-8) / (denom + 1e-8))
    return loss[has_pos].mean()


def embed(model, patches, cls, batch_size=256, device="cuda", fusion=False):
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(patches), batch_size):
            p = torch.from_numpy(patches[i : i + batch_size]).to(device).float()
            c = torch.from_numpy(cls[i : i + batch_size]).to(device).float()
            z = model(p, c) if fusion else model(p)
            out.append(F.normalize(z, dim=1).cpu())
    return torch.cat(out)


def recall_at_k(sim, labels, ks=(1, 5, 10)):
    topk = sim.topk(max(ks) + 1, dim=1).indices.numpy()[:, 1:]
    hits = {k: [] for k in ks}
    n_q = 0
    for i, lab in enumerate(labels):
        pos = {j for j, l2 in enumerate(labels) if l2 == lab and j != i}
        if not pos:
            continue
        n_q += 1
        for k in ks:
            hits[k].append(bool(pos.intersection(topk[i, :k].tolist())))
    return {k: float(np.mean(hits[k])) for k in ks}, n_q


def main():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features")

    patch = np.load(os.path.join(feat_dir, "patch.npy"), mmap_mode="r").astype(np.float32)
    cls = np.load(os.path.join(feat_dir, "cls.npy"), mmap_mode="r").astype(np.float32)
    with open(os.path.join(feat_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    item_keys = np.asarray(meta["item_key"])
    uniques, labels = np.unique(item_keys, return_inverse=True)

    rng = np.random.RandomState(0)
    idx = np.arange(len(uniques))
    rng.shuffle(idx)
    n_train = int(len(idx) * 0.8)
    train_keys = set(uniques[idx[:n_train]].tolist())
    val_mask = np.asarray([k not in train_keys for k in item_keys])
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    print(f"items train/val: {n_train}/{len(uniques) - n_train}, images train/val: {len(train_idx)}/{len(val_idx)}")

    v_cls = torch.from_numpy(cls[val_idx]).float()
    v_cls = F.normalize(v_cls, dim=1)
    r, nq = recall_at_k(v_cls @ v_cls.T, labels[val_idx])
    print(f"val CLS baseline       : queries={nq} R@1={r[1]:.4f} R@5={r[5]:.4f} R@10={r[10]:.4f}")

    v_patch = torch.from_numpy(patch[val_idx]).float()
    v_mean = F.normalize(v_patch.mean(dim=1), dim=1)
    r, nq = recall_at_k(v_mean @ v_mean.T, labels[val_idx])
    print(f"val mean-pool baseline : queries={nq} R@1={r[1]:.4f} R@5={r[5]:.4f} R@10={r[10]:.4f}")

    def train_model(model, fusion, epochs=15, lr=1e-3, batch_size=128):
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        for epoch in range(epochs):
            model.train()
            perm = rng.permutation(train_idx)
            total, cnt = 0.0, 0
            for i in range(0, len(perm), batch_size):
                bi = perm[i : i + batch_size]
                p = torch.from_numpy(patch[bi]).to(device).float()
                c = torch.from_numpy(cls[bi]).to(device).float()
                lab = torch.from_numpy(labels[bi]).to(device)
                z = model(p, c) if fusion else model(p)
                loss = supcon_loss(z, lab)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(bi)
                cnt += len(bi)
            if (epoch + 1) % 5 == 0:
                print(f"  epoch {epoch + 1}: loss={total / max(cnt, 1):.4f}")
        return model

    print("training AttentionPool ...")
    pool = train_model(AttentionPool(768), fusion=False)
    z = embed(pool, patch[val_idx], cls[val_idx], fusion=False)
    r, nq = recall_at_k(z @ z.T, labels[val_idx])
    print(f"val AttentionPool      : queries={nq} R@1={r[1]:.4f} R@5={r[5]:.4f} R@10={r[10]:.4f}")
    torch.save(pool.state_dict(), os.path.join(feat_dir, "attn_pool.pt"))

    print("training FusionHead (CLS + AttentionPool) ...")
    fusion = train_model(FusionHead(768), fusion=True)
    z = embed(fusion, patch[val_idx], cls[val_idx], fusion=True)
    r, nq = recall_at_k(z @ z.T, labels[val_idx])
    print(f"val FusionHead         : queries={nq} R@1={r[1]:.4f} R@5={r[5]:.4f} R@10={r[10]:.4f}")
    torch.save(fusion.state_dict(), os.path.join(feat_dir, "attn_fusion.pt"))

    print("saved work/features/attn_pool.pt and work/features/attn_fusion.pt")


if __name__ == "__main__":
    main()
