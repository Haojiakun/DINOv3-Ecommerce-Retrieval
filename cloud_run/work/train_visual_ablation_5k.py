import json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, "/root/cloud_run/work")
from models import AttentionPooling, VisualHead

ROOT = "/root/cloud_run"
FEAT_DIR = os.path.join(ROOT, "work", "features_5k")
EPOCHS = 50
BATCH = 256
LR = 1e-3
TEMP = 0.07

def recall(sim, labels, ks=(1, 5, 10), exclude_self=True):
    topk = sim.topk(max(ks) + 1, dim=1).indices.numpy()
    if exclude_self: topk = topk[:, 1:]
    else: topk = topk[:, :max(ks)]
    hits = {k: [] for k in ks}; n_q = 0
    for i, lab in enumerate(labels):
        pos = {j for j, l2 in enumerate(labels) if l2 == lab and (not exclude_self or j != i)}
        if not pos: continue
        n_q += 1
        for k in ks: hits[k].append(bool(pos.intersection(topk[i, :k].tolist())))
    return {k: float(np.mean(hits[k])) for k in ks}, n_q

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


class AttnOnlyHead(nn.Module):
    """Attention pooling on patches + projection, no CLS fusion."""
    def __init__(self, embed_dim=768, output_dim=256):
        super().__init__()
        self.attn_pool = AttentionPooling(embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, output_dim),
        )
    def forward(self, cls_token, patch_tokens):
        pooled = self.attn_pool(patch_tokens)
        return self.proj(pooled)


def train_eval(model, train_idx, test_idx, labels, patch, cls, device, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    for epoch in range(EPOCHS):
        model.train()
        perm = np.random.permutation(train_idx)
        for i in range(0, len(perm), BATCH):
            bi = perm[i:i+BATCH]
            if len(bi) < 2: continue
            p = torch.from_numpy(patch[bi]).to(device).float()
            c = torch.from_numpy(cls[bi]).to(device).float()
            lab = torch.from_numpy(labels[bi]).to(device)
            z = model(c, p)
            loss = supcon_loss(z, lab)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    Vs = []
    with torch.inference_mode():
        for i in range(0, len(test_idx), BATCH):
            bi = test_idx[i:i+BATCH]
            p = torch.from_numpy(patch[bi]).to(device).float()
            c = torch.from_numpy(cls[bi]).to(device).float()
            Vs.append(F.normalize(model(c, p), dim=1).cpu())
    V = torch.cat(Vs)
    r, nq = recall(V @ V.T, labels[test_idx], exclude_self=True)
    return r

def main():
    device = "cuda"
    patch = np.load(os.path.join(FEAT_DIR, "patch.npy"))
    cls = np.load(os.path.join(FEAT_DIR, "cls.npy"))
    with open(os.path.join(FEAT_DIR, "meta.json")) as f: meta = json.load(f)
    item_keys = np.asarray(meta["item_key"])
    uniques, labels = np.unique(item_keys, return_inverse=True)
    mrng = np.random.RandomState(0); midx = np.arange(len(uniques)); mrng.shuffle(midx)
    n_train = int(len(midx)*0.6); n_val = int(len(midx)*0.2)
    train_keys = set(uniques[midx[:n_train]].tolist())
    test_keys = set(uniques[midx[n_train+n_val:]].tolist())
    train_idx = np.where(np.asarray([k in train_keys for k in item_keys]))[0]
    test_idx = np.where(np.asarray([k in test_keys for k in item_keys]))[0]
    print(f"train={len(train_idx)} test={len(test_idx)}")

    results = {}
    for name, model in [("AttentionPool only", AttnOnlyHead()), ("CLS+Attn+MLP", VisualHead())]:
        r = train_eval(model, train_idx, test_idx, labels, patch, cls, device, seed=0)
        results[name] = r
        print(f"{name}: R@1={r[1]*100:.1f} R@5={r[5]*100:.1f} R@10={r[10]*100:.1f}")

if __name__ == "__main__":
    main()
