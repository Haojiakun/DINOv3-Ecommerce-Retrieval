import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from models import VisualHead


FEAT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("FEAT_DIR", "features"))


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def embed(model, patch, cls, batch_size=256, device="cuda"):
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(patch), batch_size):
            p = torch.from_numpy(patch[i : i + batch_size]).to(device).float()
            c = torch.from_numpy(cls[i : i + batch_size]).to(device).float()
            out.append(F.normalize(model(c, p), dim=1).cpu())
    return torch.cat(out)


def main():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    patch = np.load(os.path.join(FEAT_DIR, "patch.npy"), mmap_mode="r").astype(np.float32)
    cls = np.load(os.path.join(FEAT_DIR, "cls.npy"), mmap_mode="r").astype(np.float32)
    with open(os.path.join(FEAT_DIR, "meta.json"), encoding="utf-8") as f:
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

    model = VisualHead().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs, batch_size = 20, 128
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(train_idx)
        total, cnt = 0.0, 0
        for i in range(0, len(perm), batch_size):
            bi = perm[i : i + batch_size]
            p = torch.from_numpy(patch[bi]).to(device).float()
            c = torch.from_numpy(cls[bi]).to(device).float()
            lab = torch.from_numpy(labels[bi]).to(device)
            z = model(c, p)
            loss = supcon_loss(z, lab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(bi)
            cnt += len(bi)
        if (epoch + 1) % 5 == 0:
            z = embed(model, patch[val_idx], cls[val_idx])
            r, nq = recall_at_k(z @ z.T, labels[val_idx])
            print(f"epoch {epoch + 1}: loss={total / max(cnt, 1):.4f} val R@1={r[1]:.4f} R@5={r[5]:.4f} R@10={r[10]:.4f}")

    torch.save(model.state_dict(), os.path.join(FEAT_DIR, "visual_head.pt"))
    print("saved", os.path.join(FEAT_DIR, "visual_head.pt"))


if __name__ == "__main__":
    main()
