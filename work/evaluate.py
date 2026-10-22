import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from models import VisualHead


FEAT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features")


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
    val_idx = np.where(val_mask)[0]

    # CLS baseline on the same val split for reference.
    v_cls = torch.from_numpy(cls[val_idx]).float()
    v_cls = F.normalize(v_cls, dim=1)
    r, nq = recall_at_k(v_cls @ v_cls.T, labels[val_idx])
    print(f"CLS baseline (val): queries={nq} R@1={r[1]:.4f} R@5={r[5]:.4f} R@10={r[10]:.4f}")

    model = VisualHead().to(device)
    weight = os.path.join(FEAT_DIR, "visual_head.pt")
    model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
    model.eval()

    out = []
    with torch.inference_mode():
        for i in range(0, len(val_idx), 256):
            p = torch.from_numpy(patch[val_idx[i : i + 256]]).to(device).float()
            c = torch.from_numpy(cls[val_idx[i : i + 256]]).to(device).float()
            out.append(F.normalize(model(c, p), dim=1).cpu())
    z = torch.cat(out)
    r, nq = recall_at_k(z @ z.T, labels[val_idx])
    print(f"VisualHead (val): queries={nq} R@1={r[1]:.4f} R@5={r[5]:.4f} R@10={r[10]:.4f}")


if __name__ == "__main__":
    main()
