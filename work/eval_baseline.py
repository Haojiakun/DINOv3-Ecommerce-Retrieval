import json
import os

import numpy as np
import torch


def load(path):
    return np.load(path).astype(np.float32)


def main():
    feat_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cls = load(os.path.join(feat_dir, "cls.npy"))
    with open(os.path.join(feat_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    keys = meta["item_key"]

    x = torch.from_numpy(cls).to(device)
    x = x / x.norm(dim=1, keepdim=True)
    sim = x @ x.T

    ks = [1, 5, 10, 20]
    max_k = max(ks)
    # self column is always the top match; drop it and keep max_k neighbors.
    topk = sim.topk(max_k + 1, dim=1).indices.cpu().numpy()
    topk = topk[:, 1:]  # (N, max_k)

    hits = {k: [] for k in ks}
    n_queries = 0
    for i, key in enumerate(keys):
        pos = {j for j, k2 in enumerate(keys) if k2 == key and j != i}
        if not pos:
            continue
        n_queries += 1
        for k in ks:
            hits[k].append(bool(pos.intersection(topk[i, :k].tolist())))

    print(f"queries with at least one positive: {n_queries} / {len(keys)}")
    for k in ks:
        r = float(np.mean(hits[k]))
        print(f"Recall@{k}: {r:.4f}")


if __name__ == "__main__":
    main()
