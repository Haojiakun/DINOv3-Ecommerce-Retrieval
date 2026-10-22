import json
import os

import numpy as np
import torch
import torch.nn.functional as F


FEAT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features")


def recall(sim, labels, ks=(1, 5, 10)):
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
    meta = json.load(open(os.path.join(FEAT_DIR, "meta.json"), encoding="utf-8"))
    keys = np.asarray(meta["item_key"])
    uniques, labels = np.unique(keys, return_inverse=True)
    rng = np.random.RandomState(0)
    idx = np.arange(len(uniques))
    rng.shuffle(idx)
    n_train = int(len(idx) * 0.6)
    n_val = int(len(idx) * 0.2)
    test_keys = set(uniques[idx[n_train + n_val:]].tolist())
    test_idx = np.where(np.asarray([k in test_keys for k in keys]))[0]
    lab = labels[test_idx]

    patch = np.load(os.path.join(FEAT_DIR, "patch.npy"), mmap_mode="r").astype(np.float32)[test_idx]
    cls = np.load(os.path.join(FEAT_DIR, "cls.npy"), mmap_mode="r").astype(np.float32)[test_idx]

    def rep(name, v):
        v = F.normalize(torch.from_numpy(v), dim=1)
        r, nq = recall(v @ v.T, lab)
        print(f"{name}: queries={nq} R@1={r[1]:.4f} R@5={r[5]:.4f} R@10={r[10]:.4f}")

    rep("DINOv3 CLS", cls)
    rep("DINOv3 patch mean-pool", patch.mean(axis=1))
    p = 3.0
    gem = np.power(np.clip(patch, 0, None).__pow__(p).mean(axis=1), 1.0 / p)
    rep("DINOv3 patch GeM (p=3)", gem)


if __name__ == "__main__":
    main()
