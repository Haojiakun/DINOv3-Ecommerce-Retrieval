"""Cloud GPU version: batch-level InfoNCE, batch_size=256, skip VQA, no checkpoint.

Usage:
    N=5000 python3 run_cloud.py     # 5K balanced subset
    N=52591 python3 run_cloud.py    # full dataset
    SEEDS=3 EPOCHS=50 python3 run_cloud.py
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import TextProjection, VisualHead

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(ROOT, "work", os.environ.get("FEAT_DIR", "features"))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = list(range(int(os.environ.get("SEEDS", "3"))))
EPOCHS = int(os.environ.get("EPOCHS", "50"))
BATCH = int(os.environ.get("BATCH", "256"))
LR = float(os.environ.get("LR", "1e-3"))
TEMP = float(os.environ.get("TEMP", "0.07"))
N = int(os.environ.get("N", "5000"))  # number of images


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


def batch_infonce(v, t, temperature=TEMP):
    """Batch-level InfoNCE: positives are on the diagonal within the batch."""
    v = F.normalize(v, dim=1)
    t = F.normalize(t, dim=1)
    logits = (v @ t.T) / temperature
    gt = torch.arange(v.shape[0], device=v.device)
    return F.cross_entropy(logits, gt) + F.cross_entropy(logits.T, gt)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}, N={N}, batch={BATCH}, epochs={EPOCHS}, seeds={SEEDS}")

    # Load patch into RAM as float16 (20GB, fits in 64GB) for fast random access
    patch = np.load(os.path.join(FEAT_DIR, "patch.npy"))  # already float16, no copy
    cls = np.load(os.path.join(FEAT_DIR, "cls.npy"))      # already float16
    text = np.load(os.path.join(FEAT_DIR, "text_feats.npy")).astype(np.float32)[:N]
    print(f"[mem] patch={patch.nbytes/1e9:.1f}GB cls={cls.nbytes/1e9:.1f}GB text={text.nbytes/1e9:.1f}GB")

    with open(os.path.join(FEAT_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    item_keys = np.asarray(meta["item_key"])[:N]
    uniques, labels = np.unique(item_keys, return_inverse=True)
    print(f"[data] {len(item_keys)} images, {len(uniques)} unique products")

    # 60/20/20 split by product
    mrng = np.random.RandomState(0)
    midx = np.arange(len(uniques))
    mrng.shuffle(midx)
    n_train = int(len(midx) * 0.6)
    n_val = int(len(midx) * 0.2)
    train_keys = set(uniques[midx[:n_train]].tolist())
    test_keys = set(uniques[midx[n_train + n_val:]].tolist())
    train_idx = np.where(np.asarray([k in train_keys for k in item_keys]))[0]
    test_idx = np.where(np.asarray([k in test_keys for k in item_keys]))[0]
    print(f"[split] train={len(train_idx)} imgs ({len(train_keys)} prods), test={len(test_idx)} imgs ({len(test_keys)} prods)")

    all_results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Train from scratch (no checkpoint)
        model = VisualHead().to(device)
        text_proj = TextProjection().to(device)
        params = list(model.parameters()) + list(text_proj.parameters())
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
        warmup_epochs = 5
        total_steps = len(range(0, len(train_idx), BATCH)) * EPOCHS
        step = 0

        for epoch in range(EPOCHS):
            model.train()
            text_proj.train()
            perm = np.random.permutation(train_idx)
            total_loss = 0.0
            n_batches = 0
            n_skip = 0
            for i in range(0, len(perm), BATCH):
                bi = perm[i : i + BATCH]
                if len(bi) < 2:
                    continue
                p = torch.from_numpy(patch[bi]).to(device, non_blocking=True).float()
                c = torch.from_numpy(cls[bi]).to(device, non_blocking=True).float()
                x = torch.from_numpy(text[bi]).to(device, non_blocking=True).float()
                v = model(c, p)
                t = text_proj(x)
                loss = batch_infonce(v, t)
                if not torch.isfinite(loss):
                    n_skip += 1
                    continue
                # LR warmup
                step += 1
                warmup_steps = warmup_epochs * max(len(range(0, len(train_idx), BATCH)), 1)
                lr_scale = min(1.0, step / max(warmup_steps, 1))
                for pg in opt.param_groups:
                    pg["lr"] = LR * lr_scale
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                total_loss += loss.item()
                n_batches += 1
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  [seed={seed} epoch={epoch+1}/{EPOCHS}] loss={total_loss/max(n_batches,1):.4f} skip={n_skip}")

        # Eval
        model.eval()
        text_proj.eval()
        Vs, Ts = [], []
        with torch.inference_mode():
            for i in range(0, len(test_idx), BATCH):
                bi = test_idx[i : i + BATCH]
                p = torch.from_numpy(patch[bi]).to(device).float()
                c = torch.from_numpy(cls[bi]).to(device).float()
                x = torch.from_numpy(text[bi]).to(device).float()
                Vs.append(F.normalize(model(c, p), dim=1).cpu())
                Ts.append(F.normalize(text_proj(x), dim=1).cpu())
        V = torch.cat(Vs)
        T = torch.cat(Ts)
        r_t2i, nq_t = recall(T @ V.T, labels[test_idx], exclude_self=False)
        r_i2i, nq_i = recall(V @ V.T, labels[test_idx], exclude_self=True)
        row = (r_t2i[1], r_t2i[5], r_t2i[10], r_i2i[1], r_i2i[10])
        all_results.append(row)
        print(f"[seed={seed}] t2i R@1/5/10={r_t2i[1]:.4f}/{r_t2i[5]:.4f}/{r_t2i[10]:.4f} "
              f"| i2i R@1/10={r_i2i[1]:.4f}/{r_i2i[10]:.4f} (nq_t={nq_t}, nq_i={nq_i})")

    arr = np.asarray(all_results) * 100.0
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    line = (
        f"N={N}: t2i R@1/R@5/R@10 = {mean[0]:.1f}+/-{std[0]:.1f} / {mean[1]:.1f}+/-{std[1]:.1f} / "
        f"{mean[2]:.1f}+/-{std[2]:.1f} | i2i R@1/R@10 = {mean[3]:.1f}+/-{std[3]:.1f} / {mean[4]:.1f}+/-{std[4]:.1f}"
    )
    print(f"\n=== RESULT (3-seed mean+/-std) ===\n{line}")

    tag = f"N{N}_bs{BATCH}_ep{EPOCHS}_lr{LR}_tau{TEMP}"
    with open(os.path.join(OUT_DIR, f"cloud_results_{tag}.md"), "w", encoding="utf-8") as f:
        f.write(f"# Cloud results: N={N}, batch={BATCH}, epochs={EPOCHS}, lr={LR}, tau={TEMP}, seeds={SEEDS}\n\n")
        for i, r in enumerate(all_results):
            f.write(f"- seed {i}: t2i {r[0]*100:.1f}/{r[1]*100:.1f}/{r[2]*100:.1f} | i2i {r[3]*100:.1f}/{r[4]*100:.1f}\n")
        f.write(f"\n{line}\n")
    print(f"saved {os.path.join(OUT_DIR, f'cloud_results_{tag}.md')}")


if __name__ == "__main__":
    main()
