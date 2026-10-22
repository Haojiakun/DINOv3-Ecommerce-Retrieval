import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import TextProjection, VisualHead


ROOT = r"D:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper"
FEAT_DIR = os.path.join(ROOT, "work", "features")
OUT_DIR = os.path.join(ROOT, "outputs")
ATTR_DIMS = ["material", "neckline", "sleeve", "pattern"]
ATTR_LAMBDA = {"material": 1.0, "neckline": 0.5, "sleeve": 0.5, "pattern": 0.5}
SEEDS = [0, 1, 2]
CONFIGS = [(1.0, False, "w1"), (3.0, False, "w3"), (3.0, "all", "w3+all_t07")]


def make_codes(vals):
    codes = np.empty(len(vals), dtype=np.int64)
    seen = {}
    next_code = 0
    for i, v in enumerate(vals):
        v = str(v)
        if v in ("unknown", "none", ""):
            codes[i] = -1 - i
        else:
            if v not in seen:
                seen[v] = next_code
                next_code += 1
            codes[i] = seen[v]
    return codes


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


def attr_infonce(a, b, lab, temperature=0.07):
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    sim = (a @ b.T) / temperature
    eye = torch.eye(a.shape[0], dtype=torch.bool, device=a.device)
    pos = (lab[:, None] == lab[None, :]) & (~eye) & (lab[:, None] >= 0)
    has = pos.sum(dim=1) > 0
    if not has.any():
        return torch.tensor(0.0, device=a.device, requires_grad=True)
    exp_sim = torch.exp(sim)
    denom = (exp_sim * (~eye)).sum(dim=1, keepdim=True)
    num = (exp_sim * pos).sum(dim=1, keepdim=True)
    return -torch.log((num + 1e-8) / (denom + 1e-8))[has].mean()


def local_pool(patch, attr_text, dim=768):
    score = (patch * attr_text.unsqueeze(1)).sum(dim=2) / (dim ** 0.5)
    attn = score.softmax(dim=1)
    return (patch * attn.unsqueeze(2)).sum(dim=1)


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
    n = int(os.environ.get("N_ATTR", "5000"))

    patch = np.load(os.path.join(FEAT_DIR, "patch.npy"), mmap_mode="r").astype(np.float32)[:n]
    cls = np.load(os.path.join(FEAT_DIR, "cls.npy"), mmap_mode="r").astype(np.float32)[:n]
    text = np.load(os.path.join(FEAT_DIR, "text_feats.npy")).astype(np.float32)[:n]
    attr_emb = {d: np.load(os.path.join(FEAT_DIR, f"attr_emb_{d}.npy")).astype(np.float32)[:n] for d in ATTR_DIMS}
    attr_codes = {
        d: make_codes(np.load(os.path.join(FEAT_DIR, f"attr_label_{d}.npy"), allow_pickle=True))
        for d in ATTR_DIMS
    }
    with open(os.path.join(FEAT_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(FEAT_DIR, "filtered_attrs.json"), encoding="utf-8") as f:
        attrs = json.load(f)

    item_keys = np.asarray(meta["item_key"])[:n]
    uniques, labels = np.unique(item_keys, return_inverse=True)
    cat2 = np.asarray([a["category2"] for a in attrs])
    _, cat2_codes = np.unique(cat2, return_inverse=True)
    sleeve_codes = attr_codes["sleeve"]

    # Fixed visual-pretrain split (80/20 by item).
    vrng = np.random.RandomState(0)
    vidx = np.arange(len(uniques))
    vrng.shuffle(vidx)
    v_n_train = int(len(vidx) * 0.8)
    v_train_keys = set(uniques[vidx[:v_n_train]].tolist())
    v_val_mask = np.asarray([k not in v_train_keys for k in item_keys])
    v_train_idx = np.where(~v_val_mask)[0]
    v_val_idx = np.where(v_val_mask)[0]

    # Fixed main split (60/20/20 by item), identical to the paper.
    mrng = np.random.RandomState(0)
    midx = np.arange(len(uniques))
    mrng.shuffle(midx)
    m_n_train = int(len(midx) * 0.6)
    m_n_val = int(len(midx) * 0.2)
    m_train_keys = set(uniques[midx[:m_n_train]].tolist())
    m_val_keys = set(uniques[midx[m_n_train:m_n_train + m_n_val]].tolist())
    m_test_keys = set(uniques[midx[m_n_train + m_n_val:]].tolist())
    m_train_idx = np.where(np.asarray([k in m_train_keys for k in item_keys]))[0]
    m_test_idx = np.where(np.asarray([k in m_test_keys for k in item_keys]))[0]
    print(f"items train/val/test: {len(m_train_keys)}/{len(m_val_keys)}/{len(m_test_keys)}, test images: {len(m_test_idx)}")

    results = {tag: [] for _, _, tag in CONFIGS}

    for seed in SEEDS:
        # Step 1: visual-head pre-training on this seed.
        torch.manual_seed(seed)
        trng = np.random.RandomState(seed)
        model = VisualHead().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for epoch in range(20):
            model.train()
            perm = trng.permutation(v_train_idx)
            for i in range(0, len(perm), 128):
                bi = perm[i : i + 128]
                p = torch.from_numpy(patch[bi]).to(device).float()
                c = torch.from_numpy(cls[bi]).to(device).float()
                lab = torch.from_numpy(labels[bi]).to(device)
                z = model(c, p)
                loss = supcon_loss(z, lab)
                opt.zero_grad()
                loss.backward()
                opt.step()
        vhead_state = model.state_dict()
        torch.save(vhead_state, os.path.join(FEAT_DIR, f"visual_head_seed{seed}.pt"))

        for w, use_local, tag in CONFIGS:
            torch.manual_seed(seed)
            trng = np.random.RandomState(seed)
            model = VisualHead().to(device)
            model.load_state_dict(vhead_state)
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
                    if use_local == "all":
                        for d in ATTR_DIMS:
                            at = attr_projs[d](torch.from_numpy(attr_emb[d][bi]).to(device).float())
                            local = local_pool(p, at)
                            ac = torch.from_numpy(attr_codes[d][bi]).to(device)
                            loss = loss + ATTR_LAMBDA[d] * attr_infonce(local, at, ac, 0.07)
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
            row = (r_t2i[1], r_t2i[5], r_t2i[10], r_i2i[1], r_i2i[10])
            results[tag].append(row)
            print(f"[seed={seed} {tag}] t2i R@1/R@5/R@10={r_t2i[1]:.4f}/{r_t2i[5]:.4f}/{r_t2i[10]:.4f} | i2i R@1/R@10={r_i2i[1]:.4f}/{r_i2i[10]:.4f}")
            torch.save(model.state_dict(), os.path.join(FEAT_DIR, f"attr_align_visual_{tag}_seed{seed}.pt"))
            torch.save(text_proj.state_dict(), os.path.join(FEAT_DIR, f"attr_align_textproj_{tag}_seed{seed}.pt"))

    lines = ["# 3-seed mean±std (fixed 60/20/20 split, DINOv3)", ""]
    for _, _, tag in CONFIGS:
        arr = np.asarray(results[tag]) * 100.0
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        line = (
            f"{tag}: t2i R@1/R@5/R@10 = {mean[0]:.1f}±{std[0]:.1f} / {mean[1]:.1f}±{std[1]:.1f} / "
            f"{mean[2]:.1f}±{std[2]:.1f} | i2i R@1/R@10 = {mean[3]:.1f}±{std[3]:.1f} / {mean[4]:.1f}±{std[4]:.1f}"
        )
        print(line)
        lines.append(line)
    with open(os.path.join(OUT_DIR, "seeds_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("saved", os.path.join(OUT_DIR, "seeds_results.md"))


if __name__ == "__main__":
    main()
