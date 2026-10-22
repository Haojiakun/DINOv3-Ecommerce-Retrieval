import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import TextProjection, VisualHead


FEAT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features")
ATTR_DIMS = ["material", "neckline", "sleeve", "pattern"]
ATTR_LAMBDA = {"material": 1.0, "neckline": 0.5, "sleeve": 0.5, "pattern": 0.5}


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    set_seed(0)
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

    rng = np.random.RandomState(0)
    idx = np.arange(len(uniques))
    rng.shuffle(idx)
    n_train = int(len(idx) * 0.6)
    n_val = int(len(idx) * 0.2)
    train_keys = set(uniques[idx[:n_train]].tolist())
    val_keys = set(uniques[idx[n_train:n_train + n_val]].tolist())
    test_keys = set(uniques[idx[n_train + n_val:]].tolist())
    train_idx = np.where(np.asarray([k in train_keys for k in item_keys]))[0]
    val_idx = np.where(np.asarray([k in val_keys for k in item_keys]))[0]
    test_idx = np.where(np.asarray([k in test_keys for k in item_keys]))[0]
    print(f"items train/val/test: {len(train_keys)}/{len(val_keys)}/{len(test_keys)}, images: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    visual = torch.load(os.path.join(FEAT_DIR, "visual_head.pt"), map_location="cpu")

    def run(w, use_local, tag, epochs=12, warmup_epochs=4, temperature=0.07):
        model = VisualHead().to(device)
        model.load_state_dict(visual)
        text_proj = TextProjection().to(device)
        attr_projs = nn.ModuleDict({d: nn.Linear(512, 768).to(device) for d in ATTR_DIMS})
        params = list(model.parameters()) + list(text_proj.parameters()) + list(attr_projs.parameters())
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
        batch_size = 128
        for epoch in range(epochs):
            model.train()
            text_proj.train()
            attr_projs.train()
            eff_w = 1.0 + (w - 1.0) * min(1.0, epoch / max(1, warmup_epochs))
            perm = rng.permutation(train_idx)
            for i in range(0, len(perm), batch_size):
                bi = perm[i : i + batch_size]
                p = torch.from_numpy(patch[bi]).to(device).float()
                c = torch.from_numpy(cls[bi]).to(device).float()
                x = torch.from_numpy(text[bi]).to(device).float()
                lab = torch.from_numpy(labels[bi]).to(device)
                c2 = torch.from_numpy(cat2_codes[bi]).to(device)
                sl = torch.from_numpy(sleeve_codes[bi]).to(device)
                v = model(c, p)
                t = text_proj(x)
                loss = hard_infonce(v, t, lab, c2, sl, eff_w, temperature)
                if use_local == "all":
                    for d in ATTR_DIMS:
                        at = attr_projs[d](torch.from_numpy(attr_emb[d][bi]).to(device).float())
                        local = local_pool(p, at)
                        ac = torch.from_numpy(attr_codes[d][bi]).to(device)
                        loss = loss + ATTR_LAMBDA[d] * attr_infonce(local, at, ac, temperature)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
        model.eval()
        text_proj.eval()
        Vs, Ts = [], []
        with torch.inference_mode():
            for i in range(0, len(test_idx), 128):
                bi = test_idx[i : i + 128]
                p = torch.from_numpy(patch[bi]).to(device).float()
                c = torch.from_numpy(cls[bi]).to(device).float()
                x = torch.from_numpy(text[bi]).to(device).float()
                Vs.append(F.normalize(model(c, p), dim=1).cpu())
                Ts.append(F.normalize(text_proj(x), dim=1).cpu())
        V = torch.cat(Vs)
        T = torch.cat(Ts)
        r_t2i, _ = recall(T @ V.T, labels[test_idx], exclude_self=False)
        r_i2i, _ = recall(V @ V.T, labels[test_idx], exclude_self=True)
        print(f"[{tag}] text->image R@1={r_t2i[1]:.4f} R@5={r_t2i[5]:.4f} R@10={r_t2i[10]:.4f} | image->image R@1={r_i2i[1]:.4f} R@10={r_i2i[10]:.4f}")
        torch.save(model.state_dict(), os.path.join(FEAT_DIR, f"attr_align_visual_{tag}.pt"))
        torch.save(text_proj.state_dict(), os.path.join(FEAT_DIR, f"attr_align_textproj_{tag}.pt"))
        return V, T

    run(1.0, False, "w1", temperature=0.07)
    run(3.0, False, "w3", temperature=0.07)
    run(3.0, "all", "w3+all_t07", temperature=0.07)
    V, T = run(3.0, "all", "w3+all_t10", temperature=0.1)
    np.save(os.path.join(FEAT_DIR, "final_visual_test.npy"), V.numpy().astype(np.float16))
    np.save(os.path.join(FEAT_DIR, "final_text_test.npy"), T.numpy().astype(np.float16))


if __name__ == "__main__":
    main()
