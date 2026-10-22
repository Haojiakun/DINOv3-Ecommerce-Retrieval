import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import TextProjection, VisualHead
from train_attr_align import ATTR_DIMS, ATTR_LAMBDA, attr_infonce, hard_infonce, local_pool, make_codes, recall


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "work", "hf_home"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_key, None)

import pyarrow.parquet as pq
from transformers import CLIPConfig, CLIPModel, CLIPProcessor


FEAT_DIR = os.path.join(ROOT, "work", "features")


def build_dirty():
    raw = json.load(open(os.path.join(FEAT_DIR, "vqa_attrs.json"), encoding="utf-8"))
    cats = pq.read_table(os.path.join(ROOT, "work", "deepfashion_inshop.parquet"), columns=["category2"]).to_pandas()
    dirty = []
    for i, rec in enumerate(raw):
        row = {"category2": str(cats.iloc[i]["category2"]).strip().lower() if cats.iloc[i]["category2"] else ""}
        for d in ATTR_DIMS:
            v = rec.get(d, "")
            row[d] = v["answer"].strip().lower() if isinstance(v, dict) else str(v).strip().lower()
        dirty.append(row)
    json.dump(dirty, open(os.path.join(FEAT_DIR, "dirty_attrs.json"), "w", encoding="utf-8"), ensure_ascii=False)

    cfg = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel(cfg).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    model.load_state_dict(torch.load(
        os.path.join(ROOT, "work", "hf_home", "hub", "models--openai--clip-vit-base-patch32",
                     "snapshots", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", "pytorch_model.bin"),
        map_location="cpu", weights_only=True), strict=False)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tpl = {"material": "{} material", "neckline": "{} neckline", "sleeve": "{} sleeve", "pattern": "{} pattern"}
    for d in ATTR_DIMS:
        phrases = [tpl[d].format(x[d]) for x in dirty]
        emb = []
        with torch.inference_mode():
            for i in range(0, len(phrases), 64):
                inp = proc(text=phrases[i : i + 64], return_tensors="pt", padding=True, truncation=True)
                t = model.get_text_features(input_ids=inp["input_ids"].to(model.device), attention_mask=inp["attention_mask"].to(model.device))
                emb.append(t.float().cpu())
        emb = torch.cat(emb)
        np.save(os.path.join(FEAT_DIR, f"dirty_emb_{d}.npy"), emb.numpy().astype(np.float16))
        np.save(os.path.join(FEAT_DIR, f"dirty_label_{d}.npy"), np.asarray([x[d] for x in dirty]))
    print("built dirty attrs + embeddings")


def main():
    if not os.path.exists(os.path.join(FEAT_DIR, "dirty_emb_material.npy")):
        build_dirty()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = 5000
    patch = np.load(os.path.join(FEAT_DIR, "patch.npy"), mmap_mode="r").astype(np.float32)[:n]
    cls = np.load(os.path.join(FEAT_DIR, "cls.npy"), mmap_mode="r").astype(np.float32)[:n]
    text = np.load(os.path.join(FEAT_DIR, "text_feats.npy")).astype(np.float32)[:n]
    emb = {d: np.load(os.path.join(FEAT_DIR, f"dirty_emb_{d}.npy")).astype(np.float32)[:n] for d in ATTR_DIMS}
    lab = {d: make_codes(np.load(os.path.join(FEAT_DIR, f"dirty_label_{d}.npy"), allow_pickle=True)) for d in ATTR_DIMS}
    meta = json.load(open(os.path.join(FEAT_DIR, "meta.json"), encoding="utf-8"))
    attrs = json.load(open(os.path.join(FEAT_DIR, "dirty_attrs.json"), encoding="utf-8"))

    keys = np.asarray(meta["item_key"])[:n]
    uniques, labels = np.unique(keys, return_inverse=True)
    cat2 = np.asarray([a["category2"] for a in attrs])
    _, cat2_codes = np.unique(cat2, return_inverse=True)
    sleeve_codes = lab["sleeve"]
    rng = np.random.RandomState(0)
    idx = np.arange(len(uniques))
    rng.shuffle(idx)
    n_train = int(len(idx) * 0.6)
    n_val = int(len(idx) * 0.2)
    train_keys = set(uniques[idx[:n_train]].tolist())
    test_keys = set(uniques[idx[n_train + n_val:]].tolist())
    train_idx = np.where(np.asarray([k in train_keys for k in keys]))[0]
    test_idx = np.where(np.asarray([k in test_keys for k in keys]))[0]

    visual = torch.load(os.path.join(FEAT_DIR, "visual_head.pt"), map_location="cpu")

    def run(use_local, tag):
        model = VisualHead().to(device)
        model.load_state_dict(visual)
        text_proj = TextProjection().to(device)
        attr_projs = nn.ModuleDict({d: nn.Linear(512, 768).to(device) for d in ATTR_DIMS})
        params = list(model.parameters()) + list(text_proj.parameters()) + list(attr_projs.parameters())
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
        for epoch in range(12):
            model.train()
            text_proj.train()
            attr_projs.train()
            eff_w = 1.0 + 2.0 * min(1.0, epoch / 4.0)
            perm = rng.permutation(train_idx)
            for i in range(0, len(perm), 128):
                bi = perm[i : i + 128]
                p = torch.from_numpy(patch[bi]).to(device).float()
                c = torch.from_numpy(cls[bi]).to(device).float()
                x = torch.from_numpy(text[bi]).to(device).float()
                l = torch.from_numpy(labels[bi]).to(device)
                c2 = torch.from_numpy(cat2_codes[bi]).to(device)
                sl = torch.from_numpy(sleeve_codes[bi]).to(device)
                v = model(c, p)
                t = text_proj(x)
                loss = hard_infonce(v, t, l, c2, sl, eff_w)
                if use_local:
                    for d in ATTR_DIMS:
                        at = attr_projs[d](torch.from_numpy(emb[d][bi]).to(device).float())
                        local = local_pool(p, at)
                        ac = torch.from_numpy(lab[d][bi]).to(device)
                        loss = loss + ATTR_LAMBDA[d] * attr_infonce(local, at, ac)
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
        print(f"[dirty-{tag}] text->image R@1={r_t2i[1]:.4f} R@10={r_t2i[10]:.4f} | image->image R@1={r_i2i[1]:.4f} R@10={r_i2i[10]:.4f}")

    run(False, "w3")
    run(True, "w3+all")


if __name__ == "__main__":
    main()
