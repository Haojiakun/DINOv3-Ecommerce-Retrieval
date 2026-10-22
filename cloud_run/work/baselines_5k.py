"""Baselines on 5K balanced subset: DINOv3 CLS i2i, CLIP zero-shot, SigLIP.

Reads features from features_5k/ (DINOv3 cls) and re-encodes images with
CLIP / SigLIP for zero-shot baselines. No training, no local alignment.
"""
import io
import json
import os

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
import timm
from timm.data import create_transform, resolve_data_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(ROOT, "work", "features_5k")
PARQUET = os.path.join(ROOT, "work", "deepfashion_subset.parquet")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


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


def get_test_split(item_keys):
    uniques, labels = np.unique(item_keys, return_inverse=True)
    mrng = np.random.RandomState(0)
    midx = np.arange(len(uniques))
    mrng.shuffle(midx)
    n_train = int(len(midx) * 0.6)
    n_val = int(len(midx) * 0.2)
    test_keys = set(uniques[midx[n_train + n_val:]].tolist())
    test_idx = np.where(np.asarray([k in test_keys for k in item_keys]))[0]
    return test_idx, labels[test_idx]


def dinov3_cls_i2i(device):
    print("\n=== DINOv3 CLS (i2i only) ===")
    cls = np.load(os.path.join(FEAT_DIR, "cls.npy")).astype(np.float32)
    with open(os.path.join(FEAT_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    item_keys = np.asarray(meta["item_key"])
    test_idx, test_labels = get_test_split(item_keys)
    c = torch.from_numpy(cls[test_idx])
    c = F.normalize(c, dim=1)
    sim = c @ c.T
    r, nq = recall(sim, test_labels, exclude_self=True)
    print(f"DINOv3 CLS i2i R@1/5/10 = {r[1]*100:.1f}/{r[5]*100:.1f}/{r[10]*100:.1f} (nq={nq})")
    return r


def clip_zeroshot(device):
    print("\n=== CLIP zero-shot (t2i + i2i) ===")
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Read images
    table = pq.read_table(PARQUET)
    image_col = table.column("image")
    images = []
    for i in range(table.num_rows):
        img_struct = image_col[i].as_py()
        images.append(Image.open(io.BytesIO(img_struct["bytes"])).convert("RGB"))

    # Encode images
    img_feats = []
    with torch.inference_mode():
        for i in range(0, len(images), 64):
            inp = proc(images=images[i:i+64], return_tensors="pt", padding=True)
            inp = {k: v.to(device) for k, v in inp.items()}
            f = model.get_image_features(**inp)
            if not isinstance(f, torch.Tensor):
                f = f.pooler_output if hasattr(f, "pooler_output") else f[0]
            img_feats.append(f.float().cpu())
    img_feats = torch.cat(img_feats)

    # Text features already extracted with same CLIP
    text_feats = torch.from_numpy(np.load(os.path.join(FEAT_DIR, "text_feats.npy")).astype(np.float32))

    with open(os.path.join(FEAT_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    item_keys = np.asarray(meta["item_key"])
    test_idx, test_labels = get_test_split(item_keys)

    img_test = F.normalize(img_feats[test_idx], dim=1)
    txt_test = F.normalize(text_feats[test_idx], dim=1)

    r_t2i, nq_t = recall(txt_test @ img_test.T, test_labels, exclude_self=False)
    r_i2i, nq_i = recall(img_test @ img_test.T, test_labels, exclude_self=True)
    print(f"CLIP t2i R@1/5/10 = {r_t2i[1]*100:.1f}/{r_t2i[5]*100:.1f}/{r_t2i[10]*100:.1f} (nq={nq_t})")
    print(f"CLIP i2i R@1/5/10 = {r_i2i[1]*100:.1f}/{r_i2i[5]*100:.1f}/{r_i2i[10]*100:.1f} (nq={nq_i})")
    return r_t2i, r_i2i


def siglip_baseline(device):
    print("\n=== SigLIP zero-shot (i2i only; t2i omitted — caption mismatch with SigLIP training dist) ===")
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device).eval()
    proc = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

    table = pq.read_table(PARQUET)
    image_col = table.column("image")
    id_col = table.column("item_ID")
    cat1_col = table.column("category1")
    cat2_col = table.column("category2")
    cat3_col = table.column("category3")
    color_col = table.column("color")

    images, texts = [], []
    for i in range(table.num_rows):
        img_struct = image_col[i].as_py()
        images.append(Image.open(io.BytesIO(img_struct["bytes"])).convert("RGB"))
        item_id = str(id_col[i].as_py())
        first = item_id.split("_")[0].lower()
        gender = {"men": "men's", "women": "women's"}.get(first, "")
        parts = []
        for col in [cat1_col, cat2_col, cat3_col]:
            v = col[i].as_py()
            if v:
                parts.append(str(v).strip().lower().replace("_", " "))
        color = color_col[i].as_py()
        color = str(color).strip().lower() if color else ""
        base = "a photo of " + (gender + " " if gender else "") + " ".join(parts)
        if color:
            base += ", color " + color
        texts.append(base.strip().rstrip(".") + ".")

    # Encode images
    img_feats = []
    with torch.inference_mode():
        for i in range(0, len(images), 32):
            inp = proc(images=images[i:i+32], return_tensors="pt", padding=True)
            inp = {k: v.to(device) for k, v in inp.items()}
            out = model.get_image_features(**inp)
            if not isinstance(out, torch.Tensor):
                out = out.pooler_output if hasattr(out, "pooler_output") else out[0]
            img_feats.append(out.float().cpu())
    img_feats = torch.cat(img_feats)

    # Encode texts
    txt_feats = []
    with torch.inference_mode():
        for i in range(0, len(texts), 32):
            inp = proc(text=texts[i:i+32], return_tensors="pt", padding=True)
            inp = {k: v.to(device) for k, v in inp.items()}
            out = model.get_text_features(**inp)
            if not isinstance(out, torch.Tensor):
                out = out.pooler_output if hasattr(out, "pooler_output") else out[0]
            txt_feats.append(out.float().cpu())
    txt_feats = torch.cat(txt_feats)

    with open(os.path.join(FEAT_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    item_keys = np.asarray(meta["item_key"])
    test_idx, test_labels = get_test_split(item_keys)

    img_test = F.normalize(img_feats[test_idx], dim=1)
    txt_test = F.normalize(txt_feats[test_idx], dim=1)

    r_t2i, nq_t = recall(txt_test @ img_test.T, test_labels, exclude_self=False)
    r_i2i, nq_i = recall(img_test @ img_test.T, test_labels, exclude_self=True)
    print(f"SigLIP i2i R@1/5/10 = {r_i2i[1]*100:.1f}/{r_i2i[5]*100:.1f}/{r_i2i[10]*100:.1f} (nq={nq_i})")
    return r_t2i, r_i2i


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    r_dino = dinov3_cls_i2i(device)
    r_clip_t2i, r_clip_i2i = clip_zeroshot(device)
    r_siglip_t2i, r_siglip_i2i = siglip_baseline(device)

    lines = ["# Baselines on 5K balanced subset (no training, no local align)", ""]
    lines.append(f"DINOv3 CLS i2i: R@1={r_dino[1]*100:.1f} R@5={r_dino[5]*100:.1f} R@10={r_dino[10]*100:.1f}")
    lines.append(f"CLIP zero-shot t2i: R@1={r_clip_t2i[1]*100:.1f} R@5={r_clip_t2i[5]*100:.1f} R@10={r_clip_t2i[10]*100:.1f}")
    lines.append(f"CLIP zero-shot i2i: R@1={r_clip_i2i[1]*100:.1f} R@5={r_clip_i2i[5]*100:.1f} R@10={r_clip_i2i[10]*100:.1f}")
    lines.append(f"SigLIP zero-shot i2i: R@1={r_siglip_i2i[1]*100:.1f} R@5={r_siglip_i2i[5]*100:.1f} R@10={r_siglip_i2i[10]*100:.1f}")

    with open(os.path.join(OUT_DIR, "baselines_5k.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nsaved {os.path.join(OUT_DIR, 'baselines_5k.md')}")


if __name__ == "__main__":
    main()
