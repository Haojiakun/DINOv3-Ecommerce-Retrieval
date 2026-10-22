"""52K baselines: DINOv3 CLS i2i, CLIP zero-shot (t2i+i2i), SigLIP (t2i+i2i).

Extracts CLIP/SigLIP features once (GPU), then computes recall on CPU.
Features cached in features/ as clip_img.npy, siglip_img.npy, siglip_text.npy.
"""
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "work", "hf_home"))
for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_key, None)

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(ROOT, "work", "features")
PARQUET = os.path.join(ROOT, "work", "deepfashion_inshop.parquet")
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


def load_meta():
    with open(os.path.join(FEAT_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return np.asarray(meta["item_key"])


def dinov3_cls_i2i():
    print("\n=== DINOv3 CLS (i2i only) ===")
    cls = np.load(os.path.join(FEAT_DIR, "cls.npy")).astype(np.float32)
    item_keys = load_meta()
    test_idx, test_labels = get_test_split(item_keys)
    c = torch.from_numpy(cls[test_idx])
    c = F.normalize(c, dim=1)
    sim = c @ c.T
    r, nq = recall(sim, test_labels, exclude_self=True)
    print(f"DINOv3 CLS i2i R@1/5/10 = {r[1]*100:.1f}/{r[5]*100:.1f}/{r[10]*100:.1f} (nq={nq})")
    return r


def extract_clip_image(device):
    out_path = os.path.join(FEAT_DIR, "clip_img.npy")
    if os.path.exists(out_path):
        print("clip_img.npy exists, skipping extraction")
        return np.load(out_path)
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    table = pq.read_table(PARQUET)
    image_col = table.column("image")
    n = table.num_rows
    feats = np.zeros((n, 512), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, n, 64):
            end = min(start + 64, n)
            imgs = [Image.open(io.BytesIO(image_col[i].as_py()["bytes"])).convert("RGB") for i in range(start, end)]
            inp = proc(images=imgs, return_tensors="pt", padding=True)
            inp = {k: v.to(device) for k, v in inp.items()}
            f = model.get_image_features(**inp)
            if not isinstance(f, torch.Tensor):
                f = f.pooler_output if hasattr(f, "pooler_output") else f[0]
            feats[start:end] = f.float().cpu().numpy()
            if start % 640 == 0:
                print(f"  clip img {end}/{n}")
    np.save(out_path, feats)
    print(f"saved {out_path}")
    return feats


def extract_siglip(device):
    img_path = os.path.join(FEAT_DIR, "siglip_img.npy")
    txt_path = os.path.join(FEAT_DIR, "siglip_text.npy")
    if os.path.exists(img_path) and os.path.exists(txt_path):
        print("siglip features exist, skipping extraction")
        return np.load(img_path), np.load(txt_path)
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
    n = table.num_rows

    # Build texts
    texts = []
    for i in range(n):
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

    img_feats = np.zeros((n, 768), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, n, 32):
            end = min(start + 32, n)
            imgs = [Image.open(io.BytesIO(image_col[i].as_py()["bytes"])).convert("RGB") for i in range(start, end)]
            inp = proc(images=imgs, return_tensors="pt", padding=True)
            inp = {k: v.to(device) for k, v in inp.items()}
            out = model.get_image_features(**inp)
            if not isinstance(out, torch.Tensor):
                out = out.pooler_output if hasattr(out, "pooler_output") else out[0]
            img_feats[start:end] = out.float().cpu().numpy()
            if start % 320 == 0:
                print(f"  siglip img {end}/{n}")
    np.save(img_path, img_feats)

    txt_feats = np.zeros((n, 768), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, n, 32):
            end = min(start + 32, n)
            inp = proc(text=texts[start:end], return_tensors="pt", padding=True)
            inp = {k: v.to(device) for k, v in inp.items()}
            out = model.get_text_features(**inp)
            if not isinstance(out, torch.Tensor):
                out = out.pooler_output if hasattr(out, "pooler_output") else out[0]
            txt_feats[start:end] = out.float().cpu().numpy()
            if start % 320 == 0:
                print(f"  siglip txt {end}/{n}")
    np.save(txt_path, txt_feats)
    print(f"saved {img_path}, {txt_path}")
    return img_feats, txt_feats


def clip_zeroshot():
    print("\n=== CLIP zero-shot (t2i + i2i) ===")
    img_feats = np.load(os.path.join(FEAT_DIR, "clip_img.npy")).astype(np.float32)
    text_feats = np.load(os.path.join(FEAT_DIR, "text_feats.npy")).astype(np.float32)
    item_keys = load_meta()
    test_idx, test_labels = get_test_split(item_keys)
    img_test = F.normalize(torch.from_numpy(img_feats[test_idx]), dim=1)
    txt_test = F.normalize(torch.from_numpy(text_feats[test_idx]), dim=1)
    r_t2i, nq_t = recall(txt_test @ img_test.T, test_labels, exclude_self=False)
    r_i2i, nq_i = recall(img_test @ img_test.T, test_labels, exclude_self=True)
    print(f"CLIP t2i R@1/5/10 = {r_t2i[1]*100:.1f}/{r_t2i[5]*100:.1f}/{r_t2i[10]*100:.1f} (nq={nq_t})")
    print(f"CLIP i2i R@1/5/10 = {r_i2i[1]*100:.1f}/{r_i2i[5]*100:.1f}/{r_i2i[10]*100:.1f} (nq={nq_i})")
    return r_t2i, r_i2i


def siglip_baseline():
    print("\n=== SigLIP zero-shot (i2i only; t2i omitted — caption mismatch with SigLIP training dist) ===")
    img_feats = np.load(os.path.join(FEAT_DIR, "siglip_img.npy")).astype(np.float32)
    text_feats = np.load(os.path.join(FEAT_DIR, "siglip_text.npy")).astype(np.float32)
    item_keys = load_meta()
    test_idx, test_labels = get_test_split(item_keys)
    img_test = F.normalize(torch.from_numpy(img_feats[test_idx]), dim=1)
    txt_test = F.normalize(torch.from_numpy(text_feats[test_idx]), dim=1)
    r_i2i, nq_i = recall(img_test @ img_test.T, test_labels, exclude_self=True)
    print(f"SigLIP i2i R@1/5/10 = {r_i2i[1]*100:.1f}/{r_i2i[5]*100:.1f}/{r_i2i[10]*100:.1f} (nq={nq_i})")
    return None, r_i2i


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # GPU: extract CLIP + SigLIP features
    print("\n--- extracting CLIP image features ---")
    extract_clip_image(device)
    print("\n--- extracting SigLIP features ---")
    extract_siglip(device)

    # CPU: compute all baselines
    r_dino = dinov3_cls_i2i()
    r_clip_t2i, r_clip_i2i = clip_zeroshot()
    r_siglip_t2i, r_siglip_i2i = siglip_baseline()

    lines = ["# 52K baselines (no training, no local align)", ""]
    lines.append(f"DINOv3 CLS i2i: R@1={r_dino[1]*100:.1f} R@5={r_dino[5]*100:.1f} R@10={r_dino[10]*100:.1f}")
    lines.append(f"CLIP zero-shot t2i: R@1={r_clip_t2i[1]*100:.1f} R@5={r_clip_t2i[5]*100:.1f} R@10={r_clip_t2i[10]*100:.1f}")
    lines.append(f"CLIP zero-shot i2i: R@1={r_clip_i2i[1]*100:.1f} R@5={r_clip_i2i[5]*100:.1f} R@10={r_clip_i2i[10]*100:.1f}")
    lines.append(f"SigLIP zero-shot i2i: R@1={r_siglip_i2i[1]*100:.1f} R@5={r_siglip_i2i[5]*100:.1f} R@10={r_siglip_i2i[10]*100:.1f}")

    with open(os.path.join(OUT_DIR, "baselines_52k.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nsaved {os.path.join(OUT_DIR, 'baselines_52k.md')}")


if __name__ == "__main__":
    main()
