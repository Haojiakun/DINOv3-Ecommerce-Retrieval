import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "work", "hf_home"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_key, None)

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import CLIPConfig, CLIPModel, CLIPProcessor


def _get_embedding(out, model, proj_attr):
    # transformers >=5: get_text_features / get_image_features return the
    # backbone output object.  The pooler_output may already be projected
    # (image side, 512-d) or may still need the projection layer (text side,
    # 512-d in, 512-d out).  Apply the projection only when dimensions match.
    if torch.is_tensor(out):
        return out
    pooled = out.pooler_output
    proj = getattr(model, proj_attr)
    if pooled.shape[-1] == proj.in_features:
        return proj(pooled)
    return pooled


def recall_at_k(sim, keys, ks=(1, 5, 10), exclude_self=True):
    sim = sim.float()
    topk = sim.topk(max(ks) + 1, dim=1).indices.cpu().numpy()[:, 1:]
    hits = {k: [] for k in ks}
    n_q = 0
    for i, key in enumerate(keys):
        pos = {j for j, k2 in enumerate(keys) if k2 == key and (not exclude_self or j != i)}
        if not pos:
            continue
        n_q += 1
        for k in ks:
            hits[k].append(bool(pos.intersection(topk[i, :k].tolist())))
    return {k: float(np.mean(hits[k])) for k in ks}, n_q


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_dir = os.path.join(ROOT, "work", "features")
    parquet = os.path.join(ROOT, "work", "deepfashion_inshop.parquet")

    # transformers 4.57 refuses torch.load below torch 2.6; load the weights
    # ourselves with weights_only=True instead (safe: state dict is tensors only).
    cfg = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel(cfg).to(device).eval()
    weight_file = os.path.join(
        ROOT, "work", "hf_home", "hub", "models--openai--clip-vit-base-patch32",
        "snapshots", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", "pytorch_model.bin",
    )
    state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Read first 5000 rows across row groups (pyarrow native, no pandas).
    pf = pq.ParquetFile(parquet)
    tables, remaining = [], 5000
    for rg in range(pf.num_row_groups):
        if remaining <= 0:
            break
        tbl = pf.read_row_group(rg, columns=["image", "text", "item_ID"])
        take = min(remaining, tbl.num_rows)
        tables.append(tbl.slice(0, take))
        remaining -= take
    table = pa.concat_tables(tables)

    image_col = table.column("image")
    text_col = table.column("text")
    id_col = table.column("item_ID")
    images, captions, keys = [], [], []
    for i in range(table.num_rows):
        img_struct = image_col[i].as_py()
        images.append(Image.open(io.BytesIO(img_struct["bytes"])).convert("RGB"))
        t = text_col[i].as_py()
        captions.append(str(t) if t else "")
        item_id = str(id_col[i].as_py())
        p = item_id.split("_")
        keys.append("_".join(p[:-2]) if len(p) >= 3 else item_id)

    # Text features.
    text_feats = []
    bs = 128
    with torch.inference_mode():
        for i in range(0, len(captions), bs):
            inp = proc(text=captions[i : i + bs], return_tensors="pt", padding=True, truncation=True)
            t = model.get_text_features(input_ids=inp["input_ids"].to(device), attention_mask=inp["attention_mask"].to(device))
            text_feats.append(_get_embedding(t, model, "text_projection").float().cpu())
    text_feats = torch.cat(text_feats)

    # Image features.
    img_feats = []
    with torch.inference_mode():
        for i in range(0, len(images), bs):
            inp = proc(images=images[i : i + bs], return_tensors="pt")
            v = model.get_image_features(pixel_values=inp["pixel_values"].to(device))
            img_feats.append(_get_embedding(v, model, "visual_projection").float().cpu())
    img_feats = torch.cat(img_feats)

    text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)
    img_feats = img_feats / img_feats.norm(dim=1, keepdim=True)

    dino_cls = torch.from_numpy(np.load(os.path.join(feat_dir, "cls.npy")).astype(np.float32))
    dino_cls = dino_cls / dino_cls.norm(dim=1, keepdim=True)

    # Naive concat: DINOv3 CLS (768) || CLIP text (512), re-normalized.
    concat = torch.cat([dino_cls, text_feats], dim=1)
    concat = concat / concat.norm(dim=1, keepdim=True)

    def report(name, s, exclude_self=True):
        r, nq = recall_at_k(s, keys, exclude_self=exclude_self)
        print(f"{name}: queries={nq} R@1={r[1]:.4f} R@5={r[5]:.4f} R@10={r[10]:.4f}")

    report("DINOv3 CLS image->image", dino_cls @ dino_cls.T)
    report("DINOv3 CLS + CLIP text (concat) image->image", concat @ concat.T)
    report("CLIP native image->image", img_feats @ img_feats.T)
    report("CLIP native text->image", text_feats @ img_feats.T, exclude_self=False)

    np.save(os.path.join(feat_dir, "clip_text.npy"), text_feats.numpy().astype(np.float16))
    np.save(os.path.join(feat_dir, "clip_img.npy"), img_feats.numpy().astype(np.float16))
    print("saved clip_text.npy and clip_img.npy")


if __name__ == "__main__":
    main()
