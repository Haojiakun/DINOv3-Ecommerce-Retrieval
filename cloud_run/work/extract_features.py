import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "work", "hf_home"))
for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_key, None)

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

import timm
import torch.nn.functional as F
from timm.data import create_transform, resolve_data_config


def item_key(item_id):
    parts = item_id.split("_")
    for i, p in enumerate(parts):
        if p == "id":
            return "_".join(parts[: i + 2])
    return item_id


def main():
    import sys
    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    parquet_name = os.environ.get("PARQUET", "deepfashion_subset.parquet")
    parquet = os.path.join(ROOT, "work", parquet_name)
    out_dir = os.path.join(ROOT, "work", os.environ.get("OUT_DIR", "features"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[extract] parquet={parquet_name} -> {out_dir}")

    model = timm.create_model(
        "vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0
    )
    model = model.to(device).eval()
    data_cfg = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**data_cfg, is_training=False)
    n_prefix = model.num_prefix_tokens

    # Read parquet metadata only (not images) to save memory
    table = pq.read_table(parquet)
    n_total = table.num_rows
    image_col = table.column("image")
    id_col = table.column("item_ID")
    cat1_col = table.column("category1")
    cat2_col = table.column("category2")
    cat3_col = table.column("category3")
    color_col = table.column("color")

    # Pre-collect metadata (small)
    ids, keys, cats1, cats2, cats3, colors = [], [], [], [], [], []
    for i in range(n_total):
        item_id = id_col[i].as_py()
        ids.append(item_id)
        keys.append(item_key(item_id))
        c1 = cat1_col[i].as_py()
        c2 = cat2_col[i].as_py()
        c3 = cat3_col[i].as_py()
        col = color_col[i].as_py()
        cats1.append(str(c1) if c1 else "")
        cats2.append(str(c2) if c2 else "")
        cats3.append(str(c3) if c3 else "")
        colors.append(str(col) if col else "")

    patch_dim = 256  # DINOv3 base has 256 patch tokens
    embed_dim = 768
    batch_size = 64

    # Use open_memmap to create .npy files directly (with header), no np.save needed
    cls_path = os.path.join(out_dir, "cls.npy")
    patch_path = os.path.join(out_dir, "patch.npy")
    # Remove old files if any
    for p in (cls_path, patch_path):
        if os.path.exists(p):
            os.remove(p)
    cls_mm = np.lib.format.open_memmap(cls_path, dtype=np.float16, mode="w+",
                                       shape=(n_total, embed_dim))
    patch_mm = np.lib.format.open_memmap(patch_path, dtype=np.float16, mode="w+",
                                         shape=(n_total, patch_dim, embed_dim))

    # Stream images batch by batch (never load all into memory)
    with torch.inference_mode():
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            imgs = []
            for i in range(start, end):
                img_struct = image_col[i].as_py()
                imgs.append(Image.open(io.BytesIO(img_struct["bytes"])).convert("RGB"))
            x = torch.stack([transform(img) for img in imgs]).to(device)
            feats = model.forward_features(x)
            # L2-normalize before float16 to prevent overflow/NaN
            cls_feat = F.normalize(feats[:, 0].float(), dim=-1)
            patch_feat = F.normalize(feats[:, n_prefix:].float(), dim=-1)
            cls_mm[start:end] = cls_feat.cpu().numpy().astype(np.float16)
            patch_mm[start:end] = patch_feat.cpu().numpy().astype(np.float16)
            del imgs, x, feats
            if start == 0:
                print(f"  [debug] cls[0] norm={float(cls_feat[0].norm()):.4f} (should be ~1.0)")
            if (start // batch_size) % 10 == 0:
                print(f"processed {end}/{n_total}")

    cls_mm.flush()
    patch_mm.flush()
    del cls_mm, patch_mm

    meta = {
        "item_id": ids,
        "item_key": keys,
        "category1": cats1,
        "category2": cats2,
        "category3": cats3,
        "color": colors,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print(f"cls shape: ({n_total}, {embed_dim}), patch shape: ({n_total}, {patch_dim}, {embed_dim})")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
