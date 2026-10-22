"""Extract DINOv3 features from Fashion-Gen parquet.

Adapted from extract_features.py for the Fashion-Gen dataset schema:
  - item_id (productID, already the same-item key)
  - category, subcategory, gender, brand, name, description, pose
  - no explicit color field; color is embedded in `name`

Output goes to work/features_fg/ so DeepFashion features are untouched.

Usage:
    work\venv\Scripts\python.exe work\extract_features_fg.py
    N_IMAGES=10000 work\venv\Scripts\python.exe work\extract_features_fg.py
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
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

import timm
from timm.data import create_transform, resolve_data_config


def main():
    import sys
    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    n_images = int(os.environ.get("N_IMAGES", "10000"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    parquet = os.path.join(ROOT, "work", "fashiongen_train.parquet")
    out_dir = os.path.join(ROOT, "work", "features_fg")
    os.makedirs(out_dir, exist_ok=True)

    model = timm.create_model(
        "vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0
    )
    model = model.to(device).eval()
    data_cfg = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**data_cfg, is_training=False)
    n_prefix = model.num_prefix_tokens

    # Read up to n_images rows across row groups (no pandas dependency)
    pf = pq.ParquetFile(parquet)
    tables = []
    remaining = n_images
    for rg_idx in range(pf.num_row_groups):
        if remaining <= 0:
            break
        tbl = pf.read_row_group(rg_idx)
        take = min(remaining, tbl.num_rows)
        tables.append(tbl.slice(0, take))
        remaining -= take
    table = pa.concat_tables(tables)

    image_col = table.column("image")
    id_col = table.column("item_id")
    cat_col = table.column("category")
    subcat_col = table.column("subcategory")
    gender_col = table.column("gender")
    brand_col = table.column("brand")
    name_col = table.column("name")
    desc_col = table.column("description")
    pose_col = table.column("pose")

    ids, keys, cats, subcats, genders, brands, names, descs, poses = [], [], [], [], [], [], [], [], []
    images = []
    for i in range(table.num_rows):
        img_struct = image_col[i].as_py()
        images.append(Image.open(io.BytesIO(img_struct["bytes"])).convert("RGB"))
        item_id = id_col[i].as_py()
        ids.append(item_id)
        keys.append(item_id)  # productID is already the same-item key
        cats.append(str(cat_col[i].as_py()) or "")
        subcats.append(str(subcat_col[i].as_py()) or "")
        genders.append(str(gender_col[i].as_py()) or "")
        brands.append(str(brand_col[i].as_py()) or "")
        names.append(str(name_col[i].as_py()) or "")
        descs.append(str(desc_col[i].as_py()) or "")
        poses.append(str(pose_col[i].as_py()) or "")

    batch_size = 64
    cls_all, patch_all = [], []
    with torch.inference_mode():
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            x = torch.stack([transform(img) for img in chunk]).to(device)
            feats = model.forward_features(x)
            cls_all.append(feats[:, 0].float().cpu())
            patch_all.append(feats[:, n_prefix:].float().cpu())
            if (i // batch_size) % 10 == 0:
                print(f"processed {min(i + batch_size, len(images))}/{len(images)}")

    cls = torch.cat(cls_all).numpy().astype(np.float16)
    patches = torch.cat(patch_all).numpy().astype(np.float16)
    np.save(os.path.join(out_dir, "cls.npy"), cls)
    np.save(os.path.join(out_dir, "patch.npy"), patches)

    meta = {
        "item_id": ids,
        "item_key": keys,
        "category": cats,
        "subcategory": subcats,
        "gender": genders,
        "brand": brands,
        "name": names,
        "description": descs,
        "pose": poses,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print(f"cls shape: {cls.shape}, patch shape: {patches.shape}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
