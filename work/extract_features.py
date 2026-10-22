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
    parquet = os.path.join(ROOT, "work", "deepfashion_subset.parquet")
    out_dir = os.path.join(ROOT, "work", "features")
    os.makedirs(out_dir, exist_ok=True)

    model = timm.create_model(
        "vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0
    )
    model = model.to(device).eval()
    data_cfg = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**data_cfg, is_training=False)
    n_prefix = model.num_prefix_tokens

    # Read the full subset parquet (already 5000 rows, stratified sampled)
    table = pq.read_table(parquet)

    image_col = table.column("image")
    id_col = table.column("item_ID")
    cat1_col = table.column("category1")
    cat2_col = table.column("category2")
    cat3_col = table.column("category3")
    color_col = table.column("color")

    ids, keys, cats1, cats2, cats3, colors = [], [], [], [], [], []
    images = []
    for i in range(table.num_rows):
        img_struct = image_col[i].as_py()
        images.append(Image.open(io.BytesIO(img_struct["bytes"])).convert("RGB"))
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
        "category1": cats1,
        "category2": cats2,
        "category3": cats3,
        "color": colors,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print(f"cls shape: {cls.shape}, patch shape: {patches.shape}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
