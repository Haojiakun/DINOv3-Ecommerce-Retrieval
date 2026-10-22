"""Run a frozen DINOv3 backbone on DeepFashion images and inspect features.

This is the "first step" script: it loads the pretrained ViT-B/16 DINOv3,
feeds it a few real DeepFashion In-shop images, and prints the shape of the
CLS and patch tokens that the attribute-guided alignment will consume.

Run from the repo root with the prepared virtualenv:
    work\\venv\\Scripts\\python.exe outputs\\run_dinov3.py
"""

import io
import os

# The host cannot reach huggingface.co directly; use the mirror and keep the
# model cache inside the workspace instead of the (non-writable) home dir.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "work", "hf_home"))
# A stale sandbox proxy (127.0.0.1:9) can break Hugging Face downloads.
# Clear it so huggingface_hub reaches the mirror directly.
for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_key, None)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import timm
from timm.data import create_transform, resolve_data_config


def item_key(item_id):
    # "MEN_Denim_id_00000080_01_1_front" -> "MEN_Denim_id_00000080_01".
    parts = item_id.split("_")
    return "_".join(parts[:-2]) if len(parts) >= 3 else item_id


def load_real_images(parquet_path, limit=16):
    import pyarrow.parquet as pq

    # Read with pyarrow directly so we don't need pandas installed.
    table = pq.ParquetFile(parquet_path).read_row_group(0).slice(0, limit)
    image_col = table.column("image")
    id_col = table.column("item_ID")
    images, ids = [], []
    for i in range(table.num_rows):
        img_struct = image_col[i].as_py()
        images.append(Image.open(io.BytesIO(img_struct["bytes"])).convert("RGB"))
        ids.append(id_col[i].as_py())
    return images, ids


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(
        "vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0
    )
    model = model.to(device).eval()

    data_cfg = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**data_cfg, is_training=False)
    print("model:", "vit_base_patch16_dinov3.lvd1689m")
    print("device:", device)
    print("data_config:", data_cfg)
    print("embed_dim:", model.embed_dim)
    print("num_prefix_tokens (CLS + registers):", model.num_prefix_tokens)

    parquet = os.path.join(ROOT, "work", "deepfashion_inshop.parquet")
    if os.path.exists(parquet):
        images, ids = load_real_images(parquet, limit=16)
        print("input:", f"{len(images)} real DeepFashion In-shop images")
    else:
        images = [Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))]
        ids = ["random"]
        print("input: 1 random image (parquet not found)")

    batch = torch.stack([transform(img) for img in images]).to(device)
    with torch.inference_mode():
        feats = model.forward_features(batch)

    n_prefix = model.num_prefix_tokens
    cls = feats[:, 0]             # (B, D)
    patches = feats[:, n_prefix:]  # (B, H*W, D)
    print("CLS token shape:", tuple(cls.shape))
    print("patch tokens shape:", tuple(patches.shape))

    # Quick sanity check: same product, different views should cluster closer
    # than different products under CLS cosine similarity.
    cls = F.normalize(cls, dim=-1)
    sim = (cls @ cls.T).cpu().numpy()
    groups = {}
    for i, item_id in enumerate(ids):
        groups.setdefault(item_key(item_id), []).append(i)
    print("within-item vs cross-item CLS cosine similarity:")
    for item_id, idxs in list(groups.items())[:5]:
        within = [sim[a, b] for a in idxs for b in idxs if a != b]
        cross = [sim[a, b] for a in idxs for b in range(len(ids)) if b not in idxs]
        w = float(np.mean(within)) if within else float("nan")
        print(f"  {item_id}: views={len(idxs)} within={w:.3f} cross={float(np.mean(cross)):.3f}")


if __name__ == "__main__":
    main()
