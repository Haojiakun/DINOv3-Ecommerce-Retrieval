import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "work", "hf_home"))
for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_key, None)

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

import timm
from timm.data import create_transform, resolve_data_config


def main():
    model_name = os.environ.get("MODEL", "vit_base_patch16_clip_224.openai")
    tag = os.environ.get("TAG", "clip")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device).eval()
    cfg = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**cfg, is_training=False)
    n_prefix = model.num_prefix_tokens
    print("model", model_name, "embed_dim", model.embed_dim, "n_prefix", n_prefix, "input", cfg["input_size"])

    feat_dir = os.path.join(ROOT, "work", os.environ.get("FEAT_DIR", "features"))
    os.makedirs(feat_dir, exist_ok=True)
    parquet_name = os.environ.get("PARQUET", "deepfashion_subset.parquet")
    parquet = os.path.join(ROOT, "work", parquet_name)
    rows = pq.read_table(parquet, columns=["image"]).to_pandas()
    n_images = len(rows)
    images = [Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB") for _, r in rows.iterrows()]

    with torch.inference_mode():
        dummy = model.forward_features(torch.stack([transform(images[0])]).to(device))
    n_patches = dummy.shape[1] - n_prefix
    del dummy

    feat_dir = os.path.join(ROOT, "work", "features")
    cls_path = os.path.join(feat_dir, f"cls_{tag}.npy")
    patch_path = os.path.join(feat_dir, f"patch_{tag}.npy")
    cls_mm = np.lib.format.open_memmap(cls_path, mode="w+", dtype=np.float16, shape=(n_images, model.embed_dim))
    patch_mm = np.lib.format.open_memmap(patch_path, mode="w+", dtype=np.float16, shape=(n_images, n_patches, model.embed_dim))

    with torch.inference_mode():
        for i in range(0, n_images, 64):
            end = min(i + 64, n_images)
            x = torch.stack([transform(img) for img in images[i:end]]).to(device)
            feats = model.forward_features(x)
            cls_mm[i:end] = feats[:, 0].float().cpu().numpy().astype(np.float16)
            patch_mm[i:end] = feats[:, n_prefix:].float().cpu().numpy().astype(np.float16)
            if (i // 64) % 20 == 0:
                print(f"processed {end}/{n_images}")
    cls_mm.flush()
    patch_mm.flush()
    print("saved", f"cls_{tag}.npy", cls_mm.shape, f"patch_{tag}.npy", patch_mm.shape)


if __name__ == "__main__":
    main()
