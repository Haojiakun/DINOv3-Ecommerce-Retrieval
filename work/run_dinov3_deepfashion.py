import io
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault(
    "HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_home")
)
for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_key, None)

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

import timm
from timm.data import create_transform, resolve_data_config


def load_images(parquet_path, limit=16):
    table = pq.ParquetFile(parquet_path).read_row_group(0)
    rows = table.to_pandas()
    rows = rows.head(limit)
    images, ids, colors = [], [], []
    for _, r in rows.iterrows():
        images.append(Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"))
        ids.append(r["item_ID"])
        colors.append(r["color"])
    return images, ids, colors


def item_key(item_id):
    # item_ID embeds the product and the view suffix, e.g.
    # "MEN_Denim_id_00000080_01_1_front" -> "MEN_Denim_id_00000080_01".
    parts = item_id.split("_")
    return "_".join(parts[:-2]) if len(parts) >= 3 else item_id


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parquet = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepfashion_inshop.parquet")

    model = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0)
    model = model.to(device).eval()

    data_cfg = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**data_cfg, is_training=False)
    print("data_config:", data_cfg)

    images, ids, colors = load_images(parquet, limit=16)
    print(f"loaded {len(images)} real DeepFashion images")

    batch = torch.stack([transform(img) for img in images]).to(device)
    with torch.inference_mode():
        feats = model.forward_features(batch)  # (B, 1 + 4 + H*W, D)

    n_prefix = model.num_prefix_tokens
    cls = feats[:, 0]
    patches = feats[:, n_prefix:]
    print("CLS:", tuple(cls.shape))
    print("patch tokens:", tuple(patches.shape))
    print("patch grid:", int(round(patches.shape[1] ** 0.5)), "x", int(round(patches.shape[1] ** 0.5)))

    cls = F.normalize(cls, dim=-1)
    sim = (cls @ cls.T).cpu().numpy()

    # Report within-item vs cross-item CLS cosine similarity for the first few items.
    print("\nitem -> CLS cosine similarity (within-item avg vs cross-item avg):")
    seen = {}
    for i, item_id in enumerate(ids):
        seen.setdefault(item_key(item_id), []).append(i)
    for item_id, idxs in list(seen.items())[:5]:
        within = [sim[a, b] for a in idxs for b in idxs if a != b]
        others = [j for j in range(len(ids)) if j not in idxs]
        cross = [sim[a, b] for a in idxs for b in others]
        w = float(np.mean(within)) if within else float("nan")
        c = float(np.mean(cross))
        print(f"  {item_id}: views={len(idxs)} within={w:.3f} cross={c:.3f}")


if __name__ == "__main__":
    main()
