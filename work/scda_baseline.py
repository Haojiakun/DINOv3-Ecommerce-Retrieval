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

import timm
from timm.data import create_transform, resolve_data_config


def scda_descriptor(fmap):
    # fmap: (B, C, H, W); SCDA: aggregate channels, threshold on mean,
    # then sum-pool each channel over the selected region.
    agg = fmap.sum(dim=1, keepdim=True)  # (B,1,H,W)
    mean = agg.mean(dim=(2, 3), keepdim=True)
    mask = (agg > mean).float()
    desc = (fmap * mask).sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-8)
    return F.normalize(desc, dim=1)


def recall(sim, labels, ks=(1, 5, 10), exclude_self=True):
    topk = sim.topk(max(ks) + 1, dim=1).indices.numpy()[:, 1:]
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import safetensors.torch as st
    model = timm.create_model("resnet50", pretrained=False, num_classes=0)
    model.load_state_dict(st.load_file(os.path.join(ROOT, "work", "resnet50.safetensors")), strict=False)
    model = model.to(device).eval()
    cfg = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**cfg, is_training=False)

    meta = json.load(open(os.path.join(ROOT, "work", "features", "meta.json"), encoding="utf-8"))
    keys = np.asarray(meta["item_key"])
    uniques, labels = np.unique(keys, return_inverse=True)
    rng = np.random.RandomState(0)
    idx = np.arange(len(uniques))
    rng.shuffle(idx)
    n_train = int(len(idx) * 0.6)
    n_val = int(len(idx) * 0.2)
    test_keys = set(uniques[idx[n_train + n_val:]].tolist())
    test_idx = np.where(np.asarray([k in test_keys for k in keys]))[0]

    rows = pq.read_table(os.path.join(ROOT, "work", "deepfashion_inshop.parquet"), columns=["image"]).to_pandas()
    descs = []
    with torch.inference_mode():
        for i in range(0, len(test_idx), 64):
            bi = test_idx[i : i + 64]
            imgs = [Image.open(io.BytesIO(rows.iloc[j]["image"]["bytes"])).convert("RGB") for j in bi]
            x = torch.stack([transform(img) for img in imgs]).to(device)
            fmap = model.forward_features(x)
            descs.append(scda_descriptor(fmap).cpu())
    D = torch.cat(descs)
    r, nq = recall(D @ D.T, labels[test_idx], exclude_self=True)
    print(f"SCDA (ResNet-50, ImageNet): queries={nq} R@1={r[1]:.4f} R@5={r[5]:.4f} R@10={r[10]:.4f}")
    np.save(os.path.join(ROOT, "work", "features", "scda_test.npy"), D.numpy().astype(np.float16))


if __name__ == "__main__":
    main()
