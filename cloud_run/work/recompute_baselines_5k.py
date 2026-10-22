import io, json, os
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
import timm
from timm.data import create_transform, resolve_data_config

ROOT = "/root/cloud_run"
FEAT_DIR = os.path.join(ROOT, "work", "features_5k")
PARQUET = os.path.join(ROOT, "work", "deepfashion_subset.parquet")


def recall(sim, labels, ks=(1, 5, 10), exclude_self=True):
    topk = sim.topk(max(ks) + 1, dim=1).indices.numpy()
    if exclude_self:
        topk = topk[:, 1:]
    else:
        topk = topk[:, :max(ks)]
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


def scda_descriptor(fmap):
    agg = fmap.sum(dim=1, keepdim=True)
    mean = agg.mean(dim=(2, 3), keepdim=True)
    mask = (agg > mean).float()
    desc = (fmap * mask).sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-8)
    return F.normalize(desc, dim=1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    with open(os.path.join(FEAT_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    item_keys = np.asarray(meta["item_key"])
    test_idx, test_labels = get_test_split(item_keys)
    print(f"test: {len(test_idx)} images, {len(set(test_labels.tolist()))} products")

    # --- CLS, mean-pool, GeM from cached DINOv3 features ---
    patch = np.load(os.path.join(FEAT_DIR, "patch.npy")).astype(np.float32)[test_idx]
    cls = np.load(os.path.join(FEAT_DIR, "cls.npy")).astype(np.float32)[test_idx]
    print(f"patch shape: {patch.shape}, cls shape: {cls.shape}")

    def rep(name, v):
        v = F.normalize(torch.from_numpy(v), dim=1)
        r, nq = recall(v @ v.T, test_labels, exclude_self=True)
        print(f"{name}: R@1={r[1]*100:.1f} R@5={r[5]*100:.1f} R@10={r[10]*100:.1f} (nq={nq})")
        return r

    rep("DINOv3 CLS", cls)
    rep("DINOv3 patch mean-pool", patch.mean(axis=1))
    p = 3.0
    gem = np.power(np.clip(patch, 0, None).__pow__(p).mean(axis=1), 1.0 / p)
    rep("DINOv3 patch GeM (p=3)", gem)

    # --- SCDA with ResNet-50 ---
    print("\n=== SCDA (ResNet-50) ===")
    try:
        model = timm.create_model("resnet50", pretrained=True, num_classes=0)
        print("loaded resnet50 pretrained via timm")
    except Exception as e:
        print(f"timm pretrained failed: {e}")
        return
    model = model.to(device).eval()
    cfg = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**cfg, is_training=False)

    table = pq.read_table(PARQUET, columns=["image"])
    descs = []
    with torch.inference_mode():
        for i in range(0, len(test_idx), 64):
            bi = test_idx[i : i + 64]
            imgs = [Image.open(io.BytesIO(table.column("image")[j].as_py()["bytes"])).convert("RGB") for j in bi]
            x = torch.stack([transform(img) for img in imgs]).to(device)
            fmap = model.forward_features(x)
            descs.append(scda_descriptor(fmap).cpu())
    D = torch.cat(descs)
    r, nq = recall(D @ D.T, test_labels, exclude_self=True)
    print(f"SCDA (ResNet-50): R@1={r[1]*100:.1f} R@5={r[5]*100:.1f} R@10={r[10]*100:.1f} (nq={nq})")
    np.save(os.path.join(FEAT_DIR, "scda_test.npy"), D.numpy().astype(np.float16))
    print("saved scda_test.npy")


if __name__ == "__main__":
    main()
