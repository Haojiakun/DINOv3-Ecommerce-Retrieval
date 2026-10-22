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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import timm
from timm.data import create_transform, resolve_data_config
from transformers import CLIPConfig, CLIPModel, CLIPProcessor


def load_clip(device):
    cfg = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel(cfg).to(device).eval()
    w = os.path.join(ROOT, "work", "hf_home", "hub", "models--openai--clip-vit-base-patch32",
                     "snapshots", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", "pytorch_model.bin")
    model.load_state_dict(torch.load(w, map_location="cpu", weights_only=True), strict=False)
    return model, CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def t2i(sim):
    top = sim.topk(11, dim=1).indices.numpy()
    hits = {1: 0, 5: 0, 10: 0}
    for i in range(sim.shape[0]):
        for k in hits:
            hits[k] += int(i in top[i, :k].tolist())
    return {k: v / sim.shape[0] for k, v in hits.items()}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pq.read_table(os.path.join(ROOT, "work", "flickr30k_test.parquet")).to_pandas()
    images = [Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB") for _, r in df.iterrows()]
    caps = [str(r["caption"]) for _, r in df.iterrows()]
    n = len(images)

    dino = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0).to(device).eval()
    dcfg = resolve_data_config(dino.pretrained_cfg)
    dt = create_transform(**dcfg, is_training=False)
    np_ = dino.num_prefix_tokens
    cls, patch = [], []
    with torch.inference_mode():
        for i in range(0, n, 32):
            x = torch.stack([dt(im) for im in images[i : i + 32]]).to(device)
            f = dino.forward_features(x)
            cls.append(f[:, 0].float().cpu())
            patch.append(f[:, np_:].float().cpu())
    cls = torch.cat(cls)
    patch = torch.cat(patch)

    clip, cproc = load_clip(device)
    T = []
    with torch.inference_mode():
        for i in range(0, n, 128):
            inp = cproc(text=caps[i : i + 128], return_tensors="pt", padding=True, truncation=True)
            T.append(clip.get_text_features(input_ids=inp["input_ids"].to(device), attention_mask=inp["attention_mask"].to(device)).float().cpu())
    T = torch.cat(T)

    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    tr, te = perm[:700], perm[700:]

    vis_proj = nn.Sequential(nn.Linear(1536, 768), nn.ReLU(), nn.Linear(768, 256)).to(device)
    txt_proj = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 256)).to(device)
    opt = torch.optim.AdamW(list(vis_proj.parameters()) + list(txt_proj.parameters()), lr=1e-3, weight_decay=1e-4)

    for step in range(400):
        bi = tr[rng.choice(len(tr), 128, replace=False)]
        c = cls[bi].to(device)
        p = patch[bi].to(device)
        x = torch.cat([c, p.mean(dim=1)], dim=1)
        v = F.normalize(vis_proj(x), dim=1)
        t = F.normalize(txt_proj(T[bi].to(device)), dim=1)
        logits = (v @ t.T) / 0.07
        gt = torch.arange(128, device=device)
        loss = F.cross_entropy(logits, gt) + F.cross_entropy(logits.T, gt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(vis_proj.parameters()) + list(txt_proj.parameters()), 1.0)
        opt.step()

    vis_proj.eval()
    txt_proj.eval()
    with torch.inference_mode():
        V = []
        for i in range(0, len(te), 64):
            bi = te[i : i + 64]
            c = cls[bi].to(device)
            p = patch[bi].to(device)
            V.append(F.normalize(vis_proj(torch.cat([c, p.mean(dim=1)], dim=1)), dim=1).cpu())
        V = torch.cat(V)
        Tt = F.normalize(txt_proj(T[te].to(device)), dim=1).cpu()
    print("Probe (adapted projections, Flickr30K 700/300) t2i:", t2i(Tt @ V.T))


if __name__ == "__main__":
    main()
