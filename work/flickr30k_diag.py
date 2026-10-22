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
import torch.nn.functional as F
from PIL import Image

import timm
from timm.data import create_transform, resolve_data_config
from transformers import CLIPConfig, CLIPModel, CLIPProcessor

from models import TextProjection, VisualHead


FEAT_DIR = os.path.join(ROOT, "work", "features")


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
    parquet = os.path.join(ROOT, "work", "flickr30k_test.parquet")
    df = pq.read_table(parquet).to_pandas()
    images = [Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB") for _, r in df.iterrows()]
    caps = [str(r["caption"]) for _, r in df.iterrows()]
    n = len(images)

    dino = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0).to(device).eval()
    dcfg = resolve_data_config(dino.pretrained_cfg)
    dt = create_transform(**dcfg, is_training=False)
    np_ = dino.num_prefix_tokens
    cls_all, patch_all = [], []
    with torch.inference_mode():
        for i in range(0, n, 32):
            x = torch.stack([dt(im) for im in images[i : i + 32]]).to(device)
            f = dino.forward_features(x)
            cls_all.append(f[:, 0].float().cpu())
            patch_all.append(f[:, np_:].float().cpu())
    cls = torch.cat(cls_all)
    patch = torch.cat(patch_all)

    clip, cproc = load_clip(device)
    T = []
    with torch.inference_mode():
        for i in range(0, n, 128):
            inp = cproc(text=caps[i : i + 128], return_tensors="pt", padding=True, truncation=True)
            T.append(clip.get_text_features(input_ids=inp["input_ids"].to(device), attention_mask=inp["attention_mask"].to(device)).float().cpu())
    T = torch.cat(T)
    I_clip = []
    with torch.inference_mode():
        for i in range(0, n, 32):
            inp = cproc(images=images[i : i + 32], return_tensors="pt")
            I_clip.append(clip.get_image_features(pixel_values=inp["pixel_values"].to(device)).float().cpu())
    I_clip = torch.cat(I_clip)
    T = F.normalize(T, dim=1)
    I_clip = F.normalize(I_clip, dim=1)
    print("CLIP native t2i:", t2i(T @ I_clip.T))

    vhead = VisualHead().to(device)
    vhead.load_state_dict(torch.load(os.path.join(FEAT_DIR, "attr_align_visual_w3+all_t07.pt"), map_location="cpu", weights_only=True))
    tproj = TextProjection().to(device)
    tproj.load_state_dict(torch.load(os.path.join(FEAT_DIR, "attr_align_textproj_w3+all_t07.pt"), map_location="cpu", weights_only=True))
    vhead.eval()
    tproj.eval()

    with torch.inference_mode():
        To = F.normalize(tproj(T.to(device)), dim=1).cpu()
        # V1: trained attention pooling.
        I1 = []
        for i in range(0, n, 64):
            I1.append(F.normalize(vhead(cls[i : i + 64].to(device), patch[i : i + 64].to(device)), dim=1).cpu())
        I1 = torch.cat(I1)
        # V2: bypass AttentionPooling, use patch mean-pooling through the same MLP.
        I2 = []
        for i in range(0, n, 64):
            c = cls[i : i + 64].to(device)
            p = patch[i : i + 64].to(device)
            fused = torch.cat([c, p.mean(dim=1)], dim=1)
            I2.append(F.normalize(vhead.proj(fused), dim=1).cpu())
        I2 = torch.cat(I2)

    print("Ours V1 attention t2i:", t2i(To @ I1.T))
    print("Ours V2 mean-pool t2i:", t2i(To @ I2.T))


if __name__ == "__main__":
    main()
