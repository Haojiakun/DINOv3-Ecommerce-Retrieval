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
    w = os.path.join(
        ROOT, "work", "hf_home", "hub", "models--openai--clip-vit-base-patch32",
        "snapshots", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", "pytorch_model.bin",
    )
    model.load_state_dict(torch.load(w, map_location="cpu", weights_only=True), strict=False)
    return model, CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def retrieval_metrics(sim_qt, n_img, n_cap_per_img, direction):
    # direction: "t2i" (query=caption, gallery=image) or "i2t"
    if direction == "t2i":
        # sim_qt: (n_cap, n_img)
        top = sim_qt.topk(11, dim=1).indices.numpy()[:, :]
        hits = {1: 0, 5: 0, 10: 0}
        n_q = sim_qt.shape[0]
        for i in range(n_q):
            gt_img = i // n_cap_per_img
            for k in hits:
                hits[k] += int(gt_img in top[i, :k].tolist())
    else:
        # sim_qt: (n_cap, n_img) -> transpose for image queries
        top = sim_qt.T.topk(11, dim=1).indices.numpy()[:, :]
        hits = {1: 0, 5: 0, 10: 0}
        n_q = sim_qt.shape[1]
        for i in range(n_q):
            gt_caps = set(range(i * n_cap_per_img, (i + 1) * n_cap_per_img))
            for k in hits:
                hits[k] += int(bool(gt_caps.intersection(top[i, :k].tolist())))
    return {k: v / n_q for k, v in hits.items()}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parquet = os.path.join(ROOT, "work", "flickr30k_test.parquet")
    if not os.path.exists(parquet):
        raise SystemExit("flickr30k_test.parquet missing; download it first (see outputs/scda_flickr.md)")

    table = pq.read_table(parquet)
    print("columns:", table.column_names)
    df = table.to_pandas()
    images = [Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB") for _, r in df.iterrows()]
    caps = [r["caption"] for _, r in df.iterrows()]  # list of 5 captions per image
    n_img = len(images)
    n_cap_per_img = len(caps[0]) if isinstance(caps[0], (list, tuple)) else 1
    flat_caps = [c for group in caps for c in (group if isinstance(group, (list, tuple)) else [group])]
    flat_caps = [str(c) for c in flat_caps]  # ensure all strings (parquet may store bytes)
    print(f"images={n_img}, captions per image={n_cap_per_img}, total captions={len(flat_caps)}")

    # DINOv3 visual features (frozen backbone).
    dino = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0).to(device).eval()
    dcfg = resolve_data_config(dino.pretrained_cfg)
    dtransform = create_transform(**dcfg, is_training=False)
    n_prefix = dino.num_prefix_tokens
    V = []
    with torch.inference_mode():
        for i in range(0, n_img, 32):
            x = torch.stack([dtransform(im) for im in images[i : i + 32]]).to(device)
            feats = dino.forward_features(x)
            V.append((feats[:, 0], feats[:, n_prefix:]))
    cls = torch.cat([v[0] for v in V])
    patch = torch.cat([v[1] for v in V])

    # CLIP text + image (native zero-shot baseline).
    clip, cproc = load_clip(device)
    T_cap = []
    with torch.inference_mode():
        for i in range(0, len(flat_caps), 128):
            inp = cproc(text=flat_caps[i : i + 128], return_tensors="pt", padding=True, truncation=True)
            t = clip.get_text_features(input_ids=inp["input_ids"].to(device), attention_mask=inp["attention_mask"].to(device))
            T_cap.append(t.float().cpu())
    T_cap = torch.cat(T_cap)
    I_clip = []
    with torch.inference_mode():
        for i in range(0, n_img, 32):
            inp = cproc(images=images[i : i + 32], return_tensors="pt")
            v = clip.get_image_features(pixel_values=inp["pixel_values"].to(device))
            I_clip.append(v.float().cpu())
    I_clip = torch.cat(I_clip)

    T_cap = F.normalize(T_cap, dim=1)
    I_clip = F.normalize(I_clip, dim=1)
    sim_clip = T_cap @ I_clip.T
    print("CLIP native t2i:", retrieval_metrics(sim_clip, n_img, n_cap_per_img, "t2i"))
    print("CLIP native i2t:", retrieval_metrics(sim_clip, n_img, n_cap_per_img, "i2t"))

    # Our DeepFashion-trained alignment, applied zero-shot.
    vhead = VisualHead().to(device)
    vhead.load_state_dict(torch.load(os.path.join(FEAT_DIR, "attr_align_visual_w3+all_t07.pt"), map_location="cpu", weights_only=True))
    tproj = TextProjection().to(device)
    tproj.load_state_dict(torch.load(os.path.join(FEAT_DIR, "attr_align_textproj_w3+all_t07.pt"), map_location="cpu", weights_only=True))
    vhead.eval()
    tproj.eval()
    I_ours = []
    with torch.inference_mode():
        for i in range(0, n_img, 64):
            I_ours.append(F.normalize(vhead(cls[i : i + 64].to(device), patch[i : i + 64].to(device)), dim=1).cpu())
    I_ours = torch.cat(I_ours)
    T_ours = []
    with torch.inference_mode():
        for i in range(0, len(flat_caps), 128):
            T_ours.append(F.normalize(tproj(T_cap[i : i + 128].to(device)), dim=1).cpu())
    T_ours = torch.cat(T_ours)
    sim_ours = T_ours @ I_ours.T
    print("Ours (zero-shot) t2i:", retrieval_metrics(sim_ours, n_img, n_cap_per_img, "t2i"))
    print("Ours (zero-shot) i2t:", retrieval_metrics(sim_ours, n_img, n_cap_per_img, "i2t"))

    # Diagnostic: replace learned AttentionPooling with generic mean-pooling,
    # but keep the SAME trained projection heads (VisualHead.proj + TextProjection).
    # If this recovers performance -> attention specialization is the bottleneck.
    # If still near zero -> TextProjection (clothing-attribute text) is the bottleneck.
    I_mean = []
    with torch.inference_mode():
        for i in range(0, n_img, 64):
            p = patch[i : i + 64].to(device)
            mean_pooled = p.mean(dim=1)  # (B, 768) generic patch aggregation
            fused = torch.cat([cls[i : i + 64].to(device), mean_pooled], dim=-1)  # (B, 1536)
            I_mean.append(F.normalize(vhead.proj(fused), dim=1).cpu())
    I_mean = torch.cat(I_mean)
    sim_mean = T_ours @ I_mean.T  # reuse the same TextProjection output
    print("Mean-pool (same proj, no attn) t2i:", retrieval_metrics(sim_mean, n_img, n_cap_per_img, "t2i"))
    print("Mean-pool (same proj, no attn) i2t:", retrieval_metrics(sim_mean, n_img, n_cap_per_img, "i2t"))


if __name__ == "__main__":
    main()
