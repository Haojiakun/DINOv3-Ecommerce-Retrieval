import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "work", "hf_home"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_key, None)

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import CLIPConfig, CLIPModel, CLIPProcessor


def gender_word(item_id):
    first = item_id.split("_")[0].lower()
    return {"men": "men's", "women": "women's"}.get(first, "")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parquet = os.path.join(ROOT, "work", "deepfashion_subset.parquet")
    feat_dir = os.path.join(ROOT, "work", "features")

    cfg = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel(cfg).to(device).eval()
    weight_file = os.path.join(
        ROOT, "work", "hf_home", "hub", "models--openai--clip-vit-base-patch32",
        "snapshots", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", "pytorch_model.bin",
    )
    model.load_state_dict(torch.load(weight_file, map_location="cpu", weights_only=True), strict=False)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    cols = ["item_ID", "category1", "category2", "category3", "color"]
    rows = pq.read_table(parquet, columns=cols).to_pandas()

    def clean(v):
        s = str(v).strip().lower().replace("_", " ") if v is not None and str(v) != "nan" else ""
        return s

    texts = []
    for _, r in rows.iterrows():
        gender = gender_word(str(r["item_ID"]))
        parts = [p for p in [clean(r["category1"]), clean(r["category2"]), clean(r["category3"])] if p]
        color = clean(r["color"])
        base = "a photo of " + (gender + " " if gender else "") + " ".join(parts)
        if color:
            base += ", color " + color
        texts.append(base.strip().rstrip(".") + ".")

    feats = []
    bs = 128
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            inp = proc(text=texts[i : i + bs], return_tensors="pt", padding=True, truncation=True)
            t = model.get_text_features(input_ids=inp["input_ids"].to(device), attention_mask=inp["attention_mask"].to(device))
            feats.append(t.float().cpu())
    feats = torch.cat(feats)
    np.save(os.path.join(feat_dir, "text_feats.npy"), feats.numpy().astype(np.float16))
    with open(os.path.join(feat_dir, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    print("saved text_feats.npy", tuple(feats.shape), "and texts.json")


if __name__ == "__main__":
    main()
