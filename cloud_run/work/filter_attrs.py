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
from transformers import CLIPModel, CLIPProcessor


def norm(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


def main():
    feat_dir = os.path.join(ROOT, "work", os.environ.get("FEAT_DIR", "features"))
    with open(os.path.join(feat_dir, "attributes.json"), encoding="utf-8") as f:
        dictionary = json.load(f)
    with open(os.path.join(feat_dir, "vqa_attrs.json"), encoding="utf-8") as f:
        attrs = json.load(f)

    dict_vals = {k: set(norm(v) for v in vals) for k, vals in dictionary.items()}
    bottom = {"pants", "shorts", "skirts", "leggings", "denim", "jeans"}

    parquet = os.path.join(ROOT, "work", "deepfashion_subset.parquet")
    cats = pq.read_table(parquet, columns=["item_ID", "category2"]).to_pandas()

    filtered = []
    for i, rec in enumerate(attrs):
        item_id = rec["item_ID"]
        cat2 = str(cats.iloc[i]["category2"]).strip().lower() if cats.iloc[i]["category2"] else ""
        out = {"item_ID": item_id, "category2": cat2}
        conf_thr = float(os.environ.get("CONF_THRESHOLD", "0.6"))
        for dim in ["material", "neckline", "sleeve", "pattern"]:
            raw = rec.get(dim, "")
            if isinstance(raw, dict):
                answer = raw.get("answer", "")
                conf = float(raw.get("confidence", 1.0))
            else:
                answer = raw
                conf = 1.0
            val = norm(answer)
            if cat2 in bottom and dim in ("neckline", "sleeve"):
                out[dim] = "none"
            elif conf < conf_thr:
                out[dim] = "unknown"
            elif val in dict_vals.get(dim, set()):
                out[dim] = answer.strip().lower()
            else:
                out[dim] = "unknown"
        filtered.append(out)

    with open(os.path.join(feat_dir, "filtered_attrs.json"), "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    # CLIP text embeddings of the "sleeve" attribute phrase, for local alignment.
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda" if torch.cuda.is_available() else "cpu").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    phrase_tpl = {"material": "{} material", "neckline": "{} neckline", "sleeve": "{} sleeve", "pattern": "{} pattern"}
    for dim in ["material", "neckline", "sleeve", "pattern"]:
        phrases = [phrase_tpl[dim].format(rec[dim]) for rec in filtered]
        emb = []
        with torch.inference_mode():
            for i in range(0, len(phrases), 64):
                inp = proc(text=phrases[i : i + 64], return_tensors="pt", padding=True, truncation=True)
                t = model.get_text_features(input_ids=inp["input_ids"].to(model.device), attention_mask=inp["attention_mask"].to(model.device))
                if hasattr(t, "pooler_output"):
                    t = t.pooler_output
                emb.append(t.float().cpu())
        emb = torch.cat(emb)
        np.save(os.path.join(feat_dir, f"attr_emb_{dim}.npy"), emb.numpy().astype(np.float16))
        np.save(os.path.join(feat_dir, f"attr_label_{dim}.npy"), np.asarray([rec[dim] for rec in filtered]))
    print("saved filtered_attrs.json and 4 attribute embeddings; records", len(filtered))


if __name__ == "__main__":
    main()
