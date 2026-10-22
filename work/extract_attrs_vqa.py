import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "work", "hf_home"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_key, None)

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor


QUESTIONS = {
    "material": "What is the main material of this clothing?",
    "neckline": "What type of neckline does this garment have?",
    "sleeve": "What is the sleeve length of this garment?",
    "pattern": "What is the pattern of this garment?",
}


def main():
    n_images = int(os.environ.get("VQA_N", "5000"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parquet = os.path.join(ROOT, "work", "deepfashion_subset.parquet")
    feat_dir = os.path.join(ROOT, "work", "features")

    model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base", use_safetensors=True
    ).to(device).eval()
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

    rows = pq.read_table(parquet, columns=["image", "item_ID"]).to_pandas()

    results = []
    for idx, (_, r) in enumerate(rows.iterrows()):
        img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        record = {"item_ID": str(r["item_ID"])}
        for attr, q in QUESTIONS.items():
            inputs = processor(images=img, text=q, return_tensors="pt").to(device)
            with torch.inference_mode():
                out = model.generate(
                    **inputs, max_new_tokens=12,
                    output_scores=True, return_dict_in_generate=True,
                )
            ans = processor.decode(out.sequences[0], skip_special_tokens=True).strip()
            if out.scores:
                probs = [F.softmax(s, dim=-1).max(dim=-1).values for s in out.scores]
                conf = float(torch.stack(probs, dim=1).mean(dim=1)[0])
            else:
                conf = 1.0
            record[attr] = {"answer": ans, "confidence": conf}
        results.append(record)
        if (idx + 1) % 20 == 0:
            print(f"processed {idx + 1}/{n_images}")

    out = os.path.join(feat_dir, "vqa_attrs.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
