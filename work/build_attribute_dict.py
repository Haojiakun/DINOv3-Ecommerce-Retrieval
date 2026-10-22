import json
import os
from collections import Counter

import pyarrow.parquet as pq


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parquet = os.path.join(ROOT, "work", "deepfashion_inshop.parquet")
    rows = pq.read_table(parquet, columns=["item_ID", "category1", "category2"]).to_pandas()

    id_tokens = Counter()
    cat2 = Counter()
    gender = Counter()
    for _, r in rows.iterrows():
        sid = str(r["item_ID"])
        parts = sid.split("_")
        gender[parts[0].lower()] += 1
        if len(parts) >= 2:
            id_tokens[parts[1].lower()] += 1
        c2 = str(r["category2"]).strip().lower() if r["category2"] and str(r["category2"]) != "nan" else ""
        if c2:
            cat2[c2] += 1

    print("gender vocab:", dict(gender))
    print("item_ID type vocab (top 40):", dict(id_tokens.most_common(40)))
    print("category2 vocab (top 40):", dict(cat2.most_common(40)))

    # Domain-grounded fine-grained attribute dictionary. The "category" axis is
    # filled from the dataset vocabulary; the other axes are editable seeds that
    # a local LLM/API can expand.
    dictionary = {
        "gender": sorted(gender),
        "category": sorted(cat2),
        "material": [
            "cotton", "denim", "silk", "chiffon", "leather", "wool", "linen",
            "polyester", "knit", "velvet", "lace", "satin",
        ],
        "neckline": [
            "crew", "v-neck", "round", "square", "turtleneck", "strapless",
            "collar", "sweetheart",
        ],
        "sleeve": ["long", "short", "sleeveless", "three-quarter", "cap"],
        "pattern": ["solid", "striped", "plaid", "graphic", "floral", "print", "polka-dot"],
        "fit": ["slim", "regular", "loose", "skinny", "oversized"],
    }

    out = os.path.join(ROOT, "work", "features", "attributes.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
