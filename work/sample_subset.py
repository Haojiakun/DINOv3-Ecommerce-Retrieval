"""Stratified subset sampling from DeepFashion In-shop parquet.

Samples 5000 images by item ID, preserving the men:women ratio of the full
dataset (~15:85). Outputs a new parquet with the same schema.

Usage:
    work\\venv\\Scripts\\python.exe work\\sample_subset.py
"""
import os

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL = os.path.join(ROOT, "work", "deepfashion_inshop.parquet")
OUT = os.path.join(ROOT, "work", "deepfashion_subset.parquet")
N_IMAGES = 5000
SEED = 42


def item_key(item_id):
    parts = item_id.split("_")
    for i, p in enumerate(parts):
        if p == "id":
            return "_".join(parts[: i + 2])
    return item_id


def main():
    pf = pq.ParquetFile(FULL)
    cols = ["item_ID", "category1"]
    table = pf.read(columns=cols)
    item_ids = table.column("item_ID").to_pylist()
    cats = table.column("category1").to_pylist()

    # Group by PRODUCT-LEVEL key (up to id_XXXXXXX), not raw item_ID
    from collections import defaultdict
    item_rows = defaultdict(list)
    for idx, iid in enumerate(item_ids):
        item_rows[item_key(iid)].append((idx, cats[idx]))

    men_items = []
    women_items = []
    for iid, rows in item_rows.items():
        if len(rows) < 2:
            continue  # skip single-image products (no retrieval positive)
        g = rows[0][1]
        if g == "men":
            men_items.append(iid)
        else:
            women_items.append(iid)

    print(f"full products: men={len(men_items)}, women={len(women_items)}")
    print(f"full images: men={sum(len(item_rows[i]) for i in men_items)}, "
          f"women={sum(len(item_rows[i]) for i in women_items)}")

    import random
    rng = random.Random(SEED)
    rng.shuffle(men_items)
    rng.shuffle(women_items)

    target_men = int(N_IMAGES * 0.15)
    target_women = N_IMAGES - target_men

    def pick(items, target):
        picked = []
        count = 0
        for iid in items:
            if count >= target:
                break
            picked.append(iid)
            count += len(item_rows[iid])
        return picked, count

    picked_men, n_men = pick(men_items, target_men)
    picked_women, n_women = pick(women_items, target_women)

    picked_ids = set(picked_men + picked_women)
    print(f"picked products: men={len(picked_men)} ({n_men} imgs), "
          f"women={len(picked_women)} ({n_women} imgs)")
    print(f"total images: {n_men + n_women}")
    print(f"men ratio: {n_men / (n_men + n_women) * 100:.1f}%")

    keep_rows = []
    for iid in picked_ids:
        for idx, _ in item_rows[iid]:
            keep_rows.append(idx)
    keep_rows.sort()

    print("reading full table...")
    full_table = pf.read()
    subset = full_table.take(keep_rows)
    print(f"subset rows: {subset.num_rows}")

    pq.write_table(subset, OUT)
    size = os.path.getsize(OUT) / 1024 / 1024
    print(f"saved {OUT} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
