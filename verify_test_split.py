"""Verify that every test query has at least one positive in the gallery.

Uses the SAME split logic as run_seeds_local.py (60/20/20 by item_key)
and the CORRECTED item_key function (up to id_XXXXXXX).
"""
import numpy as np
import pyarrow.parquet as pq


def item_key(item_id):
    parts = item_id.split("_")
    for i, p in enumerate(parts):
        if p == "id":
            return "_".join(parts[: i + 2])
    return item_id


def main():
    parquet = r"d:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\work\deepfashion_subset.parquet"
    t = pq.read_table(parquet, columns=["item_ID", "category1"]).to_pandas()
    item_ids = t["item_ID"].tolist()
    cats = t["category1"].tolist()

    keys = [item_key(i) for i in item_ids]
    uniques, labels = np.unique(keys, return_inverse=True)

    # Same split as run_seeds_local.py
    rng = np.random.RandomState(0)
    idx = np.arange(len(uniques))
    rng.shuffle(idx)
    n_train = int(len(idx) * 0.6)
    n_val = int(len(idx) * 0.2)
    train_keys = set(uniques[idx[:n_train]].tolist())
    test_keys = set(uniques[idx[n_train + n_val :]].tolist())

    train_idx = np.where(np.isin(keys, list(train_keys)))[0]
    test_idx = np.where(np.isin(keys, list(test_keys)))[0]

    print(f"Total images: {len(item_ids)}")
    print(f"Unique items (correct key): {len(uniques)}")
    print(f"Train items: {len(train_keys)}, images: {len(train_idx)}")
    print(f"Test items: {len(test_keys)}, images: {len(test_idx)}")
    print()

    # For each test image, count how many OTHER test images share its item_key
    from collections import Counter
    test_key_counts = Counter(keys[i] for i in test_idx)

    no_pos = 0
    single_count = 0
    for i in test_idx:
        k = keys[i]
        # positive = other test images with same key
        n_pos = test_key_counts[k] - 1
        if n_pos == 0:
            no_pos += 1
        if test_key_counts[k] == 1:
            single_count += 1

    print(f"Test queries with ZERO positives in gallery: {no_pos}/{len(test_idx)}")
    print(f"Test items with only 1 image (no possible positive): {single_count}")
    print()

    # Also check by gender
    men_test = [i for i in test_idx if cats[i] == "men"]
    women_test = [i for i in test_idx if cats[i] == "women"]
    men_no_pos = sum(1 for i in men_test if test_key_counts[keys[i]] == 1)
    women_no_pos = sum(1 for i in women_test if test_key_counts[keys[i]] == 1)
    print(f"Men test: {len(men_test)} images, {men_no_pos} without positives")
    print(f"Women test: {len(women_test)} images, {women_no_pos} without positives")
    print()

    # Distribution of images per test item
    img_per_item = sorted(test_key_counts.values(), reverse=True)
    print(f"Images per test item: min={min(img_per_item)}, max={max(img_per_item)}, "
          f"mean={np.mean(img_per_item):.1f}, median={np.median(img_per_item):.0f}")
    print(f"Items with 1 image: {sum(1 for v in img_per_item if v == 1)}")
    print(f"Items with 2+ images: {sum(1 for v in img_per_item if v >= 2)}")

    if no_pos == 0:
        print("\nPASS: Every test query has at least one positive in gallery.")
    else:
        print(f"\nFAIL: {no_pos} test queries have no positives.")


if __name__ == "__main__":
    main()
