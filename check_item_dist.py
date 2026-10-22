"""Check image count distribution per item in the subset and full dataset."""
from collections import Counter
import pyarrow.parquet as pq


def item_key(item_id):
    parts = item_id.split("_")
    for i, p in enumerate(parts):
        if p == "id":
            return "_".join(parts[: i + 2])
    return item_id


def analyze(parquet_path, label):
    t = pq.read_table(parquet_path, columns=["item_ID"]).to_pandas()
    ids = t["item_ID"].tolist()
    keys = [item_key(i) for i in ids]
    c = Counter(keys)
    counts = sorted(c.values(), reverse=True)
    print(f"\n=== {label} ===")
    print(f"Total images: {len(ids)}, unique items: {len(c)}")
    print(f"Images per item: min={min(counts)}, max={max(counts)}, "
          f"mean={sum(counts)/len(counts):.2f}, median={counts[len(counts)//2]}")
    print(f"Items with 1 image: {sum(1 for v in counts if v == 1)} ({sum(1 for v in counts if v == 1)/len(c)*100:.1f}%)")
    print(f"Items with 2+ images: {sum(1 for v in counts if v >= 2)} ({sum(1 for v in counts if v >= 2)/len(c)*100:.1f}%)")
    print(f"Items with 4+ images: {sum(1 for v in counts if v >= 4)}")
    # Show distribution
    dist = Counter(counts)
    for k in sorted(dist.keys())[:10]:
        print(f"  {k} images: {dist[k]} items")


analyze(r"d:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\work\deepfashion_subset.parquet", "Subset (5000)")
analyze(r"d:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\work\deepfashion_inshop.parquet", "Full dataset")
