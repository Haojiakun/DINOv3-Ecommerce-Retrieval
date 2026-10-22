import pyarrow.parquet as pq
import json
from collections import Counter

t = pq.read_table(r'd:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\work\deepfashion_inshop.parquet',
                  columns=['item_ID', 'category1', 'category2'])
df = t.to_pandas()

with open(r'd:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\work\features\meta.json') as f:
    meta = json.load(f)
sub_ids = set(meta['item_id'])
sub_df = df[df['item_ID'].isin(sub_ids)]

print('=== category2 distribution (top 15) ===')
full_c2 = Counter(df['category2'])
sub_c2 = Counter(sub_df['category2'])
all_c2 = sorted(set(full_c2) | set(sub_c2), key=lambda c: -full_c2[c])[:15]
fn, sn = sum(full_c2.values()), sum(sub_c2.values())

print(f'{"category2":<25} {"full%":>8} {"sub%":>8} {"delta":>8}')
for c in all_c2:
    fp = full_c2[c] / fn * 100
    sp = sub_c2[c] / sn * 100
    print(f'{c:<25} {fp:>7.2f}  {sp:>7.2f}  {sp-fp:>+7.2f}')

print(f'\nfull unique category2: {len(full_c2)}')
print(f'subset unique category2: {len(sub_c2)}')
print(f'subset items: {sub_df["item_ID"].nunique()}')
print(f'full items: {df["item_ID"].nunique()}')
