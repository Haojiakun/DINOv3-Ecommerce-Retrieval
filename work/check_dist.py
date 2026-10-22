import pyarrow.parquet as pq
import json, math
from collections import Counter

t = pq.read_table(r'd:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\work\deepfashion_inshop.parquet',
                  columns=['item_ID', 'category1'])
df = t.to_pandas()
print(f'full: {len(df)} rows, {df["category1"].nunique()} categories')

with open(r'd:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\work\features\meta.json') as f:
    meta = json.load(f)
sub_ids = set(meta['item_id'])
print(f'subset: {len(sub_ids)} images')

sub_df = df[df['item_ID'].isin(sub_ids)]
print(f'subset matched: {len(sub_df)} rows')

full_counts = Counter(df['category1'])
sub_counts = Counter(sub_df['category1'])
cats = sorted(set(full_counts) | set(sub_counts))
fn, sn = sum(full_counts.values()), sum(sub_counts.values())

print(f'\n{"category":<20} {"full%":>8} {"sub%":>8} {"delta":>8}')
max_delta = 0
for c in cats:
    fp = full_counts[c] / fn * 100
    sp = sub_counts[c] / sn * 100
    d = sp - fp
    max_delta = max(max_delta, abs(d))
    print(f'{c:<20} {fp:>7.2f}  {sp:>7.2f}  {d:>+7.2f}')

p = [(full_counts[c] + 1) / (fn + len(cats)) for c in cats]
q = [(sub_counts[c] + 1) / (sn + len(cats)) for c in cats]
kl_pq = sum(p[i] * math.log(p[i] / q[i]) for i in range(len(cats)))
kl_qp = sum(q[i] * math.log(q[i] / p[i]) for i in range(len(cats)))
js = 0.5 * kl_pq + 0.5 * kl_qp
print(f'\nimages: full={fn} subset={sn}')
print(f'KL(full||sub)={kl_pq:.6f}  KL(sub||full)={kl_qp:.6f}  JS={js:.6f}')
print(f'max |delta|={max_delta:.2f}%')
