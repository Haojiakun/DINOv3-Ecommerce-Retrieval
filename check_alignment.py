import json, numpy as np

meta = json.load(open(r'd:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\work\features\meta.json', encoding='utf-8'))
print('meta keys:', list(meta.keys()))
print('num entries:', len(meta['item_id']))
print()
for i in range(3):
    print(f'row {i}: item_id={meta["item_id"][i]}, item_key={meta["item_key"][i]}, cat1={meta["category1"][i]}')
print()

patch = np.load(r'd:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\work\features\patch.npy', mmap_mode='r')
cls = np.load(r'd:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\work\features\cls.npy', mmap_mode='r')
text = np.load(r'd:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\work\features\text_feats.npy')
print(f'patch shape: {patch.shape}')
print(f'cls shape: {cls.shape}')
print(f'text shape: {text.shape}')
print(f'meta item_id count: {len(meta["item_id"])}')
print()

assert patch.shape[0] == cls.shape[0] == text.shape[0] == len(meta['item_id']), 'MISMATCH!'
print('All shapes aligned: OK')
