import numpy as np
import sys

patch = np.load('/root/cloud_run/work/features/patch.npy', mmap_mode='r')
cls = np.load('/root/cloud_run/work/features/cls.npy', mmap_mode='r')
text = np.load('/root/cloud_run/work/features/text_feats.npy', mmap_mode='r')
print(f"patch dtype={patch.dtype} shape={patch.shape}")
print(f"cls dtype={cls.dtype} shape={cls.shape}")
print(f"text dtype={text.dtype} shape={text.shape}")

for name, arr in [("patch", patch), ("cls", cls), ("text", text)]:
    has_nan = False
    has_inf = False
    n_zero = 0
    n_total = 0
    for i in range(0, arr.shape[0], 1000):
        chunk = np.array(arr[i:i+1000])
        if np.isnan(chunk).any(): has_nan = True
        if np.isinf(chunk).any(): has_inf = True
        n_zero += int((chunk == 0).sum())
        n_total += chunk.size
    print(f"{name}: nan={has_nan} inf={has_inf} zero_ratio={n_zero/n_total:.4f}")
    s = np.array(arr[:100])
    print(f"  sample min={s.min():.4f} max={s.max():.4f} mean={s.mean():.4f} std={s.std():.4f}")

# Check per-sample norms (any zero-norm samples?)
print("\nper-sample norms:")
for name, arr in [("cls", cls), ("text", text)]:
    norms = np.linalg.norm(np.array(arr[:1000]), axis=-1)
    print(f"  {name}: min_norm={norms.min():.4f} max_norm={norms.max():.4f} n_zero={int((norms==0).sum())}")
