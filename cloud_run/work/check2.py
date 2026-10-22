import numpy as np
cls = np.load('/root/cloud_run/work/features/cls.npy', mmap_mode='r')
patch = np.load('/root/cloud_run/work/features/patch.npy', mmap_mode='r')
print(f"cls dtype={cls.dtype} shape={cls.shape}")
print(f"cls[0,:5] = {cls[0,:5]}")
print(f"cls[0] norm = {np.linalg.norm(cls[0].astype(np.float32)):.4f}")
print(f"cls min/max (first 1000) = {cls[:1000].min():.4f} / {cls[:1000].max():.4f}")

print(f"\npatch dtype={patch.dtype} shape={patch.shape}")
print(f"patch[0,0,:5] = {patch[0,0,:5]}")
print(f"patch[0,0] norm = {np.linalg.norm(patch[0,0].astype(np.float32)):.4f}")

# Check npy header
with open('/root/cloud_run/work/features/cls.npy','rb') as f:
    print(f"\ncls.npy header: {f.read(80)}")
