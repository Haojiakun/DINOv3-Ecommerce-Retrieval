import time
import paramiko
from paramiko import SSHClient, AutoAddPolicy

HOST = "cpod-1wqalz8nohiv.podtcp.compshare.cn"
PORT = 23246
USER = "root"
PASS = "45i27fPwzA9I160q"

def connect(retries=10):
    for i in range(retries):
        try:
            ssh = SSHClient()
            ssh.set_missing_host_key_policy(AutoAddPolicy())
            ssh.connect(HOST, port=PORT, username=USER, password=PASS,
                        timeout=60, banner_timeout=60, auth_timeout=60)
            return ssh
        except Exception as e:
            print(f"  retry {i+1}: {str(e)[:60]}")
            time.sleep(10)
    raise RuntimeError("connect failed")

ssh = connect()
check = """
import numpy as np
patch = np.load('/root/cloud_run/work/features/patch.npy', mmap_mode='r')
cls = np.load('/root/cloud_run/work/features/cls.npy', mmap_mode='r')
text = np.load('/root/cloud_run/work/features/text_feats.npy', mmap_mode='r')
print(f"patch dtype={patch.dtype} shape={patch.shape}")
print(f"cls dtype={cls.dtype} shape={cls.shape}")
print(f"text dtype={text.dtype} shape={text.shape}")

# Check for nan/inf in chunks (avoid loading all into RAM)
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
    # sample stats
    s = np.array(arr[:100])
    print(f"  sample min={s.min():.4f} max={s.max():.4f} mean={s.mean():.4f} std={s.std():.4f}")
"""
stdin, stdout, _ = ssh.exec_command(f"python3 -c {repr(check)}", timeout=120, get_pty=True)
print(stdout.read().decode())
ssh.close()
