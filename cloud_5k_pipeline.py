import time
import sys
import paramiko
from paramiko import SSHClient, AutoAddPolicy

HOST = "cpod-1wqalz8nohiv.podtcp.compshare.cn"
PORT = 23246
USER = "root"
PASS = "45i27fPwzA9I160q"

def connect(retries=15):
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
sftp = ssh.open_sftp()
for f in ["extract_features.py", "build_text.py", "run_cloud.py", "baselines_5k.py"]:
    sftp.put(rf"d:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\cloud_run\work\{f}",
             f"/root/cloud_run/work/{f}")
print("uploaded all scripts")
sftp.close()

# Phase 2: 5K pipeline
# 1. Extract 5K DINOv3 features (L2 normalized) -> features_5k/
print("\n=== [1/4] Extract 5K DINOv3 features ===")
stdin, stdout, stderr = ssh.exec_command(
    "cd /root/cloud_run/work && export HF_ENDPOINT=https://hf-mirror.com && "
    "PARQUET=deepfashion_subset.parquet OUT_DIR=features_5k python3 -u extract_features.py",
    timeout=600, get_pty=True)
for line in iter(stdout.readline, ""):
    print(line, end="")
    sys.stdout.flush()
rc = stdout.channel.recv_exit_status()
print(f"[exit={rc}]")

# Verify 5K features clean
print("\n=== verify 5K features ===")
stdin, stdout, _ = ssh.exec_command(
    "cd /root/cloud_run/work && python3 -c \"import numpy as np; "
    "c=np.load('features_5k/cls.npy'); p=np.load('features_5k/patch.npy'); "
    "print(f'cls shape={c.shape} max={c.max():.4f} nan={np.isnan(c).any()}'); "
    "print(f'patch shape={p.shape} max={p[:100].max():.4f} nan={np.isnan(p[:100]).any()}')\"",
    timeout=60, get_pty=True)
print(stdout.read().decode())

# 2. Build 5K CLIP text features
print("\n=== [2/4] Build 5K CLIP text features ===")
stdin, stdout, stderr = ssh.exec_command(
    "cd /root/cloud_run/work && export HF_ENDPOINT=https://hf-mirror.com && "
    "PARQUET=deepfashion_subset.parquet OUT_DIR=features_5k python3 -u build_text.py",
    timeout=300, get_pty=True)
for line in iter(stdout.readline, ""):
    print(line, end="")
    sys.stdout.flush()
rc = stdout.channel.recv_exit_status()
print(f"[exit={rc}]")

# 3. Train 5K: 3 seeds x 50 epochs
print("\n=== [3/4] Train 5K (3 seeds x 50 epochs, bs=256) ===")
stdin, stdout, stderr = ssh.exec_command(
    "cd /root/cloud_run/work && N=5003 BATCH=256 EPOCHS=50 SEEDS=3 FEAT_DIR=features_5k python3 -u run_cloud.py",
    timeout=3600, get_pty=True)
for line in iter(stdout.readline, ""):
    print(line, end="")
    sys.stdout.flush()
rc = stdout.channel.recv_exit_status()
print(f"[exit={rc}]")

# 4. Run 5K baselines (CLIP/SigLIP)
print("\n=== [4/4] 5K baselines (CLIP/SigLIP) ===")
stdin, stdout, stderr = ssh.exec_command(
    "cd /root/cloud_run/work && python3 -u baselines_5k.py",
    timeout=1200, get_pty=True)
for line in iter(stdout.readline, ""):
    print(line, end="")
    sys.stdout.flush()
rc = stdout.channel.recv_exit_status()
print(f"[exit={rc}]")

ssh.close()
print("\n=== 5K pipeline DONE ===")
