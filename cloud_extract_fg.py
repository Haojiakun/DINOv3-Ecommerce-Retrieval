import time
import sys
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
print("Running extract_features in foreground (timeout 30min)...")
stdin, stdout, stderr = ssh.exec_command(
    "cd /root/cloud_run/work && export HF_ENDPOINT=https://hf-mirror.com && "
    "PARQUET=deepfashion_inshop.parquet python3 -u extract_features.py",
    timeout=1800, get_pty=True)
for line in iter(stdout.readline, ""):
    print(line, end="")
    sys.stdout.flush()
rc = stdout.channel.recv_exit_status()
print(f"\n[exit={rc}]")

if rc == 0:
    # Verify
    stdin, stdout, _ = ssh.exec_command(
        "python3 -c \"import numpy as np; c=np.load('/root/cloud_run/work/features/cls.npy'); p=np.load('/root/cloud_run/work/features/patch.npy', mmap_mode='r'); print('cls:', c.shape, 'patch:', p.shape)\"",
        timeout=30, get_pty=True)
    print(f"\n=== verify ===\n{stdout.read().decode()}")

ssh.close()
