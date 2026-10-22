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
sftp = ssh.open_sftp()

# Upload fixed scripts
sftp.put(r"d:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\cloud_run\work\extract_features.py",
         "/root/cloud_run/work/extract_features.py")
print("uploaded extract_features.py (memmap streaming)")

sftp.put(r"d:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\cloud_run\work\run_cloud.py",
         "/root/cloud_run/work/run_cloud.py")
print("uploaded run_cloud.py (float16 memmap)")

sftp.close()

# Kill old, restart tmux
ssh.exec_command("pkill -f extract_features 2>/dev/null; tmux kill-session -t full_run 2>/dev/null; sleep 2")

script = """#!/bin/bash
cd /root/cloud_run/work
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
echo "=== FULL 52K START $(date) ==="
echo "[1/3] extract_features (memmap streaming)..."
PARQUET=deepfashion_inshop.parquet python3 extract_features.py
echo "extract rc=$?"
echo "[2/3] build_text..."
PARQUET=deepfashion_inshop.parquet python3 build_text.py
echo "build_text rc=$?"
echo "[3/3] run_cloud (float16 memmap)..."
N=52591 BATCH=256 EPOCHS=50 SEEDS=3 python3 run_cloud.py
echo "run_cloud rc=$?"
echo "=== DONE $(date) ==="
"""
sftp = ssh.open_sftp()
with sftp.file("/root/run_full.sh", "w") as f:
    f.write(script)
sftp.close()
ssh.exec_command("chmod +x /root/run_full.sh")

ssh.exec_command("tmux new-session -d -s full_run 'bash /root/run_full.sh'")
time.sleep(3)

stdin, stdout, _ = ssh.exec_command("tmux capture-pane -t full_run -p | tail -10", timeout=10, get_pty=True)
print(f"\n=== initial output ===\n{stdout.read().decode()}")

ssh.close()
print("\n=== tmux 'full_run' launched with OOM fixes ===")
