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

# Check uptime (did server reboot?)
stdin, stdout, _ = ssh.exec_command("uptime", timeout=10, get_pty=True)
print(f"uptime: {stdout.read().decode().strip()}")

# Kill any leftover
ssh.exec_command("pkill -f extract_features 2>/dev/null; sleep 1")

# Ensure script exists
sftp = ssh.open_sftp()
script = """#!/bin/bash
cd /root/cloud_run/work
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
echo "=== FULL 52K START $(date) ==="
echo "[1/3] extract_features..."
PARQUET=deepfashion_inshop.parquet python3 extract_features.py
echo "extract rc=$?"
echo "[2/3] build_text..."
PARQUET=deepfashion_inshop.parquet python3 build_text.py
echo "build_text rc=$?"
echo "[3/3] run_cloud..."
N=52591 BATCH=256 EPOCHS=50 SEEDS=3 python3 run_cloud.py
echo "run_cloud rc=$?"
echo "=== DONE $(date) ==="
"""
with sftp.file("/root/run_full.sh", "w") as f:
    f.write(script)
sftp.close()
ssh.exec_command("chmod +x /root/run_full.sh")

# Launch in tmux
ssh.exec_command("tmux kill-session -t full_run 2>/dev/null")
time.sleep(1)
ssh.exec_command("tmux new-session -d -s full_run 'bash /root/run_full.sh'")
time.sleep(3)

stdin, stdout, _ = ssh.exec_command("tmux ls 2>&1", timeout=10, get_pty=True)
print(f"tmux sessions: {stdout.read().decode().strip()}")

stdin, stdout, _ = ssh.exec_command("tmux capture-pane -t full_run -p | tail -10", timeout=10, get_pty=True)
print(f"\n=== output ===\n{stdout.read().decode()}")

ssh.close()
print("\n=== tmux 'full_run' launched ===")
