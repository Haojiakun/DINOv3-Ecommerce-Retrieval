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

def run(ssh, cmd, timeout=60):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout, get_pty=True)
    out = stdout.read().decode()
    return out

ssh = connect()

# Kill previous runs
print("=== cleanup ===")
print(run(ssh, "pkill -f 'extract_features\\|run_full_pipeline\\|run_cloud' 2>/dev/null; sleep 1; echo cleaned"))

# Check tmux
print("=== tmux check ===")
out = run(ssh, "which tmux || apt-get install -y tmux 2>&1 | tail -3", timeout=120)
print(out)
out = run(ssh, "tmux -V")
print(f"tmux version: {out.strip()}")

# Kill old tmux session if exists
run(ssh, "tmux kill-session -t full_run 2>/dev/null; echo ok")

# Create the pipeline script
script = """#!/bin/bash
cd /root/cloud_run/work
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

echo "=== FULL 52K PIPELINE START $(date) ==="
echo ""
echo "[1/3] extract_features (full 52K)..."
PARQUET=deepfashion_inshop.parquet python3 extract_features.py
echo "extract_features rc=$?"
echo ""
echo "[2/3] build_text (full 52K)..."
PARQUET=deepfashion_inshop.parquet python3 build_text.py
echo "build_text rc=$?"
echo ""
echo "[3/3] run_cloud (full 52K, 3 seeds, bs=256)..."
N=52591 BATCH=256 EPOCHS=50 SEEDS=3 python3 run_cloud.py
echo "run_cloud rc=$?"
echo ""
echo "=== DONE $(date) ==="
"""
sftp = ssh.open_sftp()
with sftp.file("/root/run_full.sh", "w") as f:
    f.write(script)
sftp.close()
run(ssh, "chmod +x /root/run_full.sh")

# Create tmux session and run
print("\n=== launch tmux ===")
run(ssh, "tmux new-session -d -s full_run 'bash /root/run_full.sh'")
time.sleep(2)
out = run(ssh, "tmux ls")
print(f"sessions: {out.strip()}")

# Capture initial output
time.sleep(3)
out = run(ssh, "tmux capture-pane -t full_run -p | tail -20")
print(f"\n=== initial output ===\n{out}")

ssh.close()
print("\n=== tmux session 'full_run' launched ===")
print("Monitor with: tmux capture-pane -t full_run -p")
