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

# Run full pipeline with setsid to survive SSH disconnect
script = """#!/bin/bash
exec &>> /root/full_pipeline.log
echo "=== START $(date) ==="
cd /root/cloud_run/work
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

echo "[1/3] extract_features (full 52K)..."
PARQUET=deepfashion_inshop.parquet python3 extract_features.py
echo "extract done, rc=$?"

echo "[2/3] build_text (full 52K)..."
PARQUET=deepfashion_inshop.parquet python3 build_text.py
echo "build_text done, rc=$?"

echo "[3/3] run_cloud (full 52K, 3 seeds)..."
N=52591 BATCH=256 EPOCHS=50 SEEDS=3 python3 run_cloud.py
echo "run_cloud done, rc=$?"

echo "=== DONE $(date) ==="
"""

sftp = ssh.open_sftp()
with sftp.file("/root/run_full_pipeline.sh", "w") as f:
    f.write(script)
sftp.close()

# Launch with setsid to fully detach from SSH session
stdin, stdout, stderr = ssh.exec_command(
    "chmod +x /root/run_full_pipeline.sh && "
    "setsid bash /root/run_full_pipeline.sh < /dev/null > /dev/null 2>&1 & "
    "echo LAUNCHED_PID=$!",
    timeout=15, get_pty=True)
print(stdout.read().decode())

time.sleep(5)
stdin, stdout, _ = ssh.exec_command("ps aux | grep '[r]un_full_pipeline\\|[e]xtract_features' | head -5", timeout=10, get_pty=True)
print(f"Processes:\n{stdout.read().decode()}")

ssh.close()
print("\nFull pipeline launched with setsid. Will survive SSH disconnect.")
print("Monitor: /root/full_pipeline.log")
