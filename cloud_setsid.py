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
sftp.put(r"d:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\cloud_run\work\extract_features.py",
         "/root/cloud_run/work/extract_features.py")
print("uploaded fixed extract_features.py")
sftp.close()

# Clean up
ssh.exec_command("pkill -f extract_features 2>/dev/null; tmux kill-server 2>/dev/null; sleep 2")

# Write pipeline script
script = """#!/bin/bash
exec &>> /root/full_pipeline.log
echo "=== START $(date) ==="
cd /root/cloud_run/work
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

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
sftp = ssh.open_sftp()
with sftp.file("/root/run_full.sh", "w") as f:
    f.write(script)
sftp.close()
ssh.exec_command("chmod +x /root/run_full.sh")

# Launch with setsid (fully detached, survives SSH/supervisord)
stdin, stdout, _ = ssh.exec_command(
    "setsid bash /root/run_full.sh < /dev/null > /dev/null 2>&1 & echo PID=$!",
    timeout=15, get_pty=True)
print(f"Launched: {stdout.read().decode().strip()}")

time.sleep(5)
stdin, stdout, _ = ssh.exec_command("ps aux | grep '[r]un_full\\|[e]xtract_features' | head -5", timeout=10, get_pty=True)
print(f"Processes:\n{stdout.read().decode()}")

ssh.close()
print("\n=== setsid pipeline launched. Log: /root/full_pipeline.log ===")
