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

# Run the full pipeline in background using nohup with proper shell
cmd = """cd /root/cloud_run/work && \
export HF_ENDPOINT=https://hf-mirror.com && \
export PYTHONUNBUFFERED=1 && \
echo "=== build_text 5K ===" && \
PARQUET=deepfashion_subset.parquet python3 build_text.py 2>&1 && \
echo "=== run_cloud 5K ===" && \
N=5000 BATCH=256 EPOCHS=50 SEEDS=3 python3 run_cloud.py 2>&1 && \
echo "=== extract_features FULL ===" && \
PARQUET=deepfashion_inshop.parquet python3 extract_features.py 2>&1 && \
echo "=== build_text FULL ===" && \
PARQUET=deepfashion_inshop.parquet python3 build_text.py 2>&1 && \
echo "=== run_cloud FULL ===" && \
N=52591 BATCH=256 EPOCHS=50 SEEDS=3 python3 run_cloud.py 2>&1 && \
echo "=== ALL DONE ==="
"""

# Write the command to a script file first, then run it
sftp = ssh.open_sftp()
with sftp.file("/root/run_pipeline.sh", "w") as f:
    f.write("#!/bin/bash\n")
    f.write(cmd)
sftp.close()

stdin, stdout, stderr = ssh.exec_command(
    "chmod +x /root/run_pipeline.sh && "
    "nohup /bin/bash /root/run_pipeline.sh > /root/run_all.log 2>&1 & "
    "echo PID=$!",
    timeout=30, get_pty=True)
result = stdout.read().decode()
print(f"Started: {result}")

time.sleep(3)
stdin, stdout, _ = ssh.exec_command("ps aux | grep '[r]un_pipeline' | head -3", timeout=10, get_pty=True)
print(f"Process: {stdout.read().decode()}")

ssh.close()
print("\nPipeline running in background. Log: /root/run_all.log")
