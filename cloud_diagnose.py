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

print("=== run_all.sh exists? ===")
stdin, stdout, _ = ssh.exec_command("ls -la /root/cloud_run/run_all.sh", timeout=10, get_pty=True)
print(stdout.read().decode())

print("=== run_all.sh content (head) ===")
stdin, stdout, _ = ssh.exec_command("head -15 /root/cloud_run/run_all.sh", timeout=10, get_pty=True)
print(stdout.read().decode())

print("=== try run directly (first few lines) ===")
stdin, stdout, stderr = ssh.exec_command(
    "cd /root/cloud_run/work && export HF_ENDPOINT=https://hf-mirror.com && "
    "PARQUET=deepfashion_subset.parquet python3 extract_features.py 2>&1 | head -30",
    timeout=120, get_pty=True)
out = stdout.read().decode()
err = stderr.read().decode()
print(out)
if err:
    print(f"[stderr] {err[:500]}")

ssh.close()
