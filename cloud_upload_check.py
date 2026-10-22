import time
import os
import paramiko
from paramiko import SSHClient, AutoAddPolicy

HOST = "cpod-1wqalz8nohiv.podtcp.compshare.cn"
PORT = 23246
USER = "root"
PASS = "45i27fPwzA9I160q"
LOCAL = r"d:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\cloud_run\work\run_cloud.py"

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

# Upload run_cloud.py
sftp = ssh.open_sftp()
sftp.put(LOCAL, "/root/cloud_run/work/run_cloud.py")
print("uploaded run_cloud.py")
sftp.close()

# Check torch install
stdin, stdout, _ = ssh.exec_command(
    "python3 -c 'import torch; print(f\"torch={torch.__version__} cuda={torch.cuda.is_available()}\")' 2>&1",
    timeout=30, get_pty=True)
out = stdout.read().decode()
print(f"\n=== torch check ===\n{out}")

if "No module" in out:
    # Check if still installing
    stdin, stdout, _ = ssh.exec_command("ps aux | grep '[p]ip3 install' | wc -l", timeout=10, get_pty=True)
    print(f"pip processes running: {stdout.read().decode().strip()}")

ssh.close()
