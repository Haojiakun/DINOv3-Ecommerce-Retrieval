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
stdin, stdout, _ = ssh.exec_command("sed -n '85,110p' /root/cloud_run/work/extract_features.py", timeout=10, get_pty=True)
print("=== extract_features.py lines 85-110 ===")
print(stdout.read().decode())

# Also check if there's a __pycache__ that might be stale
stdin, stdout, _ = ssh.exec_command("ls -la /root/cloud_run/work/__pycache__/ 2>/dev/null; find /root/cloud_run -name '*.pyc' 2>/dev/null | head", timeout=10, get_pty=True)
print("=== pycache ===")
print(stdout.read().decode())

ssh.close()
