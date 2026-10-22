import time
import paramiko
from paramiko import SSHClient, AutoAddPolicy

HOST = "cpod-1wqalz8nohiv.podtcp.compshare.cn"
PORT = 23246
USER = "root"
PASS = "45i27fPwzA9I160q"

def connect(retries=15):
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
for cmd in [
    "uptime",
    "free -h | head -3",
    "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null",
    "ls -la /root/cloud_run/work/*.py 2>/dev/null",
    "ls -la /root/cloud_run/work/features/ 2>/dev/null",
    "ls -la /root/cloud_run/work/*.parquet 2>/dev/null",
    "ls -la /root/cloud_run/work/features_5k/ 2>/dev/null",
]:
    stdin, stdout, _ = ssh.exec_command(cmd, timeout=15, get_pty=True)
    print(f"$ {cmd}")
    print(stdout.read().decode())
ssh.close()
