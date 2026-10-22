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
stdin, stdout, _ = ssh.exec_command("uptime; who -b", timeout=10, get_pty=True)
print(f"uptime: {stdout.read().decode()}")

# Check if features dir was modified recently
stdin, stdout, _ = ssh.exec_command("ls -la --time-style=full-iso /root/cloud_run/work/features/cls.npy", timeout=10, get_pty=True)
print(f"cls.npy: {stdout.read().decode()}")

# Check if there's enough memory
stdin, stdout, _ = ssh.exec_command("free -h; df -h /", timeout=10, get_pty=True)
print(f"resources:\n{stdout.read().decode()}")

ssh.close()
