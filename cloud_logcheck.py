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
stdin, stdout, _ = ssh.exec_command("tail -15 /root/full_pipeline.log 2>/dev/null", timeout=15, get_pty=True)
print(f"=== log ===\n{stdout.read().decode()}")

stdin, stdout, _ = ssh.exec_command("ps aux | grep python3 | grep -v grep | head -5", timeout=10, get_pty=True)
print(f"=== procs ===\n{stdout.read().decode()}")

stdin, stdout, _ = ssh.exec_command("uptime", timeout=10, get_pty=True)
print(f"uptime: {stdout.read().decode().strip()}")

ssh.close()
