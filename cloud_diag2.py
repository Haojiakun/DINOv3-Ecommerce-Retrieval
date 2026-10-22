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

print("=== all python processes ===")
stdin, stdout, _ = ssh.exec_command("ps aux | grep python3 | grep -v grep", timeout=10, get_pty=True)
print(stdout.read().decode() or "(none)")

print("=== features dir full ===")
stdin, stdout, _ = ssh.exec_command("ls -la /root/cloud_run/work/features/", timeout=10, get_pty=True)
print(stdout.read().decode())

print("=== check cls.npy shape ===")
stdin, stdout, _ = ssh.exec_command(
    "python3 -c \"import numpy as np; c=np.load('/root/cloud_run/work/features/cls.npy'); print('cls shape:', c.shape)\" 2>&1",
    timeout=15, get_pty=True)
print(stdout.read().decode())

print("=== outputs dir ===")
stdin, stdout, _ = ssh.exec_command("ls -la /root/cloud_run/outputs/ 2>/dev/null; ls -la /root/results_*.md 2>/dev/null", timeout=10, get_pty=True)
print(stdout.read().decode() or "(none)")

ssh.close()
