import time
import os
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
sftp.put(r"d:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\cloud_run\work\baselines_5k.py",
         "/root/cloud_run/work/baselines_5k.py")
sftp.close()
print("uploaded baselines_5k.py")

# Check if full extract_features is still running
stdin, stdout, _ = ssh.exec_command("ps aux | grep '[e]xtract_features' | wc -l", timeout=10, get_pty=True)
ext_running = stdout.read().decode().strip()
print(f"extract_features running: {ext_running}")

# Check features size (full vs 5k)
stdin, stdout, _ = ssh.exec_command("ls -lh /root/cloud_run/work/features/cls.npy", timeout=10, get_pty=True)
print(stdout.read().decode())

ssh.close()
