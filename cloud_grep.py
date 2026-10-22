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
# Check if normalization code exists
stdin, stdout, _ = ssh.exec_command("grep -n 'normalize\\|F.normalize' /root/cloud_run/work/extract_features.py", timeout=10, get_pty=True)
print("=== grep normalize ===")
print(stdout.read().decode())

# Check file size and md5
stdin, stdout, _ = ssh.exec_command("ls -la /root/cloud_run/work/features/*.npy", timeout=10, get_pty=True)
print("=== feature files ===")
print(stdout.read().decode())

ssh.close()
