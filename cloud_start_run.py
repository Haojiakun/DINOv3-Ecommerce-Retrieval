import time
import os
import paramiko
from paramiko import SSHClient, AutoAddPolicy

HOST = "cpod-1wqalz8nohiv.podtcp.compshare.cn"
PORT = 23246
USER = "root"
PASS = "45i27fPwzA9I160q"

LOCAL = r"d:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\cloud_run"

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

# Upload updated build_text.py and run_all.sh
sftp.put(os.path.join(LOCAL, "work", "build_text.py"), "/root/cloud_run/work/build_text.py")
print("uploaded build_text.py")
sftp.put(os.path.join(LOCAL, "run_all.sh"), "/root/cloud_run/run_all.sh")
print("uploaded run_all.sh")
sftp.close()

# Make run_all.sh executable and start it in background
stdin, stdout, stderr = ssh.exec_command(
    "chmod +x /root/cloud_run/run_all.sh && "
    "cd /root/cloud_run && "
    "nohup bash run_all.sh > /root/run_all.log 2>&1 & echo $!",
    timeout=30, get_pty=True)
pid = stdout.read().decode().strip()
print(f"run_all.sh started with PID={pid}")

# Give it a moment and check initial output
time.sleep(5)
stdin, stdout, _ = ssh.exec_command("head -30 /root/run_all.log 2>/dev/null", timeout=15, get_pty=True)
print("\n=== initial log ===")
print(stdout.read().decode())

ssh.close()
