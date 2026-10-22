import time
import sys
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
            time.sleep(15)
    raise RuntimeError("connect failed")

ssh = connect()
print("Running FULL 52K: 3 seeds x 50 epochs...")
stdin, stdout, stderr = ssh.exec_command(
    "cd /root/cloud_run/work && N=52591 BATCH=256 EPOCHS=50 SEEDS=3 python3 -u run_cloud.py",
    timeout=7200, get_pty=True)
for line in iter(stdout.readline, ""):
    print(line, end="")
    sys.stdout.flush()
rc = stdout.channel.recv_exit_status()
print(f"\n[exit={rc}]")
ssh.close()
