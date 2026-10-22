import time
import sys
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

def run_stream(ssh, cmd, timeout=1800):
    print(f"\n[run] {cmd[:100]}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line, end="")
        sys.stdout.flush()
    rc = stdout.channel.recv_exit_status()
    print(f"\n[exit={rc}]")
    return rc

ssh = connect()

# Step 1: build_text for 5K subset
rc = run_stream(ssh,
    "cd /root/cloud_run/work && export HF_ENDPOINT=https://hf-mirror.com && "
    "PARQUET=deepfashion_subset.parquet python3 build_text.py",
    timeout=300)
ssh.close()
print(f"build_text 5K rc={rc}")
