import time
import sys
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

def run_stream(ssh, cmd, timeout=1800):
    print(f"\n[run] {cmd[:120]}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout, get_pty=True)
    for line in iter(stdout.readline, ""):
        print(line, end="")
        sys.stdout.flush()
    rc = stdout.channel.recv_exit_status()
    print(f"\n[exit={rc}]")
    return rc

ssh = connect()
sftp = ssh.open_sftp()
sftp.put(r"d:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\cloud_run\work\build_text.py",
         "/root/cloud_run/work/build_text.py")
sftp.close()
print("uploaded fixed build_text.py")

# build_text 5K
rc = run_stream(ssh,
    "cd /root/cloud_run/work && export HF_ENDPOINT=https://hf-mirror.com && "
    "PARQUET=deepfashion_subset.parquet python3 build_text.py",
    timeout=300)
print(f"build_text rc={rc}")

if rc == 0:
    # run_cloud 5K
    rc2 = run_stream(ssh,
        "cd /root/cloud_run/work && N=5000 BATCH=256 EPOCHS=50 SEEDS=3 python3 run_cloud.py",
        timeout=1800)
    print(f"run_cloud 5K rc={rc2}")

ssh.close()
