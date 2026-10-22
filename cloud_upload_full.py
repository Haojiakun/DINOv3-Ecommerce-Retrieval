import time
import os
import paramiko
from paramiko import SSHClient, AutoAddPolicy

HOST = "cpod-1wqalz8nohiv.podtcp.compshare.cn"
PORT = 23246
USER = "root"
PASS = "45i27fPwzA9I160q"

LOCAL_PARQUET = r"d:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\work\deepfashion_inshop.parquet"
LOCAL_FEATURES = r"d:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\cloud_run\work\extract_features.py"

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

# Upload full parquet (206MB)
print(f"Uploading {os.path.getsize(LOCAL_PARQUET)/1024/1024:.1f} MB parquet...")
sftp.put(LOCAL_PARQUET, "/root/cloud_run/work/deepfashion_inshop.parquet")
print("  done")

# Upload updated extract_features.py
sftp.put(LOCAL_FEATURES, "/root/cloud_run/work/extract_features.py")
print("uploaded extract_features.py")

sftp.close()

# Verify
stdin, stdout, _ = ssh.exec_command("ls -lh /root/cloud_run/work/*.parquet", timeout=15, get_pty=True)
print(stdout.read().decode())
ssh.close()
