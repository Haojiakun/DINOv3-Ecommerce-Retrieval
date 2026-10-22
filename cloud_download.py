import time
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
sftp = ssh.open_sftp()
sftp.get("/root/cloud_run/outputs/cloud_results_N52591_bs256_ep50.md",
         r"d:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\cloud_run\outputs\cloud_results_N52591_bs256_ep50.md")
print("downloaded results")
sftp.close()
ssh.close()
