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
# Backup 5K features before full run overwrites them
stdin, stdout, _ = ssh.exec_command(
    "mkdir -p /root/cloud_run/work/features_5k && "
    "cp /root/cloud_run/work/features/cls.npy /root/cloud_run/work/features_5k/ && "
    "cp /root/cloud_run/work/features/patch.npy /root/cloud_run/work/features_5k/ && "
    "cp /root/cloud_run/work/features/text_feats.npy /root/cloud_run/work/features_5k/ && "
    "cp /root/cloud_run/work/features/meta.json /root/cloud_run/work/features_5k/ && "
    "ls -lh /root/cloud_run/work/features_5k/",
    timeout=30, get_pty=True)
print(stdout.read().decode())
ssh.close()
