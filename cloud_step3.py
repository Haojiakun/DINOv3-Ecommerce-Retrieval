import time
import sys
import paramiko
from paramiko import SSHClient, AutoAddPolicy

HOST = "cpod-1wqalz8nohiv.podtcp.compshare.cn"
PORT = 23246
USER = "root"
PASS = "45i27fPwzA9I160q"

LOCAL_PARQUET = r"d:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\work\deepfashion_subset.parquet"

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
# Kill any running python
stdin, stdout, _ = ssh.exec_command("pkill -f python3; sleep 1; echo killed", timeout=10, get_pty=True)
print(stdout.read().decode())

# Upload new subset parquet
print(f"Uploading new subset parquet ({__import__('os').path.getsize(LOCAL_PARQUET)/1024/1024:.1f} MB)...")
sftp = ssh.open_sftp()
sftp.put(LOCAL_PARQUET, "/root/cloud_run/work/deepfashion_subset.parquet")
sftp.close()
print("uploaded")

# Re-extract features (new subset has different images)
rc = run_stream(ssh,
    "cd /root/cloud_run/work && export HF_ENDPOINT=https://hf-mirror.com && "
    "PARQUET=deepfashion_subset.parquet python3 extract_features.py 2>&1 | tail -15",
    timeout=300)
print(f"extract_features rc={rc}")

if rc == 0:
    rc2 = run_stream(ssh,
        "cd /root/cloud_run/work && export HF_ENDPOINT=https://hf-mirror.com && "
        "PARQUET=deepfashion_subset.parquet python3 build_text.py",
        timeout=300)
    print(f"build_text rc={rc2}")

if rc == 0 and rc2 == 0:
    rc3 = run_stream(ssh,
        "cd /root/cloud_run/work && N=5003 BATCH=256 EPOCHS=50 SEEDS=3 python3 run_cloud.py",
        timeout=1800)
    print(f"run_cloud 5K rc={rc3}")

ssh.close()
