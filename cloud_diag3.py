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

# Check file sizes
stdin, stdout, _ = ssh.exec_command("ls -lh /root/cloud_run/work/features/cls.npy /root/cloud_run/work/features/patch.npy", timeout=10, get_pty=True)
print(f"files:\n{stdout.read().decode()}")

# Check dmesg for OOM
stdin, stdout, _ = ssh.exec_command("dmesg 2>/dev/null | grep -i 'oom\\|killed' | tail -10", timeout=10, get_pty=True)
out = stdout.read().decode()
print(f"dmesg OOM:\n{out or '(none)'}")

# Check syslog
stdin, stdout, _ = ssh.exec_command("grep -i 'oom\\|killed\\|tmux' /var/log/syslog 2>/dev/null | tail -10", timeout=10, get_pty=True)
out = stdout.read().decode()
print(f"syslog:\n{out or '(none)'}")

# Try reading cls with allow_pickle
stdin, stdout, _ = ssh.exec_command(
    "python3 -c \"import numpy as np; c=np.load('/root/cloud_run/work/features/cls.npy', allow_pickle=True); print(type(c), c.shape if hasattr(c,'shape') else 'no shape')\" 2>&1",
    timeout=15, get_pty=True)
print(f"cls with pickle: {stdout.read().decode().strip()}")

ssh.close()
