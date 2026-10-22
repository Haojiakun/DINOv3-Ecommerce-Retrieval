import time
import paramiko
from paramiko import SSHClient, AutoAddPolicy

HOST = "cpod-1wqalz8nohiv.podtcp.compshare.cn"
PORT = 23246
USER = "root"
PASS = "45i27fPwzA9I160q"

def connect(retries=5):
    for i in range(retries):
        try:
            ssh = SSHClient()
            ssh.set_missing_host_key_policy(AutoAddPolicy())
            ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=60, banner_timeout=60)
            return ssh
        except Exception as e:
            print(f"  retry {i+1}/{retries}: {e}")
            time.sleep(5)
    raise RuntimeError("Failed to connect")

def run(ssh, cmd, timeout=1800):
    print(f"\n$ {cmd[:120]}{'...' if len(cmd)>120 else ''}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout, get_pty=True)
    out = stdout.read().decode("utf-8", errors="replace")
    print(out[-3000:] if len(out) > 3000 else out)
    return out

ssh = connect()
# Install pip via get-pip.py
run(ssh, "curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py 2>&1 | tail -5", timeout=120)
run(ssh, "python3 -m pip --version")
ssh.close()
