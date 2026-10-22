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
            print(f"  retry {i+1}: {str(e)[:60]}")
            time.sleep(5)
    raise RuntimeError("connect failed")

ssh = connect()
print("=== /tmp/install_torch.log (full) ===")
stdin, stdout, _ = ssh.exec_command("cat /tmp/install_torch.log 2>/dev/null || echo 'NO FILE'", timeout=15, get_pty=True)
print(stdout.read().decode())

print("\n=== /tmp/install_deps.log (full) ===")
stdin, stdout, _ = ssh.exec_command("cat /tmp/install_deps.log 2>/dev/null || echo 'NO FILE'", timeout=15, get_pty=True)
print(stdout.read().decode())

print("\n=== pip list ===")
stdin, stdout, _ = ssh.exec_command("python3 -m pip list 2>/dev/null | head -20", timeout=15, get_pty=True)
print(stdout.read().decode())

ssh.close()
