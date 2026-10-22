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
print("=== torch install log (tail) ===")
stdin, stdout, _ = ssh.exec_command("tail -5 /tmp/install_torch.log 2>/dev/null", timeout=15, get_pty=True)
print(stdout.read().decode())

print("=== deps install log (tail) ===")
stdin, stdout, _ = ssh.exec_command("tail -5 /tmp/install_deps.log 2>/dev/null", timeout=15, get_pty=True)
print(stdout.read().decode())

print("=== verify ===")
stdin, stdout, _ = ssh.exec_command(
    "python3 -c 'import torch, timm, transformers; print(f\"torch={torch.__version__} cuda={torch.cuda.is_available()}\")' 2>&1",
    timeout=30, get_pty=True)
print(stdout.read().decode())

print("=== running procs ===")
stdin, stdout, _ = ssh.exec_command("ps aux | grep '[p]ip3 install' | wc -l", timeout=10, get_pty=True)
print(f"pip processes running: {stdout.read().decode().strip()}")

ssh.close()
