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
            print(f"  connect retry {i+1}/{retries}: {str(e)[:80]}")
            time.sleep(10)
    raise RuntimeError("connect failed")

def run_foreground(ssh, cmd, timeout=1800):
    """Run command and stream output."""
    print(f"\n[run] {cmd[:100]}")
    try:
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout, get_pty=True)
        # Read line by line
        for line in iter(stdout.readline, ""):
            print(line, end="")
        rc = stdout.channel.recv_exit_status()
        print(f"\n[exit={rc}]")
        return rc
    except Exception as e:
        print(f"  error: {e}")
        return -1

# Step 1: torch
ssh = connect()
rc = run_foreground(ssh,
    "python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -8",
    timeout=1200)
ssh.close()
print(f"torch install rc={rc}")

if rc == 0:
    # Step 2: other deps
    ssh = connect()
    rc2 = run_foreground(ssh,
        "python3 -m pip install timm transformers pyarrow pandas scikit-learn tqdm pillow 2>&1 | tail -8",
        timeout=600)
    ssh.close()
    print(f"deps install rc={rc2}")

# Verify
ssh = connect()
stdin, stdout, _ = ssh.exec_command(
    "python3 -c 'import torch,timm,transformers; print(f\"OK torch={torch.__version__} cuda={torch.cuda.is_available()}\")' 2>&1",
    timeout=30, get_pty=True)
print("\n=== verify ===")
print(stdout.read().decode())
ssh.close()
