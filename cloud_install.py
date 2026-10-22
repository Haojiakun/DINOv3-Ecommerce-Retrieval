"""Install dependencies on cloud GPU."""
import paramiko
from paramiko import SSHClient, AutoAddPolicy

HOST = "cpod-1wqalz8nohiv.podtcp.compshare.cn"
PORT = 23246
USER = "root"
PASS = "45i27fPwzA9I160q"


def connect():
    ssh = SSHClient()
    ssh.set_missing_host_key_policy(AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    return ssh


def run(ssh, cmd, timeout=1800):
    print(f"\n$ {cmd[:100]}{'...' if len(cmd)>100 else ''}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout, get_pty=True)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    print(out[-3000:] if len(out) > 3000 else out)
    if err.strip() and "Warning" not in err and "warning" not in err.lower():
        print(f"[err] {err[-500:]}")
    return out


ssh = connect()

# Install torch + CUDA 12.1
run(ssh, "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -5", timeout=600)

# Install other deps
run(ssh, "pip3 install timm transformers pyarrow pandas scikit-learn tqdm pillow 2>&1 | tail -10", timeout=600)

# Verify
run(ssh, "python3 -c 'import torch, timm, transformers; print(f\"torch={torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0)}\")' 2>&1")

ssh.close()
print("\n=== Install complete ===")
