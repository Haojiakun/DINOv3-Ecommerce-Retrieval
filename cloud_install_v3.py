import time
import sys
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
# torch from official CUDA index (the only reliable source for GPU torch)
# + Tsinghua for other deps via --extra-index-url
rc = run_stream(ssh,
    "python3 -m pip install --timeout 300 --retries 5 "
    "torch torchvision "
    "--index-url https://download.pytorch.org/whl/cu121 "
    "--extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>&1",
    timeout=1800)
ssh.close()
print(f"torch rc={rc}")

if rc == 0:
    ssh = connect()
    rc2 = run_stream(ssh,
        "python3 -m pip install timm transformers pyarrow pandas scikit-learn tqdm pillow "
        "-i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1",
        timeout=600)
    ssh.close()
    print(f"deps rc={rc2}")

# Verify
ssh = connect()
stdin, stdout, _ = ssh.exec_command(
    "python3 -c 'import torch,timm,transformers; print(f\"OK torch={torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0)}\")' 2>&1",
    timeout=30, get_pty=True)
print("\n=== verify ===")
print(stdout.read().decode())
ssh.close()
