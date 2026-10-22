"""Check cloud GPU environment."""
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


def run(ssh, cmd, timeout=120):
    print(f"\n$ {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout, get_pty=True)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    if out.strip():
        print(out)
    if err.strip() and "Warning" not in err:
        print(f"[err] {err}")
    return out


ssh = connect()
run(ssh, "nvidia-smi")
run(ssh, "python --version; python3 --version; which python python3")
run(ssh, "python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)' 2>&1")
run(ssh, "pip list 2>/dev/null | grep -iE 'timm|transformers|pyarrow|pandas|sklearn|tqdm|torch'")
run(ssh, "df -h /root")
run(ssh, "free -h")
ssh.close()
