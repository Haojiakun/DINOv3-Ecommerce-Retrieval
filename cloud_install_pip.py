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
    print(f"\n$ {cmd[:120]}{'...' if len(cmd)>120 else ''}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout, get_pty=True)
    out = stdout.read().decode("utf-8", errors="replace")
    print(out[-3000:] if len(out) > 3000 else out)
    return out

ssh = connect()
run(ssh, "apt-get update -qq && apt-get install -y -qq python3-pip python3-venv 2>&1 | tail -5", timeout=300)
run(ssh, "python3 -m pip --version")
ssh.close()
