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

# Check supervisord config
stdin, stdout, _ = ssh.exec_command("cat /etc/supervisor/supervisord.conf 2>/dev/null | head -40", timeout=10, get_pty=True)
print(f"=== supervisord.conf ===\n{stdout.read().decode()}")

# Check supervisor conf.d
stdin, stdout, _ = ssh.exec_command("ls -la /etc/supervisor/conf.d/ 2>/dev/null; cat /etc/supervisor/conf.d/*.conf 2>/dev/null", timeout=10, get_pty=True)
print(f"=== conf.d ===\n{stdout.read().decode()}")

# Check if there's a process killer
stdin, stdout, _ = ssh.exec_command("crontab -l 2>/dev/null; ls /etc/cron* 2>/dev/null", timeout=10, get_pty=True)
print(f"=== cron ===\n{stdout.read().decode()}")

ssh.close()
