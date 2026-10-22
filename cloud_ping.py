import time
import paramiko
from paramiko import SSHClient, AutoAddPolicy

HOST = "cpod-1wqalz8nohiv.podtcp.compshare.cn"
PORT = 23246
USER = "root"
PASS = "45i27fPwzA9I160q"

for attempt in range(20):
    try:
        ssh = SSHClient()
        ssh.set_missing_host_key_policy(AutoAddPolicy())
        ssh.connect(HOST, port=PORT, username=USER, password=PASS,
                    timeout=30, banner_timeout=30, auth_timeout=30)
        print(f"[attempt {attempt+1}] CONNECTED!")
        stdin, stdout, _ = ssh.exec_command("uptime", timeout=10, get_pty=True)
        print(f"uptime: {stdout.read().decode().strip()}")
        ssh.close()
        break
    except Exception as e:
        print(f"[attempt {attempt+1}] {str(e)[:80]}")
        time.sleep(30)
else:
    print("All attempts failed")
