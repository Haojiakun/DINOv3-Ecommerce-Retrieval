"""SSH helper: connect, upload, run commands on cloud GPU."""
import os
import sys
import paramiko
from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient

HOST = "cpod-1wqalz8nohiv.podtcp.compshare.cn"
PORT = 23246
USER = "root"
PASS = "45i27fPwzA9I160q"

LOCAL_CLOUD = r"d:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\cloud_run"
REMOTE_DIR = "/root/cloud_run"


def connect():
    ssh = SSHClient()
    ssh.set_missing_host_key_policy(AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    print(f"[ssh] connected to {USER}@{HOST}:{PORT}")
    return ssh


def upload_dir(ssh, local_dir, remote_dir):
    """Upload a directory tree using SFTP."""
    sftp = ssh.open_sftp()
    # Create remote dir
    try:
        sftp.stat(remote_dir)
    except FileNotFoundError:
        sftp.mkdir(remote_dir)

    for root, dirs, files in os.walk(local_dir):
        rel = os.path.relpath(root, local_dir)
        remote_path = remote_dir if rel == "." else os.path.join(remote_dir, rel).replace("\\", "/")
        try:
            sftp.stat(remote_path)
        except FileNotFoundError:
            sftp.mkdir(remote_path)
        for f in files:
            if f.endswith(".pyc"):
                continue
            local_file = os.path.join(root, f)
            remote_file = os.path.join(remote_path, f).replace("\\", "/")
            sftp.put(local_file, remote_file)
            size = os.path.getsize(local_file) / 1024 / 1024
            print(f"  uploaded {remote_file} ({size:.1f} MB)")
    sftp.close()
    print(f"[upload] done -> {remote_dir}")


def run(ssh, cmd, timeout=600):
    print(f"\n[run] {cmd[:120]}{'...' if len(cmd)>120 else ''}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout, get_pty=True)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    if out.strip():
        print(out[-2000:] if len(out) > 2000 else out)
    if err.strip() and "Warning" not in err:
        print(f"[stderr] {err[-500:]}")
    return out, err


def main():
    ssh = connect()

    # Step 1: upload cloud_run
    print("\n=== Step 1: Upload cloud_run/ ===")
    upload_dir(ssh, LOCAL_CLOUD, REMOTE_DIR)

    # Step 2: verify
    out, _ = run(ssh, f"ls -la {REMOTE_DIR} && echo '---' && du -sh {REMOTE_DIR}")

    ssh.close()
    print("\n=== Upload complete ===")


if __name__ == "__main__":
    main()
