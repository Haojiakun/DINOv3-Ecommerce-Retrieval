import time
import paramiko
from paramiko import SSHClient, AutoAddPolicy

HOST = "cpod-1wqalz8nohiv.podtcp.compshare.cn"
PORT = 23246
USER = "root"
PASS = "45i27fPwzA9I160q"

def connect(retries=8):
    for i in range(retries):
        try:
            ssh = SSHClient()
            ssh.set_missing_host_key_policy(AutoAddPolicy())
            ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=60, banner_timeout=60)
            return ssh
        except Exception as e:
            print(f"  retry {i+1}/{retries}: {str(e)[:80]}")
            time.sleep(8)
    raise RuntimeError("Failed to connect")

def run_bg(ssh, cmd, logfile):
    """Run command in background, return immediately."""
    full = f"nohup bash -c '{cmd}' > {logfile} 2>&1 & echo $!"
    print(f"\n[bg] {cmd[:100]}...")
    stdin, stdout, stderr = ssh.exec_command(full, timeout=30, get_pty=True)
    pid = stdout.read().decode().strip()
    print(f"  PID={pid}, log={logfile}")
    return pid

def wait_for(ssh, logfile, keyword, timeout=1800, poll=15):
    """Poll logfile until keyword appears or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            stdin, stdout, stderr = ssh.exec_command(f"tail -3 {logfile}", timeout=15)
            out = stdout.read().decode()
            if out.strip():
                print(f"  [{int(time.time()-start)}s] {out.strip()[-200:]}")
            if keyword and keyword in out:
                print(f"  -> found '{keyword}', done")
                return True
            # Check if process still running
            stdin2, stdout2, _ = ssh.exec_command(f"grep -c '{keyword}' {logfile} 2>/dev/null; ps aux | grep -c '[p]ip3 install'", timeout=10)
            cnt = stdout2.read().decode().strip()
        except Exception as e:
            print(f"  poll error: {e}")
        time.sleep(poll)
    print(f"  timeout after {timeout}s")
    return False

ssh = connect()

# Install torch (large, ~2GB) in background
log_torch = "/tmp/install_torch.log"
run_bg(ssh, "python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121", log_torch)
print("\nWaiting for torch install...")
wait_for(ssh, log_torch, "Successfully installed", timeout=900, poll=20)

# Install other deps
log_deps = "/tmp/install_deps.log"
run_bg(ssh, "python3 -m pip install timm transformers pyarrow pandas scikit-learn tqdm pillow", log_deps)
print("\nWaiting for deps install...")
wait_for(ssh, log_deps, "Successfully installed", timeout=600, poll=15)

# Verify
print("\n=== Verify ===")
stdin, stdout, _ = ssh.exec_command(
    "python3 -c 'import torch, timm, transformers; print(f\"torch={torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0)}\")'",
    timeout=30, get_pty=True)
print(stdout.read().decode())

ssh.close()
print("\n=== All deps installed ===")
