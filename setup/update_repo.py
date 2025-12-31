#!/usr/bin/env python3
"""
setup/update_repo.py

Safe repo updater for live_cam_changer with logging.
- Pulls latest code from remote.
- Does NOT touch models/onnx or models/tensorrt.
- Logs output to setup/logs/
"""

import subprocess
from pathlib import Path
import sys
import time

# --- Config ---
REPO_DIR = Path("/workspace/live_cam_changer")  # adjust if needed
BRANCH = "main"  # branch to update
LOG_DIR = Path("setup/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[update_repo {ts}] {msg}"
    print(line)
    with open(LOG_DIR / f"update-{int(time.time())}.log", "a") as f:
        f.write(line + "\n")

def run_cmd(cmd, cwd=None):
    """Run command, capture output, and log it."""
    log(f"Running: {cmd}")
    proc = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    log(f"stdout:\n{proc.stdout}")
    log(f"stderr:\n{proc.stderr}")
    if proc.returncode != 0:
        log(f"Command failed: {cmd} (rc={proc.returncode})")
        sys.exit(1)
    return proc.stdout.strip()

def update_repo():
    if not REPO_DIR.exists():
        log(f"Repo directory does not exist: {REPO_DIR}")
        sys.exit(1)

    log(f"Fetching latest changes in {REPO_DIR}...")
    run_cmd("git fetch origin", cwd=REPO_DIR)

    log(f"Resetting branch {BRANCH} to match origin...")
    run_cmd(f"git reset --hard origin/{BRANCH}", cwd=REPO_DIR)

    log("Repo update complete.")

if __name__ == "__main__":
    update_repo()
