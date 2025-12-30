#!/usr/bin/env python3
"""
setup/setup.py

Idempotent setup script for GPU instance.

What it does:
- Reads setup/config.yaml
- Checks GPU (nvidia-smi) if enabled
- Optionally verifies trtexec is available
- Clones or pulls git repo
- Downloads models (ONNX / .pth) if missing
- Converts ONNX -> TensorRT (.trt) using trtexec if requested
- Leaves everything under configured paths:
    models_dir/onnx/..., models_dir/tensorrt/...
- Safe to re-run (skips work already done)

Run manually on the GPU machine:
    python3 setup/setup.py

Dependencies:
    pip3 install pyyaml requests
"""

import os
import sys
import yaml
import subprocess
import argparse
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse
from pathlib import Path
from dotenv import load_dotenv

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

try:
    import requests
except Exception:
    print("Missing dependency 'requests'. Please install: pip3 install requests")
    sys.exit(1)

# ---------------------------
# Helpers
# ---------------------------
def log(msg):
    print(f"[setup] {msg}")

def run_cmd(cmd, timeout=None, check=False):
    """Run shell command. cmd may be list or string."""
    if isinstance(cmd, (list, tuple)):
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    else:
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if proc.returncode != 0 and check:
        raise RuntimeError(f"Command failed: {cmd}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def download_file(url, out_path, chunk_size=1 << 20):
    """Download with resume fallback. Overwrites partially only on success."""
    out_path = Path(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    log(f"Downloading: {url} -> {out_path}")
    session = requests.Session()
    with session.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = resp.headers.get("Content-Length")
        if total:
            total = int(total)
            log(f"Content-Length: {total} bytes")
        with open(tmp, "wb") as f:
            downloaded = 0
            start = time.time()
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
            f.flush()
    tmp.replace(out_path)
    elapsed = time.time() - start
    log(f"Downloaded {out_path} ({downloaded} bytes) in {elapsed:.1f}s")

# ---------------------------
# Core logic
# ---------------------------
def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def check_gpu(cfg):
    if not cfg.get("system", {}).get("require_gpu", False):
        log("GPU not required by config.")
        return True
    cmd = cfg.get("system", {}).get("gpu_check_cmd", "nvidia-smi")
    try:
        proc = run_cmd([cmd], check=True)
        log(f"GPU check OK. nvidia-smi output (short):\n{proc.stdout.splitlines()[:10]}")
        return True
    except Exception as e:
        log(f"GPU check failed: {e}")
        return False

def check_trtexec(cfg):
    if not cfg.get("actions", {}).get("verify_trtexec", True):
        log("Skipping trtexec verification (config).")
        return True
    trtexec = cfg.get("system", {}).get("trtexec_cmd", "trtexec")
    proc = shutil.which(trtexec)
    if proc:
        log(f"Found trtexec at: {proc}")
        return True
    else:
        log("trtexec not found in PATH. You must install TensorRT or add trtexec to PATH.")
        return False

def git_clone_or_pull(repo_url, branch, dest):
    dest = Path(dest)
    if dest.exists() and (dest / ".git").exists():
        log(f"Repo exists at {dest}, pulling latest from branch '{branch}'")
        try:
            proc = run_cmd(["git", "-C", str(dest), "fetch"], check=True)
            proc = run_cmd(["git", "-C", str(dest), "checkout", branch], check=True)
            proc = run_cmd(["git", "-C", str(dest), "pull", "origin", branch], check=True)
            log("Git pull completed.")
        except Exception as e:
            log(f"Git pull failed: {e}")
            raise
    else:
        log(f"Cloning repo {repo_url} -> {dest}")
        ensure_dir(dest.parent)
        try:
            proc = run_cmd(["git", "clone", "--branch", branch, repo_url, str(dest)], check=True)
            log("Git clone completed.")
        except Exception as e:
            log(f"Git clone failed: {e}")
            raise

def ensure_models(cfg):
    paths = cfg["paths"]
    models_dir = Path(paths["models_dir"])
    onnx_dir = Path(paths["onnx_dir"])
    trt_dir = Path(paths["trt_dir"])
    ensure_dir(models_dir)
    ensure_dir(onnx_dir)
    ensure_dir(trt_dir)

    for m in cfg.get("models", []):
        mname = m.get("name")
        mtype = m.get("type")
        url = m.get("url", "").strip()
        filename = m.get("filename")
        if not filename:
            log(f"Model '{mname}' has no filename in config, skipping.")
            continue

        if mtype == "onnx":
            out_path = onnx_dir / filename
            if out_path.exists():
                log(f"ONNX model exists: {out_path} (skipping download)")
            else:
                if not url:
                    log(f"No URL for ONNX model '{mname}', skipping download.")
                else:
                    download_file(url, out_path)
        elif mtype in ("pth", "pt"):
            out_path = models_dir / filename
            if out_path.exists():
                log(f"PyTorch model exists: {out_path} (skipping download)")
            else:
                if not url:
                    log(f"No URL for PyTorch model '{mname}', skipping download.")
                else:
                    download_file(url, out_path)
        else:
            log(f"Unknown model type '{mtype}' for model '{mname}', skipping.")

def convert_models_to_trt(cfg):
    paths = cfg["paths"]
    onnx_dir = Path(paths["onnx_dir"])
    trt_dir = Path(paths["trt_dir"])
    trtexec = cfg.get("system", {}).get("trtexec_cmd", "trtexec")
    timeout = cfg.get("system", {}).get("trtexec_timeout_sec", 900)

    for m in cfg.get("models", []):
        if not m.get("convert_to_trt", False):
            continue
        if m.get("type") != "onnx":
            log(f"Model {m.get('name')} marked convert_to_trt but isn't ONNX type; skipping.")
            continue

        onnx_file = onnx_dir / m["filename"]
        trt_file = trt_dir / m["trt_filename"]

        if not onnx_file.exists():
            log(f"ONNX file missing for {m['name']}: {onnx_file} (cannot convert).")
            continue

        if trt_file.exists():
            log(f"TRT engine already exists: {trt_file} (skipping conversion).")
            continue

        # Build trtexec command
        extra_args = m.get("trtexec_args", "")
        cmd = f"{trtexec} --onnx={onnx_file} --saveEngine={trt_file} {extra_args}"
        log(f"Converting ONNX -> TensorRT for {m['name']}:")
        log(f"  command: {cmd}")
        try:
            proc = run_cmd(cmd, timeout=timeout)
            if proc.returncode == 0:
                log(f"Conversion succeeded for {m['name']}, saved: {trt_file}")
            else:
                log(f"Conversion returned non-zero code ({proc.returncode}). stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
                raise RuntimeError("trtexec failed")
        except Exception as e:
            log(f"Conversion failed for {m['name']}: {e}")
            # If conversion failed, ensure partial output is removed
            if trt_file.exists():
                try:
                    trt_file.unlink()
                    log(f"Removed partial trt file: {trt_file}")
                except Exception:
                    pass
            raise

def print_summary(cfg):
    paths = cfg["paths"]
    log("SETUP SUMMARY")
    log(f"Repo dir: {paths['repo_dir']}")
    log(f"Models dir: {paths['models_dir']}")
    log(f"ONNX dir: {paths['onnx_dir']}")
    log(f"TensorRT dir: {paths['trt_dir']}")
    for m in cfg.get("models", []):
        name = m.get("name")
        mtype = m.get("type")
        if mtype == "onnx":
            onnx_path = Path(paths["onnx_dir"]) / m.get("filename")
            trt_path = Path(paths["trt_dir"]) / m.get("trt_filename") if m.get("trt_filename") else None
            log(f"Model {name}: onnx={'exists' if onnx_path.exists() else 'MISSING'} trt={'exists' if trt_path and trt_path.exists() else 'MISSING' if m.get('convert_to_trt',False) else 'N/A'}")
        else:
            model_path = Path(paths["models_dir"]) / m.get("filename")
            log(f"Model {name}: file={'exists' if model_path.exists() else 'MISSING'}")

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Idempotent GPU instance setup script")
    parser.add_argument("--config", type=str, default="setup/config.yaml", help="Path to YAML config")
    parser.add_argument("--no-gpu-check", action="store_true", help="Skip GPU check")
    parser.add_argument("--skip-convert", action="store_true", help="Skip ONNX->TensorRT conversion step")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}")
        sys.exit(2)

    cfg = load_config(cfg_path)

    # Ensure directories are present
    ensure_dir(cfg["paths"]["models_dir"])
    ensure_dir(cfg["paths"]["onnx_dir"])
    ensure_dir(cfg["paths"]["trt_dir"])
    ensure_dir(cfg["paths"]["repo_dir"])

    # 1) GPU check
    if (not args.no_gpu_check) and cfg.get("system", {}).get("require_gpu", True):
        ok = check_gpu(cfg)
        if not ok:
            log("GPU check failed. If you want to continue without GPU check, re-run with --no-gpu-check")
            sys.exit(3)

    # 2) trtexec check (optional)
    if cfg.get("actions", {}).get("verify_trtexec", True):
        ok_trt = check_trtexec(cfg)
        if not ok_trt:
            log("Warning: trtexec not found. You will NOT be able to convert ONNX -> TensorRT until TensorRT is installed.")
            # do not exit; user may still want to download models

    if cfg.get("actions", {}).get("run_clone_repo", True):
        repo_cfg = cfg.get("git", {})
        repo_url = repo_cfg.get("repo")
        branch = repo_cfg.get("branch", "main")
        dest = cfg["paths"]["repo_dir"]

        # Load PAT from environment
        GITHUB_PAT = os.getenv("GITHUB_PAT")
        if GITHUB_PAT and repo_url.startswith("https://github.com/"):
            # Insert token into HTTPS URL
            repo_url = repo_url.replace("https://", f"https://{GITHUB_PAT}@")
            
        if repo_url:
            git_clone_or_pull(repo_url, branch, dest)
        else:
            log("No git repo URL provided in config; skipping clone/pull.")


    # 4) Download models if missing
    if cfg.get("actions", {}).get("run_download_models", True):
        ensure_models(cfg)

    # 5) Convert ONNX -> TensorRT (if requested and not skipped)
    if (not args.skip_convert) and cfg.get("actions", {}).get("run_convert_trt", True):
        # If trtexec missing, conversion will fail; we attempt anyway and bubble errors
        convert_models_to_trt(cfg)
    else:
        log("Skipping conversion step (skip flag or configured false).")

    # 6) Final summary
    print_summary(cfg)
    log("Setup finished. If everything exists, the instance is ready for inference tests.")

if __name__ == "__main__":
    main()
