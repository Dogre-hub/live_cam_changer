#!/usr/bin/env python3
"""
setup/setup.py

Enhanced idempotent installer for live_cam_changer repo.

- Reads .env and setup/config.yaml
- Installs missing python packages (requests, tqdm, pyyaml, python-dotenv) automatically
- Checks GPU (nvidia-smi) and optional CUDA version
- Installs/unpacks TensorRT from a local tarball (if configured) and wires environment
- Moves local .onnx/.trt into models/onnx and models/trt (idempotent)
- Downloads ONNX / PyTorch models (if urls provided)
- Converts ONNX -> TensorRT using trtexec (streams logs)
- Logs everything to setup/logs/

Usage:
    python3 setup/setup.py [--config setup/config.yaml] [--no-gpu-check] [--skip-convert]

Make sure to add a .env file for private repo PAT (GITHUB_PAT) if cloning a private repo.
"""

import os
import sys
import time
import tarfile
import stat
import shutil
import argparse
import subprocess
from pathlib import Path

# --- try optional imports; we will auto-install if missing ---
MISSING = []
try:
    import requests
except Exception:
    MISSING.append("requests")
try:
    from tqdm import tqdm
except Exception:
    MISSING.append("tqdm")
try:
    import yaml
except Exception:
    MISSING.append("pyyaml")
try:
    from dotenv import load_dotenv
except Exception:
    MISSING.append("python-dotenv")

LOG_DIR = Path("setup/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[setup {ts}] {msg}")

def run_cmd(cmd, timeout=None, check=False, env=None, cwd=None):
    """Run command and capture output."""
    if isinstance(cmd, (list, tuple)):
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, env=env, cwd=cwd)
    else:
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, env=env, cwd=cwd)
    if proc.returncode != 0 and check:
        raise RuntimeError(f"Command failed: {cmd}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc

def run_cmd_stream(cmd, cwd=None, env=None, timeout=None):
    """Run command and stream stdout/stderr to console and a timestamped log file."""
    if isinstance(cmd, (list, tuple)):
        cmd_list = cmd
    else:
        cmd_list = ["bash", "-lc", cmd]
    logfile = LOG_DIR / f"run-{int(time.time())}.log"
    log(f"Running: {' '.join(cmd_list) if isinstance(cmd_list, list) else cmd_list}")
    with open(logfile, "w") as lf:
        proc = subprocess.Popen(cmd_list, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
        lines = []
        try:
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                lines.append(line)
                print(line.rstrip())
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            log(f"Command timeout. See log: {logfile}")
            raise
    return proc.returncode, "".join(lines)

def auto_install_python_pkgs():
    """If optional packages are missing, attempt pip install them."""
    if not MISSING:
        return
    log(f"Missing python packages detected: {MISSING}. Attempting to pip install...")
    pkg_str = " ".join(MISSING)
    proc = run_cmd(f"pip3 install {pkg_str}")
    if proc.returncode != 0:
        log(f"pip install failed: {proc.stderr}")
        raise RuntimeError("Please install required Python packages and re-run.")

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def download_file(url, out_path, chunk_size=1 << 20):
    out_path = Path(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    log(f"Downloading: {url} -> {out_path}")
    sess = requests.Session()
    with sess.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = resp.headers.get("Content-Length")
        if total:
            total = int(total)
        with open(tmp, "wb") as f:
            if total:
                pbar = tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name)
            else:
                pbar = None
            downloaded = 0
            start = time.time()
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if pbar:
                        pbar.update(len(chunk))
            if pbar:
                pbar.close()
    tmp.replace(out_path)
    log(f"Downloaded {out_path} ({downloaded} bytes) in {time.time() - start:.1f}s")

def file_move_or_copy(src, dst):
    """Move src->dst if possible, else copy. Idempotent (skips if dst exists)."""
    src = Path(src)
    dst = Path(dst)
    ensure_dir(dst.parent)
    if not src.exists():
        return False, f"src-missing:{src}"
    if dst.exists():
        return True, f"exists:{dst}"
    try:
        src.rename(dst)
        return True, f"moved:{dst}"
    except Exception:
        shutil.copy2(src, dst)
        return True, f"copied:{dst}"

def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

# --- system checks ---
def check_gpu(require_gpu=True):
    if not require_gpu:
        log("GPU not required by config.")
        return True
    try:
        proc = run_cmd(["nvidia-smi"])
        log("nvidia-smi OK.")
        return True
    except Exception as e:
        log(f"nvidia-smi failed: {e}")
        return False

def check_nvcc(cuda_version=None):
    if not cuda_version:
        return True
    try:
        proc = run_cmd(["nvcc", "--version"])
        out = proc.stdout + proc.stderr
        if cuda_version in out:
            log(f"nvcc reports expected CUDA version {cuda_version}.")
            return True
        else:
            log(f"nvcc output did not show expected version {cuda_version}. nvcc output sample:\n{out.splitlines()[:6]}")
            return False
    except Exception:
        log("nvcc not found or error running nvcc --version.")
        return False

def check_trtexec(trtexec_cmd="trtexec"):
    p = shutil.which(trtexec_cmd)
    if p:
        log(f"Found trtexec: {p}")
        return True
    log("trtexec not found in PATH.")
    return False

# --- TensorRT tarball extraction & wiring ---
def wire_tensorrt_env(install_dir: Path):
    binpath = str(install_dir / "bin")
    libpath = str(install_dir / "lib")
    profile = "/etc/profile.d/tensorrt.sh"
    ldconf = "/etc/ld.so.conf.d/tensorrt.conf"
    content = (
        f'export TENSORRT_ROOT="{install_dir}"\n'
        f'export LD_LIBRARY_PATH="{libpath}:${{LD_LIBRARY_PATH:-}}"\n'
        f'export PATH="{binpath}:${{PATH:-}}"\n'
    )
    try:
        log(f"Writing {profile}")
        with open(profile, "w") as f:
            f.write(content)
        os.chmod(profile, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        log(f"Writing {ldconf}")
        with open(ldconf, "w") as f:
            f.write(libpath + "\n")
        log("Running ldconfig...")
        run_cmd(["ldconfig"])
        log("TensorRT environment wired (profile + ld).")
        return True
    except PermissionError:
        log(f"Permission error while writing {profile} or {ldconf}. Run the script as root or create these files manually with:\n{content}")
        return False
    except Exception as e:
        log(f"Failed to wire env: {e}")
        return False

def extract_tensorrt_tarball(tarball: Path, install_dir: Path):
    if not tarball.exists():
        log(f"TensorRT tarball not found: {tarball}")
        return False
    if (install_dir / "bin" / "trtexec").exists():
        log(f"TensorRT already extracted at {install_dir}")
        return True
    tmp = install_dir.parent / (install_dir.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    ensure_dir(tmp)
    log(f"Extracting TensorRT tarball {tarball} -> {tmp}")
    try:
        with tarfile.open(tarball, "r:gz") as tar:
            members = tar.getmembers()
            pbar = tqdm(total=len(members), desc="extracting TensorRT")
            for m in members:
                tar.extract(m, path=tmp)
                pbar.update(1)
            pbar.close()
        # move folder into install_dir
        entries = [p for p in tmp.iterdir() if p.exists()]
        if len(entries) == 1 and entries[0].is_dir():
            src = entries[0]
        else:
            src = tmp
        if install_dir.exists():
            shutil.rmtree(install_dir)
        shutil.move(str(src), str(install_dir))
        if tmp.exists():
            try:
                shutil.rmtree(tmp)
            except Exception:
                pass
        log(f"TensorRT extracted to {install_dir}")
        return True
    except Exception as e:
        log(f"Failed to extract tarball: {e}")
        return False

# --- models handling and conversion ---
def relocate_local_models(cfg):
    """
    Find any .onnx/.trt in repo root, /workspace, or PWD and move them into onnx_dir/trt_dir.
    """
    onnx_dir = Path(cfg["paths"]["onnx_dir"])
    trt_dir = Path(cfg["paths"]["trt_dir"])
    ensure_dir(onnx_dir); ensure_dir(trt_dir)
    search_roots = [Path.cwd(), Path("/workspace"), Path(os.getenv("PWD", "."))]
    moved = []
    for root in search_roots:
        if not root.exists():
            continue
        for f in list(root.glob("*.onnx")) + list(root.glob("*.trt")):
            dst = onnx_dir / f.name if f.suffix == ".onnx" else trt_dir / f.name
            if dst.exists():
                continue
            ok,info = file_move_or_copy(f, dst)
            if ok:
                moved.append((f, dst, info))
                log(f"Relocated local model: {f} -> {dst} ({info})")
    return moved

def ensure_models(cfg):
    paths = cfg["paths"]
    onnx_dir = Path(paths["onnx_dir"])
    trt_dir = Path(paths["trt_dir"])
    models_dir = Path(paths["models_dir"])
    ensure_dir(onnx_dir); ensure_dir(trt_dir); ensure_dir(models_dir)

    # relocate local files first
    relocate_local_models(cfg)

    for m in cfg.get("models", []):
        name = m.get("name")
        typ = m.get("type")
        url = (m.get("url") or "").strip()
        filename = m.get("filename")
        if not filename:
            log(f"Model {name} missing 'filename' in config; skipping.")
            continue
        if typ == "onnx":
            out = onnx_dir / filename
            if out.exists():
                log(f"ONNX exists: {out}")
                continue
            if not url:
                log(f"No URL configured for ONNX model {name} and file missing; skipping.")
                continue
            download_file(url, out)
        elif typ in ("pth","pt"):
            out = models_dir / filename
            if out.exists():
                log(f"PyTorch model exists: {out}")
                continue
            if not url:
                log(f"No URL configured for PyTorch model {name} and file missing; skipping.")
                continue
            download_file(url, out)
        else:
            log(f"Unknown model type {typ} for {name}; skipping.")

def normalize_trtexec_args(args):
    """Map older --workspace/--workspaceSize into memPoolSize if necessary."""
    if not args:
        return ""
    if "--memPoolSize" in args:
        return args
    import re
    m = re.search(r"--workspace(?:Size)?(?:=|\s+)(\d+)", args)
    if m:
        val = int(m.group(1))
        if val >= 1024 and val % 1024 == 0:
            # user gave MB maybe; use M
            args = re.sub(r"--workspace(?:Size)?(?:=|\s+)\d+", "", args)
            return f"{args} --memPoolSize=workspace:{val}M"
        else:
            args = re.sub(r"--workspace(?:Size)?(?:=|\s+)\d+", "", args)
            return f"{args} --memPoolSize=workspace:{val}M"
    return args

def convert_models(cfg):
    onnx_dir = Path(cfg["paths"]["onnx_dir"])
    trt_dir = Path(cfg["paths"]["trt_dir"])
    trtexec = cfg.get("system", {}).get("trtexec_cmd", "trtexec")
    timeout = cfg.get("system", {}).get("trtexec_timeout_sec", 900)
    for m in cfg.get("models", []):
        if not m.get("convert_to_trt", False):
            continue
        if m.get("type") != "onnx":
            log(f"{m.get('name')} marked for convert but not ONNX; skipping.")
            continue
        onnx_file = onnx_dir / m["filename"]
        trt_file = trt_dir / m["trt_filename"]
        if not onnx_file.exists():
            log(f"Missing ONNX for {m.get('name')}: {onnx_file}")
            continue
        if trt_file.exists():
            log(f"TRT already exists: {trt_file} (skipping conversion)")
            continue
        args = m.get("trtexec_args", "")
        args = normalize_trtexec_args(args)
        cmd = f"{trtexec} --onnx={onnx_file} --saveEngine={trt_file} {args}"
        log(f"Converting {m.get('name')} -> TRT. Command:\n  {cmd}")
        try:
            rc, out = run_cmd_stream(cmd, timeout=timeout)
            if rc == 0:
                log(f"Conversion succeeded for {m.get('name')}: {trt_file}")
            else:
                log(f"Conversion failed (rc={rc}). See setup/logs for details.")
                if trt_file.exists():
                    try:
                        trt_file.unlink()
                        log(f"Removed partial engine {trt_file}")
                    except Exception:
                        pass
                raise RuntimeError(f"trtexec failed for {m.get('name')}")
        except Exception as e:
            log(f"Conversion raised exception: {e}")
            raise

def print_summary(cfg):
    paths = cfg["paths"]
    log("SETUP SUMMARY")
    log(f"repo_dir: {paths['repo_dir']}")
    log(f"models_dir: {paths['models_dir']}")
    log(f"onnx_dir: {paths['onnx_dir']}")
    log(f"trt_dir: {paths['trt_dir']}")
    for m in cfg.get("models", []):
        nm = m.get("name")
        typ = m.get("type")
        if typ == "onnx":
            onnxp = Path(paths["onnx_dir"]) / m.get("filename")
            trtp = Path(paths["trt_dir"]) / m.get("trt_filename") if m.get("trt_filename") else None
            log(f"{nm}: onnx={'OK' if onnxp.exists() else 'MISSING'} trt={'OK' if trtp and trtp.exists() else 'MISSING' if m.get('convert_to_trt') else 'N/A'}")
        else:
            p = Path(paths["models_dir"]) / m.get("filename")
            log(f"{nm}: file={'OK' if p.exists() else 'MISSING'}")

# --- main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="setup/config.yaml")
    parser.add_argument("--no-gpu-check", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    args = parser.parse_args()

    # auto-install missing python packages
    auto_install_python_pkgs()

    # now safe to import dotenv/requests/tqdm/yaml
    from dotenv import load_dotenv
    load_dotenv()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        log(f"Config not found: {cfg_path}")
        sys.exit(2)
    cfg = load_config(cfg_path)

    # ensure paths
    for k in ("repo_dir","models_dir","onnx_dir","trt_dir"):
        ensure_dir(cfg["paths"][k])

    # pre-checks
    if not args.no_gpu_check and cfg.get("system", {}).get("require_gpu", True):
        ok = check_gpu(True)
        if not ok:
            log("GPU check failed. Re-run with --no-gpu-check if you want to continue without GPU.")
            sys.exit(3)
    # optional nvcc check
    check_nvcc(cfg.get("system", {}).get("cuda_version"))

    # ensure trtexec or install tensorrt from tarball
    if not check_trtexec(cfg.get("system", {}).get("trtexec_cmd","trtexec")):
        tarball = cfg.get("system", {}).get("tensorrt_tarball")
        install_dir = Path(cfg.get("system", {}).get("tensorrt_install_dir", "/workspace/tensorrt"))
        if tarball:
            tarball = Path(tarball)
            ok = extract_tensorrt_tarball(tarball, install_dir)
            if ok:
                # wire env (may require root)
                wired = wire_tensorrt_env(install_dir)
                if not wired:
                    log("Failed to write profile/ld files (permission denied?). You can manually source the following env in your session:")
                    log(f'export TENSORRT_ROOT="{install_dir}"')
                    log(f'export LD_LIBRARY_PATH="{install_dir}/lib:${{LD_LIBRARY_PATH:-}}"')
                    log(f'export PATH="{install_dir}/bin:${{PATH:-}}"')
            else:
                log("TensorRT extraction failed or not configured. Conversion step will be skipped unless trtexec is available.")
        else:
            log("No TensorRT tarball configured and trtexec not found. You must install TensorRT manually to convert models.")

    # clone/pull repo if configured (handles private repo via GITHUB_PAT in .env)
    if cfg.get("actions", {}).get("run_clone_repo", True):
        repo_cfg = cfg.get("git", {})
        repo_url = repo_cfg.get("repo")
        branch = repo_cfg.get("branch","main")
        dest = Path(cfg["paths"]["repo_dir"])
        if repo_url:
            GITHUB_PAT = os.getenv("GITHUB_PAT")
            if GITHUB_PAT and repo_url.startswith("https://github.com/"):
                repo_url = repo_url.replace("https://", f"https://{GITHUB_PAT}@")
            if dest.exists() and (dest / ".git").exists():
                try:
                    run_cmd(["git","-C",str(dest),"fetch"], check=True)
                    run_cmd(["git","-C",str(dest),"checkout",branch], check=True)
                    run_cmd(["git","-C",str(dest),"pull","origin",branch], check=True)
                    log(f"Pulled repo {repo_url} -> {dest}")
                except Exception as e:
                    log(f"Git pull failed: {e}")
            else:
                try:
                    ensure_dir(dest.parent)
                    run_cmd(["git","clone","--branch",branch,repo_url,str(dest)], check=True)
                    log(f"Cloned repo {repo_url} -> {dest}")
                except Exception as e:
                    log(f"Git clone failed: {e}")
        else:
            log("No git repo URL configured; skipping clone/pull.")

    # models: relocate local files and download missing ones
    ensure_models(cfg)

    # convert models if trtexec exists and conversion enabled
    if not args.skip_convert and cfg.get("actions", {}).get("run_convert_trt", True):
        if not check_trtexec(cfg.get("system", {}).get("trtexec_cmd","trtexec")):
            log("trtexec not available; skipping conversion step.")
        else:
            convert_models(cfg)
    else:
        log("Skipping conversion step (skipped by flag or config).")

    # final summary
    print_summary(cfg)
    log("Setup completed. Check setup/logs for command outputs and errors.")

if __name__ == "__main__":
    main()
