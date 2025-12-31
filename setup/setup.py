#!/usr/bin/env python3
"""
setup/setup.py (improved)

Behavior additions:
- If trtexec missing, try:
  1) extract TensorRT tarball (if configured) and wire env
  2) search common locations for trtexec; if found but not on PATH, add permanent PATH
- Idempotent wiring: writes to /etc/profile.d if root, else appends to ~/.bashrc
- All other behavior as before: download models, relocate local files, convert ONNX->TRT, logging.

Usage:
    python3 setup/setup.py [--config setup/config.yaml] [--no-gpu-check] [--skip-convert]
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

# --- optional deps auto-install handled below ---
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

# ------------------ subprocess helpers ------------------
def run_cmd(cmd, timeout=None, check=False, env=None, cwd=None):
    if isinstance(cmd, (list, tuple)):
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, env=env, cwd=cwd)
    else:
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, env=env, cwd=cwd)
    if proc.returncode != 0 and check:
        raise RuntimeError(f"Command failed: {cmd}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc

def run_cmd_stream(cmd, cwd=None, env=None, timeout=None):
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

# ------------------ utilities ------------------
def auto_install_python_pkgs():
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
        import yaml as _yaml
        return _yaml.safe_load(f)

# ------------------ system checks ------------------
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

# ------------------ TensorRT install/wiring ------------------
def wire_tensorrt_env_root(install_dir: Path):
    """Write /etc/profile.d and /etc/ld.so.conf.d (requires root)."""
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
        log("Wired TensorRT env in /etc (root).")
        return True
    except Exception as e:
        log(f"Failed to wire root env: {e}")
        return False

def wire_tensorrt_env_user(install_dir: Path):
    """Append exports to ~/.bashrc if not present (idempotent)."""
    binpath = str(install_dir / "bin")
    libpath = str(install_dir / "lib")
    bashrc = Path.home() / ".bashrc"
    export_lines = [
        f'export TENSORRT_ROOT="{install_dir}"',
        f'export LD_LIBRARY_PATH="{libpath}:${{LD_LIBRARY_PATH:-}}"',
        f'export PATH="{binpath}:${{PATH:-}}"',
    ]
    try:
        content = ""
        if bashrc.exists():
            content = bashrc.read_text()
        else:
            bashrc.parent.mkdir(parents=True, exist_ok=True)
            bashrc.write_text("")
            content = ""
        changed = False
        for line in export_lines:
            if line not in content:
                with open(bashrc, "a") as f:
                    f.write("\n# Added by setup/setup.py\n")
                    f.write(line + "\n")
                changed = True
        if changed:
            log(f"Appended TensorRT env to {bashrc}. Source it or open a new shell to pick it up.")
        else:
            log(f"TensorRT env already present in {bashrc}.")
        return True
    except Exception as e:
        log(f"Failed to write user bashrc: {e}")
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
        entries = [p for p in tmp.iterdir() if p.exists()]
        src = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmp
        if install_dir.exists():
            shutil.rmtree(install_dir)
        shutil.move(str(src), str(install_dir))
        try:
            shutil.rmtree(tmp)
        except Exception:
            pass
        log(f"TensorRT extracted to {install_dir}")
        return True
    except Exception as e:
        log(f"Failed to extract tarball: {e}")
        return False

def find_trtexec_in_paths(search_roots=None):
    """Search for any 'trtexec' binary under common locations; return path if found."""
    if search_roots is None:
        search_roots = [Path("/workspace"), Path.cwd(), Path("/usr"), Path("/opt"), Path("/usr/local")]
    for root in search_roots:
        if not root.exists():
            continue
        # use rglob but limit depth to avoid long searches
        try:
            for p in root.rglob("trtexec"):
                if p.is_file() and os.access(p, os.X_OK):
                    return p.resolve()
        except Exception:
            continue
    return None

def add_trtexec_to_path(install_dir: Path):
    """Try to add install_dir/bin to PATH permanently (root or user)."""
    if os.geteuid() == 0:
        ok = wire_tensorrt_env_root(install_dir)
        if ok:
            return True
    # fallback to user .bashrc
    ok = wire_tensorrt_env_user(install_dir)
    return ok

# ------------------ model handling & conversion (unchanged logic) ------------------
def relocate_local_models(cfg):
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
    if not args:
        return ""
    if "--memPoolSize" in args:
        return args
    import re
    m = re.search(r"--workspace(?:Size)?(?:=|\s+)(\d+)", args)
    if m:
        val = int(m.group(1))
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

# ------------------ main ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="setup/config.yaml")
    parser.add_argument("--no-gpu-check", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    args = parser.parse_args()

    # auto-install missing python packages
    auto_install_python_pkgs()

    # safe to import dotenv now
    from dotenv import load_dotenv
    load_dotenv()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        log(f"Config not found: {cfg_path}")
        sys.exit(2)
    cfg = load_config(cfg_path)

    # ensure base paths
    for k in ("repo_dir","models_dir","onnx_dir","trt_dir"):
        ensure_dir(cfg["paths"][k])

    # pre-checks
    if not args.no_gpu_check and cfg.get("system", {}).get("require_gpu", True):
        ok = check_gpu(True)
        if not ok:
            log("GPU check failed. Re-run with --no-gpu-check to bypass.")
            sys.exit(3)
    check_nvcc(cfg.get("system", {}).get("cuda_version"))

    # trtexec availability: if missing try extraction, else search and wire
    trtexec_cmd = cfg.get("system", {}).get("trtexec_cmd", "trtexec")
    if not check_trtexec(trtexec_cmd):
        # 1) if tarball configured try extract
        tarball = cfg.get("system", {}).get("tensorrt_tarball")
        install_dir = Path(cfg.get("system", {}).get("tensorrt_install_dir", "/workspace/tensorrt"))
        extracted = False
        if tarball:
            tarball = Path(tarball)
            if tarball.exists():
                ok = extract_tensorrt_tarball(tarball, install_dir)
                if ok:
                    # try to wire env (prefer root wiring if root)
                    if os.geteuid() == 0:
                        wired = wire_tensorrt_env_root(install_dir)
                        if not wired:
                            log("root wiring failed; will fallback to user bashrc wiring.")
                            wire_tensorrt_env_user(install_dir)
                    else:
                        wire_tensorrt_env_user(install_dir)
                    extracted = True
                    # update PATH for current process so next check_trtexec sees it
                    os.environ["PATH"] = str(install_dir / "bin") + ":" + os.environ.get("PATH","")
                    os.environ["LD_LIBRARY_PATH"] = str(install_dir / "lib") + ":" + os.environ.get("LD_LIBRARY_PATH","")
            else:
                log(f"Configured TensorRT tarball not found: {tarball}")

        # 2) if not extracted, search common locations for trtexec
        if not extracted:
            found = find_trtexec_in_paths()
            if found:
                log(f"Found existing trtexec at: {found}")
                binpath = Path(found).parent
                install_guess = binpath.parent
                # If binary not in PATH, wire it (root or user)
                if str(binpath) not in os.environ.get("PATH",""):
                    ok = add_trtexec_to_path(install_guess)
                    if ok:
                        # also update current env
                        os.environ["PATH"] = str(binpath) + ":" + os.environ.get("PATH","")
                        libp = str(install_guess / "lib")
                        os.environ["LD_LIBRARY_PATH"] = libp + ":" + os.environ.get("LD_LIBRARY_PATH","")
                        log("Wired found TensorRT install into PATH/LD_LIBRARY_PATH.")
                    else:
                        log("Failed to wire found trtexec into PATH automatically. Please add it manually.")
                else:
                    log("Found trtexec already on PATH.")
            else:
                log("No trtexec found in system locations and no tarball configured. Conversions will be skipped unless trtexec is later installed.")

    else:
        log("trtexec already available in PATH.")

    # clone/pull repo if configured
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
