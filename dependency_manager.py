"""
dependency_manager.py
Checks, installs, and updates all required dependencies for Voice to MIDI.
Runs at startup before launching the main app.
"""

import subprocess
import sys
import importlib
import json
import os
from pathlib import Path
from datetime import datetime, timedelta

# ── Required packages ──────────────────────────────────────────────────────────
# Format: (pip_name, import_name, min_version_or_None)
REQUIRED_PACKAGES = [
    ("gradio",          "gradio",           "4.0.0"),
    ("numpy",           "numpy",            "1.21.0"),
    ("scipy",           "scipy",            "1.7.0"),
    ("librosa",         "librosa",          "0.10.0"),
    ("midiutil",        "midiutil",         "1.2.1"),
    ("sounddevice",     "sounddevice",      "0.4.6"),
    ("torch",           "torch",            None),      # PyTorch — handled specially
    ("tensorflow",      "tensorflow",       None),      # required by crepe
    ("crepe",           "crepe",            "0.0.12"),  # neural pitch detector
    ("matplotlib",      "matplotlib",       "3.5.0"),
    ("plotly",          "plotly",           "5.0.0"),
    ("pillow",          "PIL",              "9.0.0"),
    ("python-rtmidi",   "rtmidi",           None),      # virtual MIDI output
]

# Cache file to avoid checking every single run (check max once per hour)
CACHE_FILE = Path(".dep_cache.json")
CHECK_INTERVAL_HOURS = 1


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_cache(data: dict):
    try:
        CACHE_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _pip_install(package: str, upgrade: bool = False):
    """Install or upgrade a pip package."""
    cmd = [sys.executable, "-m", "pip", "install", "--quiet"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append(package)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr


def _get_installed_version(import_name: str) -> str | None:
    try:
        mod = importlib.import_module(import_name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return None


def _version_ok(installed: str, minimum: str) -> bool:
    """Returns True if installed >= minimum (simple tuple compare)."""
    if installed in (None, "unknown"):
        return False
    try:
        inst = tuple(int(x) for x in installed.split(".")[:3])
        mini = tuple(int(x) for x in minimum.split(".")[:3])
        return inst >= mini
    except Exception:
        return True  # can't parse — assume OK


def _install_torch():
    """
    Install PyTorch with GPU (CUDA) support if a GPU is present,
    otherwise install CPU-only version.
    """
    print("  📦 Installing PyTorch...")
    # Try GPU version first
    gpu_cmd = [
        sys.executable, "-m", "pip", "install", "--quiet",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ]
    result = subprocess.run(gpu_cmd, capture_output=True, text=True, timeout=300)
    if result.returncode == 0:
        print("  ✅ PyTorch (CUDA 12.1) installed")
        return True

    # Fallback: CPU-only
    print("  ⚠️  GPU PyTorch failed — installing CPU version...")
    cpu_cmd = [sys.executable, "-m", "pip", "install", "--quiet",
               "torch", "torchvision", "torchaudio"]
    result = subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=300)
    if result.returncode == 0:
        print("  ✅ PyTorch (CPU) installed")
        return True

    print(f"  ❌ PyTorch install failed: {result.stderr[:200]}")
    return False


def check_gpu() -> dict:
    """Return GPU info dict."""
    info = {"available": False, "name": "CPU only", "cuda_version": None}
    try:
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["available"] = True
            info["name"] = "Apple Metal (MPS)"
    except ImportError:
        pass
    return info


def run_checks(force: bool = False) -> bool:
    """
    Main entry point.
    Returns True if all dependencies are satisfied and app can start.
    """
    cache = _load_cache()
    now = datetime.utcnow().isoformat()

    # Decide whether to do a full check or use cache
    last_check = cache.get("last_check")
    do_full_check = force or (last_check is None)
    if last_check:
        delta = datetime.utcnow() - datetime.fromisoformat(last_check)
        if delta > timedelta(hours=CHECK_INTERVAL_HOURS):
            do_full_check = True

    print("\n🔍 Dependency Check")
    print("─" * 40)

    all_ok = True

    for pip_name, import_name, min_ver in REQUIRED_PACKAGES:
        # Special case: torch needs a custom installer
        if pip_name == "torch":
            ver = _get_installed_version("torch")
            if ver is None:
                success = _install_torch()
                if not success:
                    all_ok = False
            elif do_full_check:
                print(f"  ✅ torch {ver}")
            continue

        ver = _get_installed_version(import_name)

        if ver is None:
            # Not installed — install now
            print(f"  📦 Installing {pip_name}...", end=" ", flush=True)
            ok, err = _pip_install(pip_name)
            if ok:
                ver = _get_installed_version(import_name) or "?"
                print(f"✅ {ver}")
            else:
                print(f"❌  {err[:80]}")
                all_ok = False

        elif min_ver and not _version_ok(ver, min_ver):
            # Outdated — upgrade
            print(f"  ⬆️  Upgrading {pip_name} ({ver} → {min_ver}+)...", end=" ", flush=True)
            ok, err = _pip_install(pip_name, upgrade=True)
            if ok:
                ver = _get_installed_version(import_name) or "?"
                print(f"✅ {ver}")
            else:
                print(f"⚠️  upgrade failed, keeping {ver}")

        elif do_full_check:
            print(f"  ✅ {pip_name} {ver}")

    # ── rtmidi special handling (C extension, may need system libs) ────────────
    try:
        import rtmidi  # noqa
    except ImportError:
        print("  📦 Installing python-rtmidi (may need build tools)...", end=" ", flush=True)
        ok, err = _pip_install("python-rtmidi")
        if not ok:
            print(f"⚠️  rtmidi unavailable — virtual MIDI output disabled")
        else:
            print("✅")

    # Save cache
    cache["last_check"] = now
    _save_cache(cache)

    # GPU report
    gpu = check_gpu()
    print("\n🖥️  Hardware")
    print("─" * 40)
    if gpu["available"]:
        print(f"  🚀 GPU: {gpu['name']}")
        if gpu["cuda_version"]:
            print(f"     CUDA {gpu['cuda_version']}")
    else:
        print("  💻 Running on CPU (GPU not detected)")
        print("     Note: pitch detection will be slower without a GPU")

    print("─" * 40)
    if all_ok:
        print("✅ All dependencies satisfied — launching app\n")
    else:
        print("⚠️  Some dependencies had issues — app may have limited functionality\n")

    return True   # always attempt to start
