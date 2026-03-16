"""
app.py
Entry point for Voice to MIDI.
1. Runs dependency checks / auto-install
2. Loads the audio engine (with GPU if available)
3. Launches Gradio and opens the browser automatically
"""

import sys
import os
import logging
import warnings

# ── Silence TensorFlow / oneDNN noise before anything imports TF ──────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",    "3")   # suppress C++ TF logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS",   "0")   # suppress oneDNN info
os.environ.setdefault("ABSL_MIN_LOG_LEVEL",       "3")   # suppress absl logging
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH","true")

# ── Silence noisy third-party loggers before anything else imports them ────────
logging.basicConfig(level=logging.WARNING)
for noisy in ("httpx", "httpcore", "gradio", "urllib3",
              "asyncio", "multipart", "uvicorn.access",
              "uvicorn.error", "fastapi"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning)     # suppress Gradio 6 param warnings
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"            # disable Gradio telemetry pings

HOST = "127.0.0.1"
PORT = 7860
URL  = f"http://{HOST}:{PORT}"


def _banner():
    W = 52
    print()
    print("╔" + "═" * W + "╗")
    print("║" + "  🎙️   V O I C E   T O   M I D I  —  v1.3.0  ".center(W) + "║")
    print("╠" + "═" * W + "╣")


def _step(icon: str, label: str):
    print(f"║  {icon}  {label:<46}║")


def _divider():
    W = 52
    print("╠" + "═" * W + "╣")


def _footer(url: str):
    W = 52
    print("╠" + "═" * W + "╣")
    print("║" + f"  🌐  {url}".ljust(W) + "║")
    print("║" + "  Press Ctrl+C to stop".ljust(W) + "║")
    print("╚" + "═" * W + "╝")
    print()


def main():
    _banner()
    _step("🔍", "Checking dependencies...")

    from dependency_manager import run_checks, check_gpu
    run_checks()

    gpu = check_gpu()
    gpu_str = gpu["name"] if gpu["available"] else "CPU only"
    _step("🖥️ ", f"Hardware: {gpu_str}")
    _divider()

    _step("🎛️ ", "Loading audio engine...")
    from audio_engine import AudioEngine
    engine = AudioEngine()

    _step("🧠", "Loading pitch model (first run ~30s)...")
    engine.load_model()
    _divider()

    _step("🌐", "Starting interface...")
    from ui import create_ui
    demo, css, theme = create_ui(engine)

    _footer(URL)

    demo.launch(
        server_name = HOST,
        server_port = PORT,
        share       = False,
        inbrowser   = True,       # auto-open browser tab
        show_error  = True,
        quiet       = True,       # suppress Gradio's own startup banner
        css         = css,
        theme       = theme,
    )


if __name__ == "__main__":
    main()
