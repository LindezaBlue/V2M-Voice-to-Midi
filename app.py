"""
app.py
Entry point for Voice to MIDI.
1. Runs dependency checks / auto-install
2. Loads the audio engine (with GPU if available)
3. Launches the Gradio UI
"""

import sys
import logging
import os

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "═" * 50)
    print("  🎙️  Voice to MIDI  —  v1.0.0")
    print("═" * 50)

    # ── Step 1: Dependency check ───────────────────────────────────────────────
    from dependency_manager import run_checks
    run_checks()

    # ── Step 2: Load audio engine ──────────────────────────────────────────────
    print("🎛️  Loading audio engine...")
    from audio_engine import AudioEngine
    engine = AudioEngine()

    print("🧠 Loading pitch detection model (first run may take ~30s)...")
    engine.load_model()

    # ── Step 3: Launch Gradio ──────────────────────────────────────────────────
    print("🌐 Starting Gradio interface...\n")
    from ui import create_ui
    demo = create_ui(engine)

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,         # set True to get a public gradio.live URL
        show_error=True,
        quiet=False,
    )


if __name__ == "__main__":
    main()
