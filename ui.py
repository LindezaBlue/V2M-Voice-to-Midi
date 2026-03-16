"""
ui.py
Gradio-based GUI for Voice to MIDI.
Dark studio aesthetic with real-time piano roll and note display.
"""

import gradio as gr
import numpy as np
import threading
import time
import json
from typing import Optional
import logging

from audio_engine import AudioEngine, NoteEvent, MODES, NOTE_NAMES, midi_to_name
from dependency_manager import check_gpu

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark studio console aesthetic
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
  --bg-dark:    #0c0e14;
  --bg-panel:   #13161f;
  --bg-card:    #1a1e2a;
  --accent:     #00ffa3;
  --accent2:    #ff4eb8;
  --accent3:    #4e9fff;
  --text:       #e8eaf0;
  --text-dim:   #6b7280;
  --border:     #252a38;
  --glow:       0 0 20px rgba(0,255,163,0.25);
}

body, .gradio-container {
  background: var(--bg-dark) !important;
  font-family: 'Space Mono', monospace !important;
  color: var(--text) !important;
}

h1 { 
  font-family: 'Syne', sans-serif !important;
  font-weight: 800 !important;
  font-size: 2.2rem !important;
  letter-spacing: -0.02em !important;
  background: linear-gradient(90deg, var(--accent), var(--accent3));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

h3 {
  font-family: 'Syne', sans-serif !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  color: var(--text-dim) !important;
}

.gr-button {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.8rem !important;
  border-radius: 4px !important;
  transition: all 0.15s ease !important;
}
.gr-button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  box-shadow: var(--glow) !important;
}
.gr-button.primary {
  background: var(--accent) !important;
  color: #000 !important;
  font-weight: 700 !important;
  border: none !important;
}
.gr-button.primary:hover {
  filter: brightness(1.1) !important;
}

.gr-radio label, .gr-dropdown label {
  color: var(--text) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.8rem !important;
}

.gr-block, .gr-box {
  background: var(--bg-panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

#note-display {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px;
  text-align: center;
  min-height: 120px;
}

#status-bar {
  background: var(--bg-card);
  border-top: 1px solid var(--border);
  padding: 8px 16px;
  font-size: 0.7rem;
  color: var(--text-dim);
  display: flex;
  gap: 24px;
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Piano roll HTML renderer
# ─────────────────────────────────────────────────────────────────────────────
PIANO_KEYS = 49   # C2 to C6
PIANO_LOW  = 36   # MIDI C2

def build_piano_html(active_note: Optional[int] = None, history: list = None) -> str:
    """Render an SVG/HTML mini piano with the active note lit up."""
    history = history or []
    recent_notes = {e.note for e in history[-8:] if e.active}  # last 8 notes

    # Build 2-octave piano (C3–B4, 24 keys shown)
    keys_html = []
    white_keys = []
    black_keys = []

    display_low  = 48   # C3
    display_high = 72   # C5 (exclusive)
    num_whites   = 0

    key_info = []
    w_index  = 0
    for midi in range(display_low, display_high):
        semitone = midi % 12
        is_black  = semitone in (1, 3, 6, 8, 10)
        key_info.append((midi, is_black, w_index))
        if not is_black:
            w_index += 1

    total_whites = w_index
    KEY_W = 28
    KEY_H = 100
    BK_W  = 18
    BK_H  = 62
    SVG_W = total_whites * KEY_W
    SVG_H = KEY_H + 40

    # SVG
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_W}" height="{SVG_H}" '
        f'style="display:block;margin:0 auto;overflow:visible">'
    ]

    # White keys first
    wk = 0
    for midi, is_black, wi in key_info:
        if is_black:
            continue
        is_active  = (midi == active_note)
        is_recent  = (midi in recent_notes and not is_active)
        x = wi * KEY_W

        if is_active:
            fill   = "#00ffa3"
            stroke = "#00ffa3"
        elif is_recent:
            fill   = "#1a3d30"
            stroke = "#00ffa3"
        else:
            fill   = "#e8eaf0"
            stroke = "#252a38"

        note_name = NOTE_NAMES[midi % 12] + str((midi // 12) - 1)
        svg_parts.append(
            f'<rect x="{x+1}" y="0" width="{KEY_W-2}" height="{KEY_H}" '
            f'rx="2" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
        )
        if is_active:
            svg_parts.append(
                f'<text x="{x + KEY_W//2}" y="{KEY_H - 8}" '
                f'text-anchor="middle" font-size="9" font-family="Space Mono" fill="#000">'
                f'{note_name}</text>'
            )

    # Black keys on top
    for midi, is_black, wi in key_info:
        if not is_black:
            continue
        is_active = (midi == active_note)
        is_recent = (midi in recent_notes and not is_active)
        x = wi * KEY_W - BK_W // 2

        if is_active:
            fill = "#00ffa3"
        elif is_recent:
            fill = "#1a5040"
        else:
            fill = "#1a1e2a"

        svg_parts.append(
            f'<rect x="{x}" y="0" width="{BK_W}" height="{BK_H}" '
            f'rx="2" fill="{fill}" stroke="#000" stroke-width="1"/>'
        )

    # Active note label above piano
    if active_note is not None:
        label = midi_to_name(active_note)
        svg_parts.append(
            f'<text x="{SVG_W//2}" y="{KEY_H + 26}" text-anchor="middle" '
            f'font-size="18" font-family="Syne, sans-serif" font-weight="800" '
            f'fill="#00ffa3">{label}</text>'
        )
    else:
        svg_parts.append(
            f'<text x="{SVG_W//2}" y="{KEY_H + 26}" text-anchor="middle" '
            f'font-size="12" font-family="Space Mono" fill="#6b7280">— listening —</text>'
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def build_note_history_html(history: list) -> str:
    """Render a scrolling list of recently detected notes."""
    if not history:
        return '<p style="color:#6b7280;font-family:Space Mono;font-size:0.75rem;text-align:center">No notes yet</p>'

    rows = []
    for evt in reversed(history[-16:]):
        age    = time.time() - evt.timestamp
        opacity = max(0.3, 1 - age / 8)
        conf_pct = int(evt.confidence * 100)
        vel_pct  = int(evt.velocity / 127 * 100)
        bar_w    = int(vel_pct * 0.8)
        rows.append(
            f'<div style="display:flex;align-items:center;gap:12px;'
            f'padding:4px 0;opacity:{opacity:.2f};'
            f'font-family:Space Mono,monospace;font-size:0.72rem;border-bottom:1px solid #1e2233;">'
            f'<span style="color:#00ffa3;font-weight:700;width:32px">{evt.name}</span>'
            f'<span style="color:#6b7280;width:40px">#{evt.note}</span>'
            f'<span style="color:#4e9fff;width:60px">{evt.frequency:.1f}Hz</span>'
            f'<div style="flex:1;background:#13161f;height:6px;border-radius:3px">'
            f'<div style="width:{bar_w}%;height:100%;background:linear-gradient(90deg,#00ffa3,#4e9fff);border-radius:3px"></div>'
            f'</div>'
            f'<span style="color:#6b7280;width:36px">{conf_pct}%</span>'
            f'</div>'
        )

    return (
        '<div style="max-height:280px;overflow-y:auto;">'
        + "".join(rows)
        + "</div>"
    )


def build_level_meter(rms_val: float) -> str:
    """Render a simple VU-style level meter."""
    pct = min(100, int(rms_val * 1000))
    if pct > 80:
        color = "#ff4eb8"
    elif pct > 50:
        color = "#ffd700"
    else:
        color = "#00ffa3"

    bars = 20
    filled = int(pct / 100 * bars)
    meter = ""
    for i in range(bars):
        c = color if i < filled else "#1a1e2a"
        meter += f'<div style="flex:1;height:16px;background:{c};border-radius:2px;margin:0 1px"></div>'

    return (
        f'<div style="display:flex;align-items:center;gap:6px;font-family:Space Mono;font-size:0.7rem;">'
        f'<span style="color:#6b7280;width:28px">IN</span>'
        f'<div style="display:flex;flex:1">{meter}</div>'
        f'<span style="color:{color};width:36px;text-align:right">{pct}%</span>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Build Gradio app
# ─────────────────────────────────────────────────────────────────────────────

def create_ui(engine: AudioEngine) -> gr.Blocks:
    gpu_info = check_gpu()
    gpu_label = f"🚀 {gpu_info['name']}" if gpu_info["available"] else "💻 CPU"

    mode_choices = list(MODES.keys())
    mode_descs   = {k: v["description"] for k, v in MODES.items()}

    with gr.Blocks(
        title="Voice to MIDI",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="emerald",
            neutral_hue="slate",
        )
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────────
        with gr.Row():
            gr.Markdown(
                "# 🎙️ Voice to MIDI\n"
                "**Real-time voice pitch detection → MIDI notes**"
            )
            gr.Markdown(
                f"<div style='text-align:right;font-family:Space Mono;"
                f"font-size:0.7rem;color:#6b7280;padding-top:12px'>{gpu_label}</div>"
            )

        gr.HTML("<hr style='border-color:#252a38;margin:0 0 16px 0'>")

        # ── Main layout ───────────────────────────────────────────────────────
        with gr.Row(equal_height=True):

            # ── Left column: controls ──────────────────────────────────────
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### ⚙ Controls")

                with gr.Group():
                    start_btn = gr.Button("▶  Start Listening", variant="primary")
                    stop_btn  = gr.Button("⏹  Stop")

                gr.HTML("<div style='height:12px'></div>")
                gr.Markdown("### 🎛 Instrument Mode")

                mode_radio = gr.Radio(
                    choices=mode_choices,
                    value=mode_choices[0],
                    label="",
                    interactive=True,
                )
                mode_desc_txt = gr.Markdown(
                    f"*{mode_descs[mode_choices[0]]}*",
                    elem_id="mode-desc"
                )

                gr.HTML("<div style='height:12px'></div>")
                gr.Markdown("### 🎚 Settings")

                sensitivity_slider = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.65, step=0.05,
                    label="Detection Sensitivity",
                    info="Higher = detect quieter notes (may add false positives)"
                )
                bpm_slider = gr.Slider(
                    minimum=60, maximum=200, value=120, step=1,
                    label="Export BPM"
                )

                device_choices = [f"{i}: {n}" for i, n in engine.list_input_devices()]
                device_dropdown = gr.Dropdown(
                    choices=device_choices if device_choices else ["Default"],
                    value=device_choices[0] if device_choices else "Default",
                    label="🎤 Input Device",
                )

                gr.HTML("<div style='height:16px'></div>")
                export_btn  = gr.Button("💾  Export to .mid file")
                export_file = gr.File(label="Download MIDI", visible=False)
                export_msg  = gr.Markdown("")

            # ── Right column: visualizers ──────────────────────────────────
            with gr.Column(scale=2):

                # VU meter
                gr.Markdown("### 📊 Input Level")
                level_html = gr.HTML(build_level_meter(0))

                # Status
                with gr.Row():
                    status_badge = gr.HTML(
                        '<span style="background:#1a1e2a;border:1px solid #252a38;'
                        'padding:4px 12px;border-radius:4px;font-family:Space Mono;'
                        'font-size:0.7rem;color:#6b7280">⏸ Idle</span>'
                    )
                    freq_display = gr.HTML(
                        '<span style="font-family:Space Mono;font-size:0.7rem;'
                        'color:#6b7280">— Hz  |  conf: —</span>'
                    )

                # Piano roll
                gr.Markdown("### 🎹 Piano Roll")
                piano_html = gr.HTML(build_piano_html())

                # Note history
                gr.Markdown("### 📋 Note History")
                gr.HTML(
                    '<div style="font-family:Space Mono;font-size:0.65rem;color:#6b7280;'
                    'display:flex;gap:24px;padding:4px 0;border-bottom:1px solid #252a38;'
                    'margin-bottom:4px">'
                    '<span>NOTE</span><span style="margin-left:4px">#</span>'
                    '<span style="margin-left:4px">FREQ</span>'
                    '<span style="flex:1;text-align:center">VELOCITY</span>'
                    '<span>CONF</span></div>'
                )
                history_html = gr.HTML(build_note_history_html([]))

        # ── State ─────────────────────────────────────────────────────────────
        is_running_state = gr.State(False)
        refresh_timer    = gr.Timer(value=0.1, active=False)   # 10Hz UI refresh

        # ── Event handlers ────────────────────────────────────────────────────

        def on_start(sensitivity):
            from audio_engine import CONF_THRESH
            import audio_engine as ae
            ae.CONF_THRESH = float(sensitivity)

            if not engine.is_running:
                engine.start()
            return (
                True,
                '<span style="background:#0d2e1f;border:1px solid #00ffa3;'
                'padding:4px 12px;border-radius:4px;font-family:Space Mono;'
                'font-size:0.7rem;color:#00ffa3">● Recording</span>',
                gr.update(active=True),
            )

        def on_stop():
            engine.stop()
            return (
                False,
                '<span style="background:#1a1e2a;border:1px solid #252a38;'
                'padding:4px 12px;border-radius:4px;font-family:Space Mono;'
                'font-size:0.7rem;color:#6b7280">⏸ Idle</span>',
                gr.update(active=False),
                build_piano_html(),
                build_level_meter(0),
            )

        def on_mode_change(mode):
            engine.set_mode(mode)
            return f"*{mode_descs.get(mode, '')}*"

        def on_sensitivity_change(val):
            import audio_engine as ae
            ae.CONF_THRESH = float(val)

        def on_device_change(dev_str):
            try:
                idx = int(dev_str.split(":")[0])
                engine.set_input_device(idx)
            except Exception:
                pass

        def on_refresh():
            """Called by the timer to update all visualizers."""
            note   = engine.latest_note
            hist   = engine.note_history
            rms_v  = engine.latest_rms
            freq_v = engine.latest_freq
            conf_v = engine.latest_conf

            active_midi = note.note if (note and note.active) else None

            piano  = build_piano_html(active_midi, hist)
            hist_h = build_note_history_html(hist)
            level  = build_level_meter(rms_v)

            if freq_v > 0:
                freq_html = (
                    f'<span style="font-family:Space Mono;font-size:0.75rem;color:#4e9fff">'
                    f'{freq_v:.1f} Hz &nbsp;|&nbsp; '
                    f'<span style="color:#00ffa3">conf: {int(conf_v*100)}%</span></span>'
                )
            else:
                freq_html = '<span style="font-family:Space Mono;font-size:0.7rem;color:#6b7280">— Hz  |  conf: —</span>'

            return piano, hist_h, level, freq_html

        def on_export(bpm):
            if not engine.note_history:
                return gr.update(visible=False), "⚠️ No notes recorded yet."
            try:
                from midi_export import export_to_midi
                path = export_to_midi(engine.note_history, engine._mode, int(bpm))
                return gr.update(value=path, visible=True), f"✅ Exported {len(engine.note_history)} notes"
            except Exception as e:
                return gr.update(visible=False), f"❌ Export failed: {e}"

        # Wire up events
        start_btn.click(
            on_start,
            inputs=[sensitivity_slider],
            outputs=[is_running_state, status_badge, refresh_timer],
        )
        stop_btn.click(
            on_stop,
            inputs=[],
            outputs=[is_running_state, status_badge, refresh_timer, piano_html, level_html],
        )
        mode_radio.change(on_mode_change, inputs=[mode_radio], outputs=[mode_desc_txt])
        sensitivity_slider.change(on_sensitivity_change, inputs=[sensitivity_slider])
        device_dropdown.change(on_device_change, inputs=[device_dropdown])
        refresh_timer.tick(
            on_refresh,
            outputs=[piano_html, history_html, level_html, freq_display],
        )
        export_btn.click(
            on_export,
            inputs=[bpm_slider],
            outputs=[export_file, export_msg],
        )

    return demo
