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
from pitch_correction import SCALES, ROOT_NOTES
from processing import SCALE_INTERVALS, VELOCITY_CURVES, CHORD_TEMPLATES
from session_history import (save_session, list_sessions, load_session,
                              delete_session, export_session_to_midi)

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
    """Render an SVG piano. Only the live active_note is highlighted — no ghost notes."""
    display_low  = 48   # C3
    display_high = 72   # C5 (exclusive)

    key_info = []
    w_index  = 0
    for midi in range(display_low, display_high):
        semitone = midi % 12
        is_black = semitone in (1, 3, 6, 8, 10)
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

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_W}" height="{SVG_H}" '
        f'style="display:block;margin:0 auto;overflow:visible">'
    ]

    # White keys
    for midi, is_black, wi in key_info:
        if is_black:
            continue
        is_active = (midi == active_note)
        x = wi * KEY_W
        fill   = "#00ffa3" if is_active else "#e8eaf0"
        stroke = "#00ffa3" if is_active else "#252a38"
        svg_parts.append(
            f'<rect x="{x+1}" y="0" width="{KEY_W-2}" height="{KEY_H}" '
            f'rx="2" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
        )
        if is_active:
            note_name = NOTE_NAMES[midi % 12] + str((midi // 12) - 1)
            svg_parts.append(
                f'<text x="{x + KEY_W//2}" y="{KEY_H - 8}" '
                f'text-anchor="middle" font-size="9" font-family="Space Mono" fill="#000">'
                f'{note_name}</text>'
            )

    # Black keys
    for midi, is_black, wi in key_info:
        if not is_black:
            continue
        is_active = (midi == active_note)
        x    = wi * KEY_W - BK_W // 2
        fill = "#00ffa3" if is_active else "#1a1e2a"
        svg_parts.append(
            f'<rect x="{x}" y="0" width="{BK_W}" height="{BK_H}" '
            f'rx="2" fill="{fill}" stroke="#000" stroke-width="1"/>'
        )

    # Note label below piano
    if active_note is not None:
        label = midi_to_name(active_note)
        svg_parts.append(
            f'<text x="{SVG_W//2}" y="{KEY_H + 26}" text-anchor="middle" '
            f'font-size="18" font-family="Syne, sans-serif" font-weight="800" '
            f'fill="#00ffa3" style="filter:drop-shadow(0 0 6px #00ffa3)">{label}</text>'
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


def build_waveform_html(
    waveform: np.ndarray,
    pitch_history: list,
    raw_freq: float,
    corrected_freq: float,
    is_running: bool,
) -> str:
    """
    Pure SVG oscilloscope — no JavaScript, works inside Gradio's sandboxed iframe.
    Waveform rendered as an SVG polyline. Pitch curve as two SVG polylines.
    """
    W, H        = 600, 120
    W_PITCH     = 600
    H_PITCH     = 70
    PADDING     = 4

    # ── Waveform polyline ──────────────────────────────────────────────────────
    n_points = 300
    if len(waveform) > n_points:
        step    = max(1, len(waveform) // n_points)
        wave_ds = waveform[::step][:n_points]
    else:
        wave_ds = waveform if len(waveform) > 0 else np.zeros(2)

    # Build SVG points string
    wave_points = []
    for i, v in enumerate(wave_ds):
        x = PADDING + (i / max(len(wave_ds) - 1, 1)) * (W - 2 * PADDING)
        y = H / 2 - float(v) * (H / 2 - PADDING) * 0.9
        y = max(PADDING, min(H - PADDING, y))
        wave_points.append(f"{x:.1f},{y:.1f}")
    wave_pts_str = " ".join(wave_points)

    # ── Pitch curve polylines ──────────────────────────────────────────────────
    ph       = pitch_history[-80:] if pitch_history else []
    raw_pts  = [p[1] for p in ph]
    corr_pts = [p[2] for p in ph]

    all_hz = [v for v in raw_pts + corr_pts if v > 0]
    if all_hz:
        min_hz  = max(50,   min(all_hz) * 0.93)
        max_hz  = min(1400, max(all_hz) * 1.07)
        hz_range = max_hz - min_hz or 1
    else:
        min_hz, hz_range = 100, 1000

    def make_pitch_polyline(pts: list, color: str) -> str:
        segments = []
        seg = []
        for i, v in enumerate(pts):
            if v <= 0:
                if seg:
                    segments.append(seg)
                    seg = []
                continue
            x = PADDING + (i / max(len(pts) - 1, 1)) * (W_PITCH - 2 * PADDING)
            y = H_PITCH - PADDING - ((v - min_hz) / hz_range) * (H_PITCH - 2 * PADDING)
            y = max(PADDING, min(H_PITCH - PADDING, y))
            seg.append(f"{x:.1f},{y:.1f}")
        if seg:
            segments.append(seg)

        result = ""
        for s in segments:
            if len(s) >= 2:
                result += (
                    f'<polyline points="{" ".join(s)}" fill="none" stroke="{color}" '
                    f'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" '
                    f'opacity="0.9"/>'
                )
        return result

    raw_line  = make_pitch_polyline(raw_pts,  "#ff4eb8")
    corr_line = make_pitch_polyline(corr_pts, "#00ffa3")

    # ── Labels ─────────────────────────────────────────────────────────────────
    status_dot  = "#00ffa3" if is_running else "#6b7280"
    status_txt  = "LIVE" if is_running else "IDLE"
    freq_label  = f"{raw_freq:.1f} Hz" if raw_freq > 0 else "—"
    corr_label  = f"→ {corrected_freq:.1f} Hz" if corrected_freq > 0 else ""

    # Grid lines for waveform (y=25%, 50%, 75%)
    grid_lines = ""
    for frac in [0.25, 0.5, 0.75]:
        y = H * frac
        color = "#252a38" if frac == 0.5 else "#1a1e2a"
        grid_lines += f'<line x1="0" y1="{y:.0f}" x2="{W}" y2="{y:.0f}" stroke="{color}" stroke-width="1"/>'

    # Grid lines for pitch panel
    pitch_grid = ""
    for frac in [0.33, 0.66]:
        y = H_PITCH * frac
        pitch_grid += f'<line x1="0" y1="{y:.0f}" x2="{W_PITCH}" y2="{y:.0f}" stroke="#1a1e2a" stroke-width="1"/>'

    html = f"""
<div style="background:#0c0e14;border:1px solid #252a38;border-radius:8px;
            padding:12px;font-family:Space Mono,monospace;">

  <!-- Header -->
  <div style="display:flex;justify-content:space-between;align-items:center;
              margin-bottom:8px;font-size:0.68rem;color:#6b7280;">
    <span>
      <span style="color:{status_dot}">●</span>&nbsp;OSCILLOSCOPE&nbsp;
      <span style="color:{status_dot};font-size:0.6rem">{status_txt}</span>
    </span>
    <span style="color:#4e9fff">{freq_label}&nbsp;
      <span style="color:#00ffa3">{corr_label}</span>
    </span>
  </div>

  <!-- Waveform SVG -->
  <svg width="100%" height="{H}" viewBox="0 0 {W} {H}"
       xmlns="http://www.w3.org/2000/svg"
       style="display:block;background:#080a10;border-radius:4px;border:1px solid #1a1e2a">
    {grid_lines}
    <defs>
      <linearGradient id="waveGrad" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%"   stop-color="#4e9fff"/>
        <stop offset="50%"  stop-color="#00ffa3"/>
        <stop offset="100%" stop-color="#4e9fff"/>
      </linearGradient>
    </defs>
    <polyline points="{wave_pts_str}"
              fill="none" stroke="url(#waveGrad)"
              stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>

  <!-- Pitch curve header -->
  <div style="margin-top:8px;margin-bottom:4px;font-size:0.65rem;color:#6b7280;">
    PITCH CURVE &nbsp;
    <span style="color:#ff4eb8">■</span> raw &nbsp;
    <span style="color:#00ffa3">■</span> corrected
  </div>

  <!-- Pitch SVG -->
  <svg width="100%" height="{H_PITCH}" viewBox="0 0 {W_PITCH} {H_PITCH}"
       xmlns="http://www.w3.org/2000/svg"
       style="display:block;background:#080a10;border-radius:4px;border:1px solid #1a1e2a">
    {pitch_grid}
    {raw_line}
    {corr_line}
  </svg>

</div>
"""
    return html


def build_tuning_needle_html(cents: float, scale_notes: list, corrector_on: bool) -> str:
    """
    A tuning meter showing how far the raw pitch is from the nearest scale note.
    -100¢ to +100¢ range, center = in tune.
    """
    W, H = 400, 56
    # Needle position: 0 = far left, W = far right, W//2 = center
    needle_x = int((cents + 100) / 200 * W)
    needle_x = max(4, min(W - 4, needle_x))

    if abs(cents) < 10:
        needle_color = "#00ffa3"
        label = "✓ IN TUNE"
        label_color = "#00ffa3"
    elif abs(cents) < 35:
        needle_color = "#ffd700"
        label = f"{cents:+.0f}¢"
        label_color = "#ffd700"
    else:
        needle_color = "#ff4eb8"
        label = f"{cents:+.0f}¢"
        label_color = "#ff4eb8"

    if not corrector_on:
        needle_color = "#4e9fff"
        label = "correction off"
        label_color = "#6b7280"

    scale_str = "  ".join(scale_notes) if scale_notes else "—"

    return f"""
<div style="background:#0c0e14;border:1px solid #252a38;border-radius:8px;
            padding:10px 14px;font-family:Space Mono,monospace;">
  <div style="font-size:0.65rem;color:#6b7280;margin-bottom:6px;letter-spacing:0.1em">
    TUNING METER &nbsp;|&nbsp;
    <span style="color:#e8eaf0">{scale_str}</span>
  </div>
  <svg width="100%" height="{H}" viewBox="0 0 {W} {H}"
       xmlns="http://www.w3.org/2000/svg" style="display:block">
    <!-- Track -->
    <rect x="0" y="20" width="{W}" height="8" rx="4" fill="#1a1e2a"/>
    <!-- Green zone center -->
    <rect x="{W//2 - 20}" y="20" width="40" height="8" rx="3" fill="#0d2e1f"/>
    <!-- Center tick -->
    <line x1="{W//2}" y1="14" x2="{W//2}" y2="36" stroke="#252a38" stroke-width="2"/>
    <!-- -50/+50 ticks -->
    <line x1="{W//4}" y1="18" x2="{W//4}" y2="30" stroke="#1a1e2a" stroke-width="1"/>
    <line x1="{3*W//4}" y1="18" x2="{3*W//4}" y2="30" stroke="#1a1e2a" stroke-width="1"/>
    <!-- Needle -->
    <rect x="{needle_x - 2}" y="16" width="4" height="16" rx="2" fill="{needle_color}"/>
    <circle cx="{needle_x}" cy="24" r="5" fill="{needle_color}"
            style="filter:drop-shadow(0 0 4px {needle_color})"/>
    <!-- Labels -->
    <text x="2"     y="50" font-size="9" font-family="Space Mono" fill="#6b7280">-100¢</text>
    <text x="{W//2 - 6}" y="50" font-size="9" font-family="Space Mono" fill="#6b7280">0</text>
    <text x="{W - 32}" y="50" font-size="9" font-family="Space Mono" fill="#6b7280">+100¢</text>
    <!-- Reading -->
    <text x="{W//2}" y="13" text-anchor="middle" font-size="10"
          font-family="Space Mono" fill="{label_color}">{label}</text>
  </svg>
</div>
"""


def build_chord_display_html(chord_evt, chord_history: list) -> str:
    """Display current chord and recent chord history."""
    if chord_evt is None and not chord_history:
        return (
            '<div style="background:#0c0e14;border:1px solid #252a38;border-radius:8px;'
            'padding:12px;font-family:Space Mono,monospace;font-size:0.72rem;color:#6b7280;'
            'text-align:center">Enable Chord Detection to see chords here</div>'
        )

    # Big current chord
    if chord_evt:
        root    = NOTE_NAMES[chord_evt["root_note"] % 12] if isinstance(chord_evt, dict) \
                  else NOTE_NAMES[chord_evt.root_note % 12]
        ctype   = chord_evt["chord_type"] if isinstance(chord_evt, dict) else chord_evt.chord_type
        conf    = chord_evt["confidence"] if isinstance(chord_evt, dict) else chord_evt.confidence
        notes_l = chord_evt["notes"] if isinstance(chord_evt, dict) else chord_evt.notes
        note_names = "  ".join(midi_to_name(n) for n in notes_l)
        conf_pct = int(conf * 100)
    else:
        root, ctype, conf_pct, note_names = "—", "—", 0, ""

    rows = ""
    for ce in reversed((chord_history or [])[-6:]):
        r2    = NOTE_NAMES[(ce.root_note if hasattr(ce, "root_note") else ce["root_note"]) % 12]
        ct2   = ce.chord_type if hasattr(ce, "chord_type") else ce["chord_type"]
        age   = time.time() - (ce.timestamp if hasattr(ce, "timestamp") else ce["timestamp"])
        op    = max(0.25, 1 - age / 10)
        rows += (
            f'<div style="opacity:{op:.2f};display:flex;gap:12px;padding:3px 0;'
            f'border-bottom:1px solid #1a1e2a;font-size:0.68rem;">'
            f'<span style="color:#ff4eb8;width:24px">{r2}</span>'
            f'<span style="color:#e8eaf0">{ct2}</span>'
            f'</div>'
        )

    return f"""
<div style="background:#0c0e14;border:1px solid #252a38;border-radius:8px;
            padding:12px;font-family:Space Mono,monospace;">
  <div style="text-align:center;margin-bottom:10px;">
    <span style="font-family:Syne,sans-serif;font-size:2rem;font-weight:800;
                 color:#ff4eb8;text-shadow:0 0 16px #ff4eb8">{root}</span>
    <span style="font-size:1rem;color:#e8eaf0;margin-left:8px">{ctype}</span>
    <div style="font-size:0.65rem;color:#6b7280;margin-top:4px">{note_names}</div>
    <div style="margin-top:6px;background:#1a1e2a;border-radius:3px;height:4px;width:80%;margin-left:auto;margin-right:auto">
      <div style="width:{conf_pct}%;height:100%;background:#ff4eb8;border-radius:3px"></div>
    </div>
    <div style="font-size:0.62rem;color:#6b7280;margin-top:2px">conf {conf_pct}%</div>
  </div>
  <div style="border-top:1px solid #1a1e2a;padding-top:8px;max-height:120px;overflow-y:auto">
    {rows}
  </div>
</div>
"""


def build_session_list_html(sessions: list) -> str:
    """Render the saved session list as an HTML table."""
    if not sessions:
        return (
            '<div style="font-family:Space Mono,monospace;font-size:0.72rem;'
            'color:#6b7280;text-align:center;padding:16px">No saved sessions yet</div>'
        )
    rows = ""
    for s in sessions:
        ts    = s.get("timestamp", 0)
        label = s.get("label", s["session_id"])
        mode  = s.get("mode", "?")
        nc    = s.get("note_count", 0)
        sid   = s["session_id"]
        rows += (
            f'<div style="display:flex;align-items:center;gap:10px;padding:6px 0;'
            f'border-bottom:1px solid #1a1e2a;font-size:0.68rem;font-family:Space Mono,monospace;">'
            f'<div style="flex:1;min-width:0">'
            f'  <div style="color:#e8eaf0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{label}</div>'
            f'  <div style="color:#6b7280;margin-top:2px">{mode} · {nc} notes</div>'
            f'</div>'
            f'<span style="color:#4e9fff;cursor:pointer;white-space:nowrap" '
            f'onclick="document.getElementById(\'sel_session\').value=\'{sid}\';">'
            f'select</span>'
            f'</div>'
        )
    return f'<div style="max-height:300px;overflow-y:auto">{rows}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Build Gradio app
# ─────────────────────────────────────────────────────────────────────────────

def create_ui(engine: AudioEngine) -> gr.Blocks:
    gpu_info = check_gpu()
    gpu_label = f"🚀 {gpu_info['name']}" if gpu_info["available"] else "💻 CPU"

    mode_choices   = list(MODES.keys())
    mode_descs     = {k: v["description"] for k, v in MODES.items()}
    scale_choices  = list(SCALES.keys())
    root_choices   = ROOT_NOTES
    vcurve_choices = list(VELOCITY_CURVES.keys())

    with gr.Blocks(title="Voice to MIDI") as demo:

        # ── Header ────────────────────────────────────────────────────────────
        with gr.Row():
            gr.Markdown("# 🎙️ Voice to MIDI\n**Real-time voice pitch detection → MIDI notes**")
            gr.Markdown(
                f"<div style='text-align:right;font-family:Space Mono;"
                f"font-size:0.7rem;color:#6b7280;padding-top:12px'>{gpu_label}</div>"
            )
        gr.HTML("<hr style='border-color:#252a38;margin:0 0 16px 0'>")

        with gr.Tabs():

            # ════════════════════════════════════════════════════════════════
            # TAB 1 — MAIN STUDIO
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("🎛 Studio"):
                with gr.Row(equal_height=False):

                    # ── Left: controls ────────────────────────────────────
                    with gr.Column(scale=1, min_width=280):
                        gr.Markdown("### ⚙ Controls")
                        with gr.Group():
                            start_btn = gr.Button("▶  Start Listening", variant="primary")
                            stop_btn  = gr.Button("⏹  Stop")

                        gr.HTML("<div style='height:10px'></div>")
                        gr.Markdown("### 🎛 Instrument Mode")
                        mode_radio = gr.Radio(
                            choices=mode_choices, value=mode_choices[0],
                            label="", interactive=True,
                        )
                        mode_desc_txt = gr.Markdown(f"*{mode_descs[mode_choices[0]]}*")

                        gr.HTML("<div style='height:10px'></div>")
                        gr.Markdown("### 🎚 Settings")
                        sensitivity_slider = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.65, step=0.05,
                            label="Detection Sensitivity",
                            info="Higher = detect quieter notes"
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

                        gr.HTML("<div style='height:10px'></div>")
                        gr.Markdown("### 🎵 Pitch Correction")
                        with gr.Group():
                            pc_enable = gr.Checkbox(label="Enable Pitch Correction", value=False)
                            with gr.Row():
                                pc_root  = gr.Dropdown(choices=root_choices, value="C",
                                                       label="Root Key", scale=1)
                                pc_scale = gr.Dropdown(choices=scale_choices, value="Major",
                                                       label="Scale", scale=2)
                            pc_strength = gr.Slider(0.0, 1.0, value=0.8, step=0.05,
                                                    label="Strength",
                                                    info="0 = monitor only  |  1 = hard snap")
                            pc_speed    = gr.Slider(0.0, 0.9, value=0.3, step=0.05,
                                                    label="Speed",
                                                    info="0 = instant  |  0.9 = gradual")

                        gr.HTML("<div style='height:10px'></div>")
                        export_btn  = gr.Button("💾  Export to .mid file")
                        export_file = gr.File(label="Download MIDI", visible=False)
                        export_msg  = gr.Markdown("")
                        save_btn    = gr.Button("📁  Save Session")
                        save_msg    = gr.Markdown("")

                    # ── Right: visualizers ────────────────────────────────
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Input Level")
                        level_html = gr.HTML(build_level_meter(0))

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

                        gr.Markdown("### 〰 Waveform & Pitch")
                        waveform_html = gr.HTML(
                            build_waveform_html(np.zeros(100), [], 0.0, 0.0, False)
                        )

                        gr.Markdown("### 🎯 Tuning")
                        tuning_html = gr.HTML(
                            build_tuning_needle_html(0.0, ["C","D","E","F","G","A","B"], False)
                        )

                        gr.Markdown("### 🎹 Piano Roll")
                        piano_html = gr.HTML(build_piano_html())

                        gr.Markdown("### 📋 Note History")
                        gr.HTML(
                            '<div style="font-family:Space Mono;font-size:0.65rem;color:#6b7280;'
                            'display:flex;gap:24px;padding:4px 0;border-bottom:1px solid #252a38;'
                            'margin-bottom:4px">'
                            '<span>NOTE</span><span>  #</span><span>  FREQ</span>'
                            '<span style="flex:1;text-align:center">VELOCITY</span>'
                            '<span>CONF</span></div>'
                        )
                        history_html = gr.HTML(build_note_history_html([]))

            # ════════════════════════════════════════════════════════════════
            # TAB 2 — PROCESSING
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("⚗ Processing"):
                with gr.Row():

                    # ── Scale Quantization ────────────────────────────────
                    with gr.Column():
                        gr.Markdown("### 🎼 Scale / Key Quantization")
                        gr.Markdown(
                            "*Snaps MIDI output notes to only the notes in your chosen key. "
                            "Applied after pitch detection and pitch correction.*"
                        )
                        sq_enable = gr.Checkbox(label="Enable Scale Quantization", value=False)
                        with gr.Row():
                            sq_root  = gr.Dropdown(choices=root_choices, value="C",
                                                   label="Root Key", scale=1)
                            sq_scale = gr.Dropdown(choices=list(SCALE_INTERVALS.keys()),
                                                   value="Major", label="Scale", scale=2)
                        sq_notes_display = gr.Markdown("*Active notes: C  D  E  F  G  A  B*")

                        gr.HTML("<div style='height:16px'></div>")

                        # ── Note Smoothing ─────────────────────────────────
                        gr.Markdown("### 🔇 Note Smoothing")
                        gr.Markdown(
                            "*Filters out jitter and micro-notes. "
                            "Raise min duration to remove stutter; raise gap fill to "
                            "bridge brief silences in held notes.*"
                        )
                        ns_enable   = gr.Checkbox(label="Enable Note Smoothing", value=True)
                        ns_min_dur  = gr.Slider(0.02, 0.3, value=0.08, step=0.01,
                                                label="Min Note Duration (s)",
                                                info="Notes shorter than this are dropped")
                        ns_gap_fill = gr.Slider(0.02, 0.2, value=0.06, step=0.01,
                                                label="Gap Fill (s)",
                                                info="Gaps shorter than this between same note are bridged")

                    # ── Velocity Shaping + Chord Detection ────────────────
                    with gr.Column():
                        gr.Markdown("### 🎚 Velocity Shaping")
                        gr.Markdown(
                            "*Maps how loud you sing to MIDI velocity. "
                            "Choose a curve to change the dynamic feel.*"
                        )
                        vs_enable  = gr.Checkbox(label="Enable Velocity Shaping", value=True)
                        vs_curve   = gr.Radio(
                            choices=vcurve_choices, value="Linear",
                            label="Velocity Curve"
                        )
                        with gr.Row():
                            vs_min = gr.Slider(1, 80,  value=20, step=1, label="Min Velocity")
                            vs_max = gr.Slider(80, 127, value=127, step=1, label="Max Velocity")

                        gr.HTML("<div style='height:16px'></div>")

                        # ── Chord Detection ────────────────────────────────
                        gr.Markdown("### 🎸 Chord Detection")
                        gr.Markdown(
                            "*Analyses your voice's harmonic overtones to detect chords. "
                            "Works best on open vowel sounds (ahh, ohh). "
                            "When a chord is detected, all its notes are sent to MIDI.*"
                        )
                        cd_enable = gr.Checkbox(label="Enable Chord Detection", value=False)
                        cd_thresh = gr.Slider(0.3, 0.95, value=0.55, step=0.05,
                                             label="Detection Confidence Threshold",
                                             info="Higher = stricter matching, fewer false positives")

                        gr.HTML("<div style='height:12px'></div>")
                        gr.Markdown("### 🎸 Last Detected Chord")
                        chord_display = gr.HTML(build_chord_display_html(None, []))

            # ════════════════════════════════════════════════════════════════
            # TAB 3 — SESSION HISTORY
            # ════════════════════════════════════════════════════════════════
            with gr.Tab("📂 Sessions"):
                gr.Markdown(
                    "### 📂 Session History\n"
                    "*Sessions are auto-saved when you click **Save Session**. "
                    "Each session stores all detected notes and settings so you can "
                    "re-export any recording at any time.*"
                )
                with gr.Row():
                    refresh_sessions_btn = gr.Button("🔄  Refresh List")
                    sh_bpm = gr.Slider(60, 200, value=120, step=1,
                                       label="Export BPM", scale=2)

                sessions_html   = gr.HTML(build_session_list_html(list_sessions()))
                selected_session = gr.Textbox(label="Selected Session ID",
                                              elem_id="sel_session",
                                              placeholder="Click 'select' on a session above…")

                with gr.Row():
                    sh_export_btn = gr.Button("💾  Export Selected to MIDI")
                    sh_delete_btn = gr.Button("🗑  Delete Selected")

                sh_export_file = gr.File(label="Download MIDI", visible=False)
                sh_msg         = gr.Markdown("")

        # ── State & Timer ─────────────────────────────────────────────────────
        is_running_state = gr.State(False)
        refresh_timer    = gr.Timer(value=0.1, active=False)

        # ── Event handlers ─────────────────────────────────────────────────────

        def on_start(sensitivity):
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
            engine.display_note = None
            return (
                False,
                '<span style="background:#1a1e2a;border:1px solid #252a38;'
                'padding:4px 12px;border-radius:4px;font-family:Space Mono;'
                'font-size:0.7rem;color:#6b7280">⏸ Idle</span>',
                gr.update(active=False),
                build_piano_html(),
                build_level_meter(0),
                build_waveform_html(np.zeros(100), [], 0.0, 0.0, False),
                build_tuning_needle_html(0.0, [], False),
            )

        def on_mode_change(mode):
            engine.set_mode(mode)
            return f"*{mode_descs.get(mode, '')}*"

        def on_sensitivity_change(val):
            import audio_engine as ae
            ae.CONF_THRESH = float(val)

        def on_device_change(dev_str):
            try:
                engine.set_input_device(int(dev_str.split(":")[0]))
            except Exception:
                pass

        # Pitch correction
        def on_pc_enable(v):   engine.pitch_corrector.enabled  = v
        def on_pc_root(v):     engine.pitch_corrector.set_scale(v, engine.pitch_corrector.scale)
        def on_pc_scale(v):    engine.pitch_corrector.set_scale(engine.pitch_corrector.root, v)
        def on_pc_strength(v): engine.pitch_corrector.strength = float(v)
        def on_pc_speed(v):    engine.pitch_corrector.speed    = float(v)

        # Scale quantization
        def on_sq_enable(v):
            engine.scale_quantizer.enabled = v
        def on_sq_root(root, scale):
            engine.scale_quantizer.set_scale(root, scale)
            notes = "  ".join(engine.scale_quantizer.active_notes)
            return f"*Active notes: {notes}*"
        def on_sq_scale(root, scale):
            engine.scale_quantizer.set_scale(root, scale)
            notes = "  ".join(engine.scale_quantizer.active_notes)
            return f"*Active notes: {notes}*"

        # Note smoothing
        def on_ns_enable(v):   engine.note_smoother.enabled      = v
        def on_ns_min(v):      engine.note_smoother.min_duration  = float(v)
        def on_ns_gap(v):      engine.note_smoother.gap_fill       = float(v)

        # Velocity shaping
        def on_vs_enable(v):   engine.velocity_shaper.enabled  = v
        def on_vs_curve(v):    engine.velocity_shaper.curve    = v
        def on_vs_min(v):      engine.velocity_shaper.min_vel  = int(v)
        def on_vs_max(v):      engine.velocity_shaper.max_vel  = int(v)

        # Chord detection
        def on_cd_enable(v):   engine.chord_detector.enabled             = v
        def on_cd_thresh(v):   engine.chord_detector.confidence_thresh   = float(v)

        def on_refresh():
            note   = engine.latest_note
            hist   = engine.note_history
            rms_v  = engine.latest_rms
            freq_v = engine.latest_freq
            conf_v = engine.latest_conf
            corr_v = engine.latest_corrected_freq

            # Use display_note: set when note fires, cleared only on silence.
            # This means the piano key stays lit while singing and between notes.
            active_midi = engine.display_note

            with engine._waveform_lock:
                wave_snap = engine.waveform_buffer.copy()

            pitch_hist = list(engine.pitch_history)
            pc_on      = engine.pitch_corrector.enabled
            cents      = engine.pitch_corrector.cents_deviation(freq_v) if freq_v > 0 else 0.0
            scale_nms  = engine.pitch_corrector.scale_note_names

            piano    = build_piano_html(active_midi)
            hist_h   = build_note_history_html(hist)
            level    = build_level_meter(rms_v)
            waveform = build_waveform_html(wave_snap, pitch_hist, freq_v, corr_v, engine.is_running)
            tuning   = build_tuning_needle_html(cents, scale_nms, pc_on)
            chord_h  = build_chord_display_html(engine.latest_chord, engine.chord_history)

            freq_html = (
                f'<span style="font-family:Space Mono;font-size:0.75rem;color:#4e9fff">'
                f'{freq_v:.1f} Hz'
                + (f' <span style="color:#00ffa3">→ {corr_v:.1f} Hz</span>'
                   if pc_on and corr_v > 0 else '')
                + f' &nbsp;|&nbsp; <span style="color:#00ffa3">conf: {int(conf_v*100)}%</span></span>'
            ) if freq_v > 0 else (
                '<span style="font-family:Space Mono;font-size:0.7rem;color:#6b7280">'
                '— Hz  |  conf: —</span>'
            )

            return piano, hist_h, level, freq_html, waveform, tuning, chord_h

        def on_export(bpm):
            if not engine.note_history:
                return gr.update(visible=False), "⚠️ No notes recorded yet."
            try:
                from midi_export import export_to_midi
                path = export_to_midi(engine.note_history, engine._mode, int(bpm))
                return gr.update(value=path, visible=True), f"✅ Exported {len(engine.note_history)} notes"
            except Exception as e:
                return gr.update(visible=False), f"❌ Export failed: {e}"

        def on_save_session():
            if not engine.note_history:
                return "⚠️ Nothing to save — record some notes first."
            settings = {
                "pitch_correction": {
                    "enabled":  engine.pitch_corrector.enabled,
                    "root":     engine.pitch_corrector.root,
                    "scale":    engine.pitch_corrector.scale,
                    "strength": engine.pitch_corrector.strength,
                },
                "scale_quantizer": {
                    "enabled": engine.scale_quantizer.enabled,
                    "root":    engine.scale_quantizer.root,
                    "scale":   engine.scale_quantizer.scale,
                },
                "velocity_shaper": {
                    "enabled": engine.velocity_shaper.enabled,
                    "curve":   engine.velocity_shaper.curve,
                },
            }
            sid = save_session(engine.note_history, engine._mode, settings=settings)
            return f"✅ Session saved: {sid}"

        def on_refresh_sessions():
            return build_session_list_html(list_sessions())

        def on_sh_export(session_id, bpm):
            if not session_id:
                return gr.update(visible=False), "⚠️ No session selected."
            try:
                path = export_session_to_midi(session_id, int(bpm))
                if not path:
                    return gr.update(visible=False), "❌ Session not found or empty."
                return gr.update(value=path, visible=True), "✅ Exported!"
            except Exception as e:
                return gr.update(visible=False), f"❌ {e}"

        def on_sh_delete(session_id):
            if not session_id:
                return "⚠️ No session selected.", build_session_list_html(list_sessions())
            delete_session(session_id)
            return "✅ Deleted.", build_session_list_html(list_sessions())

        # ── Wire up all events ─────────────────────────────────────────────────
        start_btn.click(on_start, [sensitivity_slider],
                        [is_running_state, status_badge, refresh_timer])
        stop_btn.click(on_stop, [],
                       [is_running_state, status_badge, refresh_timer,
                        piano_html, level_html, waveform_html, tuning_html])

        mode_radio.change(on_mode_change, [mode_radio], [mode_desc_txt])
        sensitivity_slider.change(on_sensitivity_change, [sensitivity_slider])
        device_dropdown.change(on_device_change, [device_dropdown])

        pc_enable.change(on_pc_enable, [pc_enable])
        pc_root.change(on_pc_root, [pc_root])
        pc_scale.change(on_pc_scale, [pc_scale])
        pc_strength.change(on_pc_strength, [pc_strength])
        pc_speed.change(on_pc_speed, [pc_speed])

        sq_enable.change(on_sq_enable, [sq_enable])
        sq_root.change(on_sq_root,   [sq_root, sq_scale], [sq_notes_display])
        sq_scale.change(on_sq_scale, [sq_root, sq_scale], [sq_notes_display])

        ns_enable.change(on_ns_enable, [ns_enable])
        ns_min_dur.change(on_ns_min,   [ns_min_dur])
        ns_gap_fill.change(on_ns_gap,  [ns_gap_fill])

        vs_enable.change(on_vs_enable, [vs_enable])
        vs_curve.change(on_vs_curve,   [vs_curve])
        vs_min.change(on_vs_min, [vs_min])
        vs_max.change(on_vs_max, [vs_max])

        cd_enable.change(on_cd_enable, [cd_enable])
        cd_thresh.change(on_cd_thresh, [cd_thresh])

        refresh_timer.tick(
            on_refresh,
            outputs=[piano_html, history_html, level_html, freq_display,
                     waveform_html, tuning_html, chord_display],
        )

        export_btn.click(on_export, [bpm_slider], [export_file, export_msg])
        save_btn.click(on_save_session, [], [save_msg])

        refresh_sessions_btn.click(on_refresh_sessions, [], [sessions_html])
        sh_export_btn.click(on_sh_export, [selected_session, sh_bpm],
                            [sh_export_file, sh_msg])
        sh_delete_btn.click(on_sh_delete, [selected_session],
                            [sh_msg, sessions_html])

    return demo, CUSTOM_CSS, gr.themes.Base(primary_hue="emerald", neutral_hue="slate")
    gpu_info = check_gpu()
