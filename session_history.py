"""
session_history.py
Save and reload past Voice-to-MIDI recording sessions.

Each session stores:
  - timestamp & name
  - instrument mode
  - list of NoteEvent dicts
  - pitch correction settings
  - processing settings (scale, velocity curve, etc.)

Sessions are stored as JSON files in a local ./sessions/ folder.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional
from dataclasses import asdict

SESSIONS_DIR = Path("./sessions")


def _ensure_dir():
    SESSIONS_DIR.mkdir(exist_ok=True)


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def save_session(
    note_events: list,
    mode: str,
    name: Optional[str] = None,
    settings: Optional[dict] = None,
) -> str:
    """
    Persist a session to disk.
    Returns the session_id (timestamp-based filename stem).
    """
    _ensure_dir()

    ts         = time.time()
    session_id = f"session_{int(ts)}"
    label      = name or time.strftime("%b %d %Y  %I:%M %p", time.localtime(ts))

    # Serialise NoteEvents — they're dataclasses so use __dict__
    events_data = []
    for evt in note_events:
        try:
            d = {
                "note":       evt.note,
                "frequency":  evt.frequency,
                "confidence": evt.confidence,
                "velocity":   evt.velocity,
                "timestamp":  evt.timestamp,
                "name":       evt.name,
                "active":     evt.active,
            }
            events_data.append(d)
        except Exception:
            pass

    data = {
        "session_id": session_id,
        "label":      label,
        "timestamp":  ts,
        "mode":       mode,
        "note_count": len(events_data),
        "events":     events_data,
        "settings":   settings or {},
    }

    _session_path(session_id).write_text(json.dumps(data, indent=2))
    return session_id


def list_sessions() -> list[dict]:
    """Return all saved sessions, newest first, without the full events list."""
    _ensure_dir()
    sessions = []
    for p in sorted(SESSIONS_DIR.glob("session_*.json"), reverse=True):
        try:
            raw  = json.loads(p.read_text())
            sessions.append({
                "session_id": raw["session_id"],
                "label":      raw.get("label", raw["session_id"]),
                "timestamp":  raw.get("timestamp", 0),
                "mode":       raw.get("mode", "?"),
                "note_count": raw.get("note_count", 0),
            })
        except Exception:
            pass
    return sessions


def load_session(session_id: str) -> Optional[dict]:
    """Load and return the full session dict including events, or None."""
    p = _session_path(session_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def delete_session(session_id: str) -> bool:
    p = _session_path(session_id)
    if p.exists():
        p.unlink()
        return True
    return False


def session_to_note_events(session: dict) -> list:
    """
    Reconstruct a list of lightweight NoteEvent-like objects
    from a saved session dict so they can be re-exported to MIDI.
    """
    from audio_engine import NoteEvent
    events = []
    for d in session.get("events", []):
        try:
            evt = NoteEvent(
                note       = d["note"],
                frequency  = d["frequency"],
                confidence = d["confidence"],
                velocity   = d["velocity"],
                timestamp  = d["timestamp"],
                active     = d.get("active", True),
            )
            events.append(evt)
        except Exception:
            pass
    return events


def export_session_to_midi(session_id: str, bpm: int = 120) -> Optional[str]:
    """Load a session and export it to a .mid file. Returns the file path."""
    session = load_session(session_id)
    if not session:
        return None
    from audio_engine import NoteEvent
    from midi_export import export_to_midi
    events = session_to_note_events(session)
    if not events:
        return None
    return export_to_midi(events, session.get("mode", "Vocals / Melody"), bpm)


def format_duration(note_events: list) -> str:
    """Return a human-readable duration string for a session."""
    active = [e for e in note_events if hasattr(e, "timestamp")]
    if len(active) < 2:
        return "< 1s"
    dur = active[-1]["timestamp"] - active[0]["timestamp"] if isinstance(active[0], dict) \
          else active[-1].timestamp - active[0].timestamp
    if dur < 60:
        return f"{int(dur)}s"
    return f"{int(dur // 60)}m {int(dur % 60)}s"
