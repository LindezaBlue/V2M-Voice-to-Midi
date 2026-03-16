"""
midi_export.py
Converts a list of NoteEvent objects into a downloadable .mid file.
"""

import tempfile
import os
from typing import List
from audio_engine import NoteEvent, MODES


def export_to_midi(events: List[NoteEvent], mode: str, bpm: int = 120) -> str:
    """
    Takes a list of NoteEvent objects and exports a .mid file.
    Returns the path to the temporary file.
    """
    try:
        from midiutil import MIDIFile
    except ImportError:
        raise RuntimeError("midiutil not installed")

    if not events:
        raise ValueError("No notes to export")

    mode_cfg = MODES.get(mode, MODES["Vocals / Melody"])
    channel  = mode_cfg["midi_channel"]
    track    = 0

    midi = MIDIFile(1)
    midi.addTempo(track, 0, bpm)
    midi.addProgramChange(track, channel, 0, mode_cfg["program"])

    # Sort by timestamp
    sorted_events = sorted([e for e in events if e.active], key=lambda e: e.timestamp)
    if not sorted_events:
        raise ValueError("No active notes to export")

    t0 = sorted_events[0].timestamp

    for i, evt in enumerate(sorted_events):
        beat_start = (evt.timestamp - t0) * (bpm / 60.0)

        # Duration: time until next note, or default 0.5 beats
        if i + 1 < len(sorted_events):
            next_t = sorted_events[i + 1].timestamp
            duration = (next_t - evt.timestamp) * (bpm / 60.0)
            duration = max(0.1, min(duration, 4.0))
        else:
            duration = 0.5

        velocity = max(30, min(127, evt.velocity))
        midi.addNote(track, channel, evt.note, beat_start, duration, velocity)

    # Write to temp file
    tmp = tempfile.NamedTemporaryFile(
        suffix=".mid", prefix="voice_to_midi_", delete=False
    )
    with open(tmp.name, "wb") as f:
        midi.writeFile(f)

    return tmp.name
