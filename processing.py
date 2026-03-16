"""
processing.py
Four signal-processing layers applied between raw pitch detection and MIDI output:

  1. ScaleQuantizer   — snap MIDI notes to a chosen key/scale
  2. NoteSmoothing    — gate out jitter & micro-notes
  3. VelocityShaper   — apply expressive curves to raw RMS velocity
  4. ChordDetector    — detect harmonic content and emit chord MIDI events
"""

import numpy as np
import time
from typing import Optional
from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────────────────
# 1. Scale Quantizer
# ─────────────────────────────────────────────────────────────────────────────

# Re-use the same scale table from pitch_correction for consistency
SCALE_INTERVALS = {
    "Chromatic":        list(range(12)),
    "Major":            [0, 2, 4, 5, 7, 9, 11],
    "Natural Minor":    [0, 2, 3, 5, 7, 8, 10],
    "Harmonic Minor":   [0, 2, 3, 5, 7, 8, 11],
    "Pentatonic Major": [0, 2, 4, 7, 9],
    "Pentatonic Minor": [0, 3, 5, 7, 10],
    "Blues":            [0, 3, 5, 6, 7, 10],
    "Dorian":           [0, 2, 3, 5, 7, 9, 10],
    "Mixolydian":       [0, 2, 4, 5, 7, 9, 10],
    "Lydian":           [0, 2, 4, 6, 7, 9, 11],
    "Phrygian":         [0, 1, 3, 5, 7, 8, 10],
    "Whole Tone":       [0, 2, 4, 6, 8, 10],
    "Diminished":       [0, 2, 3, 5, 6, 8, 9, 11],
}

ROOT_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]


class ScaleQuantizer:
    """
    Snaps a raw MIDI note number to the nearest note in the chosen key/scale.
    Applied at the MIDI stage (after pitch detection + pitch correction).
    """

    def __init__(self):
        self.enabled  = False
        self.root     = "C"
        self.scale    = "Major"
        self._scale_set: set[int] = self._build_set("C", "Major")

    def _build_set(self, root: str, scale: str) -> set[int]:
        root_idx   = ROOT_NAMES.index(root)
        intervals  = SCALE_INTERVALS.get(scale, list(range(12)))
        return {(root_idx + i) % 12 for i in intervals}

    def set_scale(self, root: str, scale: str):
        self.root  = root
        self.scale = scale
        self._scale_set = self._build_set(root, scale)

    def quantize(self, midi_note: int) -> int:
        """Return the nearest in-scale MIDI note."""
        if not self.enabled:
            return midi_note
        semitone = midi_note % 12
        if semitone in self._scale_set:
            return midi_note   # already in scale
        # Search outward ±1, ±2 … for nearest in-scale semitone
        for delta in range(1, 7):
            for sign in (-1, 1):
                candidate = (semitone + sign * delta) % 12
                if candidate in self._scale_set:
                    return max(0, min(127, midi_note + sign * delta))
        return midi_note   # fallback

    @property
    def active_notes(self) -> list[str]:
        root_idx  = ROOT_NAMES.index(self.root)
        intervals = SCALE_INTERVALS.get(self.scale, [])
        return [ROOT_NAMES[(root_idx + i) % 12] for i in intervals]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Note Smoother
# ─────────────────────────────────────────────────────────────────────────────

class NoteSmoother:
    """
    Filters jitter and micro-notes from the MIDI stream.

    Two mechanisms:
      • min_duration  — a note must be held for at least N ms before it fires
      • gap_fill      — gaps shorter than N ms between the *same* note are
                        ignored (note is considered still held)
    """

    def __init__(self):
        self.enabled      = True
        self.min_duration = 0.08   # seconds — notes shorter than this are dropped
        self.gap_fill     = 0.06   # seconds — same-note gaps shorter than this are bridged

        self._pending_note : Optional[int]  = None
        self._pending_since: float          = 0.0
        self._last_note    : Optional[int]  = None
        self._last_off_t   : float          = 0.0

    def should_emit(self, midi_note: int, now: float) -> bool:
        """
        Call each time a note candidate arrives.
        Returns True if the note should be emitted to MIDI output.
        """
        if not self.enabled:
            return True

        # Gap-fill: same note returned quickly after going silent → still on
        if (midi_note == self._last_note
                and (now - self._last_off_t) < self.gap_fill):
            self._pending_note  = midi_note
            self._pending_since = now - self.min_duration  # treat as already stable
            return True

        # New or changed note — start the stability timer
        if midi_note != self._pending_note:
            self._pending_note  = midi_note
            self._pending_since = now
            return False

        # Same as pending — check if it has been held long enough
        return (now - self._pending_since) >= self.min_duration

    def note_off(self, midi_note: int, now: float):
        """Call when a note goes off so gap-fill logic can track it."""
        self._last_note  = midi_note
        self._last_off_t = now


# ─────────────────────────────────────────────────────────────────────────────
# 3. Velocity Shaper
# ─────────────────────────────────────────────────────────────────────────────

VELOCITY_CURVES = {
    "Linear":      lambda x: x,
    "Soft":        lambda x: x ** 0.5,          # boost quiet notes
    "Hard":        lambda x: x ** 2.0,          # compress quiet, punch louder
    "Compressed":  lambda x: 0.4 + 0.6 * x,    # narrow dynamic range
    "Accented":    lambda x: np.clip(x * 1.4 - 0.1, 0, 1),  # high contrast
    "Pianissimo":  lambda x: x * 0.45,          # everything soft
    "Fortissimo":  lambda x: np.clip(x * 1.6, 0, 1),        # everything loud
}


class VelocityShaper:
    """
    Maps raw RMS amplitude (0–1) through an expressive curve
    to produce the final MIDI velocity (1–127).
    """

    def __init__(self):
        self.enabled    = True
        self.curve      = "Linear"
        self.min_vel    = 20    # floor — never go below this
        self.max_vel    = 127   # ceiling

    def shape(self, raw_rms: float) -> int:
        """
        raw_rms: 0.0–1.0 (normalised amplitude from the mic)
        Returns: MIDI velocity 1–127
        """
        # Normalise RMS to 0–1 range (typical RMS tops out ~0.3 for voice)
        normalised = float(np.clip(raw_rms / 0.3, 0.0, 1.0))

        if self.enabled:
            fn  = VELOCITY_CURVES.get(self.curve, VELOCITY_CURVES["Linear"])
            shaped = float(fn(normalised))
        else:
            shaped = normalised

        vel = int(self.min_vel + shaped * (self.max_vel - self.min_vel))
        return max(1, min(127, vel))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Chord Detector
# ─────────────────────────────────────────────────────────────────────────────

# Common chord templates as semitone intervals from root
CHORD_TEMPLATES = {
    "Major":       [0, 4, 7],
    "Minor":       [0, 3, 7],
    "Dominant 7":  [0, 4, 7, 10],
    "Major 7":     [0, 4, 7, 11],
    "Minor 7":     [0, 3, 7, 10],
    "Diminished":  [0, 3, 6],
    "Augmented":   [0, 4, 8],
    "Sus2":        [0, 2, 7],
    "Sus4":        [0, 5, 7],
    "Power":       [0, 7],
}


@dataclass
class ChordEvent:
    root_note:   int          # MIDI root note
    chord_type:  str          # e.g. "Major"
    notes:       list[int]    # all MIDI notes in the chord
    confidence:  float        # 0–1
    timestamp:   float

    @property
    def name(self) -> str:
        root_name = ROOT_NAMES[self.root_note % 12]
        return f"{root_name} {self.chord_type}"


class ChordDetector:
    """
    Analyses a short audio buffer for harmonic content using the
    Harmonic Product Spectrum (HPS) method, then matches detected
    partials to chord templates.

    This works best on open vowel sounds ("ahh", "ohh") where
    multiple harmonics are present simultaneously.
    """

    def __init__(self):
        self.enabled         = False
        self.confidence_thresh = 0.55
        self._last_chord     : Optional[ChordEvent] = None
        self._chord_history  : list[ChordEvent]     = []

    def detect(self, audio: np.ndarray, sample_rate: int) -> Optional[ChordEvent]:
        """
        Analyse audio buffer and return a ChordEvent if a chord is detected,
        otherwise None.
        """
        if not self.enabled or len(audio) < 512:
            return None

        try:
            return self._hps_chord(audio, sample_rate)
        except Exception:
            return None

    def _hps_chord(self, audio: np.ndarray, sample_rate: int) -> Optional[ChordEvent]:
        n      = len(audio)
        window = np.hanning(n)
        spec   = np.abs(np.fft.rfft(audio * window))
        freqs  = np.fft.rfftfreq(n, d=1.0 / sample_rate)

        # ── Harmonic Product Spectrum: find fundamental ────────────────────
        hps = spec.copy()
        for h in range(2, 6):
            downsampled = spec[::h][:len(hps)]
            hps[:len(downsampled)] *= downsampled

        # Restrict to voice range 80–1200 Hz
        lo = int(80  * n / sample_rate)
        hi = int(1200 * n / sample_rate)
        hps[:lo] = 0
        if hi < len(hps):
            hps[hi:] = 0

        if hps.max() == 0:
            return None

        # Top-N spectral peaks as candidate partials
        peak_indices = self._find_peaks(spec, lo, hi, n_peaks=8)
        if len(peak_indices) < 2:
            return None

        peak_freqs = [freqs[i] for i in peak_indices if freqs[i] > 0]
        peak_midi  = [int(round(69 + 12 * np.log2(f / 440.0)))
                      for f in peak_freqs if 30 < 69 + 12 * np.log2(max(f, 1) / 440.0) < 100]

        if len(peak_midi) < 2:
            return None

        # ── Match peaks to chord templates ────────────────────────────────
        best_score = 0.0
        best_root  = peak_midi[0]
        best_type  = "Major"

        for root in peak_midi:
            for chord_type, intervals in CHORD_TEMPLATES.items():
                target_notes = {(root + i) % 12 for i in intervals}
                detected_classes = {m % 12 for m in peak_midi}
                overlap = len(target_notes & detected_classes)
                score   = overlap / max(len(target_notes), len(detected_classes))
                if score > best_score:
                    best_score = score
                    best_root  = root
                    best_type  = chord_type

        if best_score < self.confidence_thresh:
            return None

        chord_notes = [
            max(0, min(127, best_root + i))
            for i in CHORD_TEMPLATES[best_type]
        ]

        evt = ChordEvent(
            root_note  = best_root,
            chord_type = best_type,
            notes      = chord_notes,
            confidence = best_score,
            timestamp  = time.time(),
        )
        self._last_chord = evt
        self._chord_history.append(evt)
        if len(self._chord_history) > 32:
            self._chord_history.pop(0)
        return evt

    def _find_peaks(self, spec: np.ndarray, lo: int, hi: int, n_peaks: int) -> list[int]:
        """Return indices of the N largest local maxima in spec[lo:hi]."""
        sub   = spec[lo:hi]
        peaks = []
        for i in range(1, len(sub) - 1):
            if sub[i] > sub[i - 1] and sub[i] > sub[i + 1]:
                peaks.append((sub[i], lo + i))
        peaks.sort(reverse=True)
        return [idx for _, idx in peaks[:n_peaks]]

    @property
    def last_chord(self) -> Optional[ChordEvent]:
        return self._last_chord

    @property
    def chord_history(self) -> list[ChordEvent]:
        return list(self._chord_history)
