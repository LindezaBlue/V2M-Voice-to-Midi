"""
pitch_correction.py
Auto-Tune style pitch correction for Voice to MIDI.

Two modes:
  • "Chromatic"  – snaps to the nearest semitone (classic hard-tune)
  • Scale-aware  – snaps only to notes in a chosen key/scale

The "correction strength" (0–1) controls how aggressively the raw
frequency is pulled toward the target:
  corrected = raw * (1 - strength) + target * strength
At 1.0 you get hard-tune; at 0.0 the signal is untouched.
"""

import numpy as np
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Scale definitions  (semitone intervals from root)
# ─────────────────────────────────────────────────────────────────────────────
SCALES = {
    "Chromatic":        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
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
    "Locrian":          [0, 1, 3, 5, 6, 8, 10],
    "Whole Tone":       [0, 2, 4, 6, 8, 10],
    "Diminished":       [0, 2, 3, 5, 6, 8, 9, 11],
}

ROOT_NOTES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]


def _midi_to_freq(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def _freq_to_midi(freq: float) -> float:
    if freq <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def build_scale_midi_set(root: str, scale: str) -> list[int]:
    """
    Return every MIDI note (0–127) that belongs to the given key + scale.
    """
    root_idx   = ROOT_NOTES.index(root)
    intervals  = SCALES.get(scale, SCALES["Chromatic"])
    notes = set()
    for octave in range(11):
        for interval in intervals:
            n = octave * 12 + root_idx + interval
            if 0 <= n <= 127:
                notes.add(n)
    return sorted(notes)


class PitchCorrector:
    """
    Stateful pitch corrector.  Call .process(freq_hz) → corrected_freq_hz.
    """

    def __init__(self):
        self.enabled    = False
        self.strength   = 0.8      # 0 = off, 1 = hard snap
        self.root       = "C"
        self.scale      = "Major"
        self.speed      = 0.3      # 0 = instant, 1 = very slow (smoothing)

        self._scale_midi: list[int] = build_scale_midi_set("C", "Major")
        self._current_corrected: Optional[float] = None

    def set_scale(self, root: str, scale: str):
        self.root  = root
        self.scale = scale
        self._scale_midi = build_scale_midi_set(root, scale)
        self._current_corrected = None

    def nearest_scale_note(self, freq: float) -> float:
        """Return the frequency of the nearest in-scale MIDI note."""
        if freq <= 0:
            return freq
        raw_midi   = _freq_to_midi(freq)
        candidates = self._scale_midi
        if not candidates:
            return freq
        nearest = min(candidates, key=lambda n: abs(n - raw_midi))
        return _midi_to_freq(nearest)

    def process(self, freq: float) -> float:
        """
        Apply pitch correction to a raw frequency.
        Returns the (possibly corrected) frequency.
        """
        if not self.enabled or freq <= 0:
            return freq

        target = self.nearest_scale_note(freq)

        # Blend raw → target by strength
        blended = freq * (1.0 - self.strength) + target * self.strength

        # Smooth over time (low-pass on MIDI cents) to avoid zipper noise
        if self._current_corrected is None:
            self._current_corrected = blended
        else:
            alpha = 1.0 - self.speed        # lower speed = faster correction
            self._current_corrected = (
                alpha * blended
                + (1.0 - alpha) * self._current_corrected
            )

        return float(self._current_corrected)

    def cents_deviation(self, raw_freq: float) -> float:
        """
        How many cents is the raw frequency from the nearest scale note?
        Useful for the UI tuning needle.
        """
        if raw_freq <= 0:
            return 0.0
        target_freq = self.nearest_scale_note(raw_freq)
        if target_freq <= 0:
            return 0.0
        cents = 1200.0 * np.log2(raw_freq / target_freq)
        return float(np.clip(cents, -100, 100))

    @property
    def scale_note_names(self) -> list[str]:
        """Human-readable list of notes in the current scale."""
        root_idx  = ROOT_NOTES.index(self.root)
        intervals = SCALES.get(self.scale, [])
        return [ROOT_NOTES[(root_idx + i) % 12] for i in intervals]
