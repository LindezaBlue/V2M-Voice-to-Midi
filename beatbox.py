"""
beatbox.py
Beatbox sound classifier for the Drums / Perc mode.

Instead of mapping pitch frequency to drum notes (useless for percussion),
this module analyses the spectral and temporal shape of each audio burst
to classify it as one of the common beatbox sounds:

  Kick  — "buh" / "boom" / "puh" — low-frequency thump, fast attack
  Snare — "psh" / "kuh"          — broadband burst, strong mid/high content
  Hi-Hat closed — "ts" / "t"     — very high frequency, very fast decay
  Hi-Hat open   — "tss"          — high freq, longer decay
  Clap  — "pap" / "clap"         — broadband, strong 1–4 kHz region
  Tom   — "duh" / "boh"          — mid-frequency thump, slower than kick

Each sound maps to a standard GM drum note on MIDI channel 10 (ch 9).
"""

import numpy as np
from typing import Optional

# GM drum note numbers
GM = {
    "kick":        36,   # Bass Drum 1
    "snare":       38,   # Acoustic Snare
    "clap":        39,   # Hand Clap
    "hihat_closed":42,   # Closed Hi-Hat
    "hihat_open":  46,   # Open Hi-Hat
    "tom_mid":     45,   # Low Mid Tom
    "tom_low":     41,   # Low Floor Tom
    "crash":       49,   # Crash Cymbal
    "ride":        51,   # Ride Cymbal
}

SAMPLE_RATE = 16000


def _spectral_features(audio: np.ndarray, sr: int) -> dict:
    """
    Extract a compact set of spectral + temporal features from a short buffer.
    All features are normalised to [0, 1] where possible.
    """
    n      = len(audio)
    window = np.hanning(n)
    spec   = np.abs(np.fft.rfft(audio * window))
    freqs  = np.fft.rfftfreq(n, d=1.0 / sr)
    energy = np.sum(spec ** 2) + 1e-12

    def band_energy(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sum(spec[mask] ** 2)) / energy

    # Energy in key frequency bands
    sub_bass  = band_energy(20,   100)    # kick fundamental
    bass      = band_energy(100,  300)    # kick body / tom
    low_mid   = band_energy(300,  800)    # tom / snare body
    mid       = band_energy(800,  2500)   # snare crack / clap
    high_mid  = band_energy(2500, 5000)   # clap / hihat body
    high      = band_energy(5000, 8000)   # hi-hat

    # Spectral centroid (normalised to 0–1 over 0–8kHz)
    centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12))
    centroid_norm = min(1.0, centroid / 8000.0)

    # Spectral flatness (0 = tonal, 1 = noise-like)
    log_mean = np.mean(np.log(spec + 1e-12))
    mean_log = np.log(np.mean(spec + 1e-12))
    flatness  = float(np.clip(np.exp(log_mean - mean_log), 0, 1))

    # Attack speed: compare first-quarter vs second-quarter RMS
    q = max(1, n // 4)
    rms_q1 = float(np.sqrt(np.mean(audio[:q] ** 2) + 1e-12))
    rms_q2 = float(np.sqrt(np.mean(audio[q:2*q] ** 2) + 1e-12))
    attack_ratio = min(2.0, rms_q1 / (rms_q2 + 1e-12))  # >1 = fast attack

    # Decay speed: compare second half vs first half
    half = n // 2
    rms_first  = float(np.sqrt(np.mean(audio[:half] ** 2) + 1e-12))
    rms_second = float(np.sqrt(np.mean(audio[half:] ** 2) + 1e-12))
    decay_ratio = min(1.0, rms_second / (rms_first + 1e-12))  # low = fast decay

    return {
        "sub_bass":     sub_bass,
        "bass":         bass,
        "low_mid":      low_mid,
        "mid":          mid,
        "high_mid":     high_mid,
        "high":         high,
        "centroid":     centroid_norm,
        "flatness":     flatness,
        "attack_ratio": attack_ratio,
        "decay_ratio":  decay_ratio,
    }


def classify_beatbox(audio: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[str, int, float]:
    """
    Classify a short audio burst into a beatbox sound type.

    Returns (label, midi_note, confidence) where:
      label      — one of: kick, snare, hihat_closed, hihat_open, clap, tom_mid, tom_low
      midi_note  — GM drum MIDI note number
      confidence — 0.0–1.0 score for the winning class
    """
    if len(audio) < 64:
        return "snare", GM["snare"], 0.0

    f = _spectral_features(audio, sr)

    # ── Rule-based scoring ───────────────────────────────────────────────────
    # Each sound type accumulates a score based on which features match.
    # This is essentially a hand-tuned nearest-centroid classifier.

    scores = {}

    # KICK — low centroid, strong sub+bass, fast attack, not flat
    scores["kick"] = (
        (f["sub_bass"] * 3.0) +
        (f["bass"]     * 2.0) +
        ((1.0 - f["centroid"]) * 2.0) +
        (f["attack_ratio"] * 0.5) +
        ((1.0 - f["flatness"]) * 0.5)
    )

    # SNARE — broadband energy, moderate centroid, noisy, fast attack
    scores["snare"] = (
        (f["low_mid"]  * 1.5) +
        (f["mid"]      * 1.5) +
        (f["flatness"] * 2.0) +
        (f["attack_ratio"] * 1.0) +
        ((1.0 - f["sub_bass"]) * 0.5)
    )

    # CLAP — strong mid + high_mid, broadband, very fast attack
    scores["clap"] = (
        (f["mid"]       * 2.0) +
        (f["high_mid"]  * 2.0) +
        (f["flatness"]  * 1.5) +
        (f["attack_ratio"] * 1.0)
    )

    # HI-HAT CLOSED — high freq dominant, fast decay, very fast attack
    scores["hihat_closed"] = (
        (f["high"]      * 3.5) +
        (f["high_mid"]  * 1.5) +
        (f["centroid"]  * 2.0) +
        ((1.0 - f["decay_ratio"]) * 2.0) +
        (f["attack_ratio"] * 0.5)
    )

    # HI-HAT OPEN — high freq, but slower decay than closed
    scores["hihat_open"] = (
        (f["high"]      * 3.0) +
        (f["high_mid"]  * 1.5) +
        (f["centroid"]  * 2.0) +
        (f["decay_ratio"] * 1.5)    # longer decay = open hat
    )

    # TOM MID — mid-frequency thump, tonal (not flat), moderate attack
    scores["tom_mid"] = (
        (f["low_mid"]  * 2.5) +
        (f["bass"]     * 1.5) +
        ((1.0 - f["flatness"]) * 1.5) +
        ((1.0 - f["centroid"]) * 1.0)
    )

    # TOM LOW — like kick but slower attack, more mid content
    scores["tom_low"] = (
        (f["bass"]     * 2.0) +
        (f["sub_bass"] * 1.5) +
        (f["low_mid"]  * 1.0) +
        ((1.0 - f["attack_ratio"]) * 1.5)
    )

    # Pick winner
    best_label = max(scores, key=lambda k: scores[k])
    best_score = scores[best_label]
    total      = sum(scores.values()) + 1e-9
    confidence = float(best_score / total)

    midi_note = GM.get(best_label, GM["snare"])
    return best_label, midi_note, confidence


# ── Onset detector ────────────────────────────────────────────────────────────
# Beatbox sounds are percussive bursts — we need to detect the *onset*
# (start of a hit) rather than tracking a sustained pitch.

class BeatboxOnsetDetector:
    """
    Detects percussive onsets in a rolling audio stream and classifies
    each detected hit using the spectral classifier above.

    Call feed(chunk) with each new audio block.
    When an onset is detected and classified, latest_hit is updated.
    """

    ONSET_WINDOW   = 512    # samples for onset detection (~32ms @ 16kHz)
    CAPTURE_WINDOW = 2048   # samples captured after onset for classification (~128ms)
    ONSET_THRESH   = 2.8    # how much louder than background to trigger
    COOLDOWN_SECS  = 0.08   # min time between hits

    def __init__(self):
        self._buf          : list[float] = []
        self._bg_rms       : float       = 0.005
        self._in_hit       : bool        = False
        self._hit_buf      : list[float] = []
        self._last_onset_t : float       = 0.0

        self.latest_hit    : Optional[tuple[str, int, float]] = None  # (label, note, conf)
        self.hit_history   : list[dict] = []

    def feed(self, chunk: np.ndarray, timestamp: float) -> Optional[tuple[str, int, float]]:
        """
        Feed a new audio chunk. Returns (label, midi_note, confidence) if a
        hit was detected and classified, otherwise None.
        """
        import time as _time
        samples = chunk.tolist()
        self._buf.extend(samples)

        # Process in windows
        result = None
        while len(self._buf) >= self.ONSET_WINDOW:
            window   = np.array(self._buf[:self.ONSET_WINDOW], dtype=np.float32)
            self._buf = self._buf[self.ONSET_WINDOW // 2:]  # 50% overlap

            current_rms = float(np.sqrt(np.mean(window ** 2)))

            # Update background level slowly
            if current_rms < self._bg_rms * 1.5:
                self._bg_rms = 0.95 * self._bg_rms + 0.05 * current_rms
            self._bg_rms = max(0.001, self._bg_rms)

            # Onset: current RMS significantly exceeds background
            cooldown_ok = (timestamp - self._last_onset_t) >= self.COOLDOWN_SECS
            is_onset    = (current_rms > self._bg_rms * self.ONSET_THRESH) and cooldown_ok

            if is_onset and not self._in_hit:
                self._in_hit    = True
                self._hit_buf   = list(window)
                self._last_onset_t = timestamp

            elif self._in_hit:
                self._hit_buf.extend(window.tolist())
                if len(self._hit_buf) >= self.CAPTURE_WINDOW:
                    # Enough captured — classify
                    hit_audio = np.array(self._hit_buf[:self.CAPTURE_WINDOW], dtype=np.float32)
                    label, note, conf = classify_beatbox(hit_audio, SAMPLE_RATE)
                    self.latest_hit = (label, note, conf)
                    self.hit_history.append({
                        "label": label, "note": note,
                        "conf": conf, "timestamp": timestamp
                    })
                    if len(self.hit_history) > 32:
                        self.hit_history.pop(0)
                    self._in_hit  = False
                    self._hit_buf = []
                    result = (label, note, conf)

        return result
