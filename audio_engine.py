"""
audio_engine.py
Real-time audio capture, pitch detection, and MIDI note generation.
Uses CREPE (neural pitch detector) with GPU acceleration when available.
"""

import numpy as np
import threading
import queue
import time
from dataclasses import dataclass, field
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000   # Hz — CREPE's native rate
BLOCK_SIZE    = 512     # samples per sounddevice callback (~32ms)
PITCH_WINDOW  = 1024    # samples fed to pitch detector (~64ms)
SILENCE_RMS   = 0.01    # RMS threshold below which we consider silence
NOTE_HOLD_SEC = 0.08    # seconds a note must be stable before emitting
CONF_THRESH   = 0.65    # CREPE confidence threshold (0–1)

# MIDI note number → note name
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]


# ── Instrument Mode Configs ────────────────────────────────────────────────────
MODES = {
    "Vocals / Melody": {
        "midi_channel":  0,
        "program":       52,   # Choir Aahs
        "octave_shift":  0,
        "quantize":      False,
        "description":   "Faithful pitch-to-MIDI, no quantization"
    },
    "Bass": {
        "midi_channel":  1,
        "program":       33,   # Electric Bass
        "octave_shift": -2,
        "quantize":      True,
        "description":   "Drops 2 octaves, quantized to chromatic"
    },
    "Synth Lead": {
        "midi_channel":  2,
        "program":       80,   # Square Lead
        "octave_shift":  1,
        "quantize":      True,
        "description":   "1 octave up, snapped to scale"
    },
    "Keys / Piano": {
        "midi_channel":  3,
        "program":       0,    # Acoustic Grand Piano
        "octave_shift":  0,
        "quantize":      True,
        "description":   "Piano-range, quantized"
    },
    "Drums / Perc": {
        "midi_channel":  9,    # MIDI channel 10 (0-indexed as 9) = drums
        "program":       0,
        "octave_shift":  0,
        "quantize":      True,
        "description":   "Maps pitch regions to drum hits"
    },
    "Strings": {
        "midi_channel":  4,
        "program":       48,   # String Ensemble
        "octave_shift":  0,
        "quantize":      False,
        "description":   "Smooth legato strings"
    },
    "Brass": {
        "midi_channel":  5,
        "program":       61,   # Brass Section
        "octave_shift":  0,
        "quantize":      True,
        "description":   "Punchy brass hits"
    },
}

DRUM_MAP = {
    # frequency range (Hz) → GM drum note
    (0,   100): 36,   # Bass Drum
    (100, 180): 38,   # Snare
    (180, 300): 42,   # Hi-Hat Closed
    (300, 500): 46,   # Hi-Hat Open
    (500, 800): 49,   # Crash Cymbal
    (800, 9999): 51,  # Ride Cymbal
}


@dataclass
class NoteEvent:
    note:       int         # MIDI note 0–127
    frequency:  float       # Hz
    confidence: float       # 0–1
    velocity:   int         # 0–127
    timestamp:  float       # time.time()
    name:       str = ""    # e.g. "A4"
    active:     bool = True

    def __post_init__(self):
        self.name = midi_to_name(self.note)


def freq_to_midi(freq: float) -> int:
    """Convert frequency (Hz) to nearest MIDI note number."""
    if freq <= 0:
        return 0
    return int(round(69 + 12 * np.log2(freq / 440.0)))


def midi_to_freq(note: int) -> float:
    return 440.0 * (2 ** ((note - 69) / 12.0))


def midi_to_name(note: int) -> str:
    if not (0 <= note <= 127):
        return "?"
    octave = (note // 12) - 1
    name   = NOTE_NAMES[note % 12]
    return f"{name}{octave}"


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples ** 2)))


def map_drum_note(freq: float) -> int:
    for (lo, hi), note in DRUM_MAP.items():
        if lo <= freq < hi:
            return note
    return 38  # default snare


class AudioEngine:
    """
    Captures microphone audio, detects pitch with CREPE,
    and streams NoteEvent objects to registered callbacks.
    """

    def __init__(self):
        self._running    = False
        self._thread     = None
        self._audio_q    : queue.Queue = queue.Queue(maxsize=50)
        self._callbacks  : list[Callable[[NoteEvent], None]] = []
        self._mode       = "Vocals / Melody"
        self._device     = None      # None = default mic
        self._buffer     = np.zeros(PITCH_WINDOW, dtype=np.float32)

        # CREPE model (lazy-loaded)
        self._crepe_model = None
        self._device_str  = "cpu"

        # State for note de-bouncing
        self._last_note      : Optional[int]  = None
        self._note_start_t   : float          = 0.0
        self._current_event  : Optional[NoteEvent] = None

        # Latest data for UI
        self.latest_note  : Optional[NoteEvent] = None
        self.latest_rms   : float = 0.0
        self.latest_freq  : float = 0.0
        self.latest_conf  : float = 0.0
        self.note_history : list[NoteEvent] = []   # last 32 notes

        # MIDI output (optional)
        self._midi_out = None
        self._init_midi()

    # ── MIDI ──────────────────────────────────────────────────────────────────

    def _init_midi(self):
        try:
            import rtmidi
            self._midi_out = rtmidi.MidiOut()
            ports = self._midi_out.get_ports()
            if ports:
                self._midi_out.open_port(0)
                logger.info(f"MIDI out: {ports[0]}")
            else:
                self._midi_out.open_virtual_port("VoiceToMIDI")
                logger.info("MIDI out: virtual port 'VoiceToMIDI'")
        except Exception as e:
            logger.warning(f"MIDI unavailable: {e}")
            self._midi_out = None

    def _send_note_on(self, note: int, velocity: int, channel: int):
        if self._midi_out:
            try:
                self._midi_out.send_message([0x90 | channel, note, velocity])
            except Exception:
                pass

    def _send_note_off(self, note: int, channel: int):
        if self._midi_out:
            try:
                self._midi_out.send_message([0x80 | channel, note, 0])
            except Exception:
                pass

    # ── CREPE setup ───────────────────────────────────────────────────────────

    def load_model(self):
        """Load CREPE model onto GPU if available."""
        try:
            import torch
            if torch.cuda.is_available():
                self._device_str = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device_str = "mps"
            logger.info(f"Using device: {self._device_str}")
        except ImportError:
            pass

        try:
            import crepe
            # Force CREPE to load its TF model
            # Warmup with a silent buffer
            dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
            crepe.predict(dummy, SAMPLE_RATE, viterbi=True, verbose=0)
            self._crepe_model = crepe
            logger.info("CREPE model loaded ✅")
        except Exception as e:
            logger.error(f"CREPE load failed: {e}")

    # ── Audio callback ─────────────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        mono = indata[:, 0].astype(np.float32) if indata.ndim > 1 else indata.astype(np.float32)
        try:
            self._audio_q.put_nowait(mono)
        except queue.Full:
            pass   # drop oldest if overwhelmed

    # ── Processing loop ───────────────────────────────────────────────────────

    def _process_loop(self):
        # Rolling buffer
        buf = np.zeros(PITCH_WINDOW, dtype=np.float32)

        while self._running:
            try:
                chunk = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Update rolling buffer
            buf = np.roll(buf, -len(chunk))
            buf[-len(chunk):] = chunk

            r = rms(buf)
            self.latest_rms = r

            if r < SILENCE_RMS:
                # Silence — send note off if we had an active note
                if self._current_event:
                    mode_cfg = MODES[self._mode]
                    self._send_note_off(self._current_event.note, mode_cfg["midi_channel"])
                    evt = NoteEvent(
                        note=self._current_event.note,
                        frequency=self._current_event.frequency,
                        confidence=self._current_event.confidence,
                        velocity=0,
                        timestamp=time.time(),
                        active=False
                    )
                    self._fire_callbacks(evt)
                    self._current_event = None
                    self._last_note = None
                continue

            # Pitch detection
            freq, conf = self._detect_pitch(buf)
            self.latest_freq = freq
            self.latest_conf = conf

            if freq <= 0 or conf < CONF_THRESH:
                continue

            mode_cfg = MODES[self._mode]
            raw_midi = freq_to_midi(freq)

            # Apply octave shift
            midi_note = max(0, min(127, raw_midi + mode_cfg["octave_shift"] * 12))

            # Drum mode: remap by frequency
            if self._mode == "Drums / Perc":
                midi_note = map_drum_note(freq)

            velocity = int(min(127, max(30, r * 800)))

            # De-bounce: note must be stable for NOTE_HOLD_SEC
            now = time.time()
            if midi_note != self._last_note:
                self._last_note    = midi_note
                self._note_start_t = now
                continue

            if (now - self._note_start_t) < NOTE_HOLD_SEC:
                continue

            # Emit note if it changed
            if self._current_event is None or self._current_event.note != midi_note:
                # Send note off for previous
                if self._current_event:
                    self._send_note_off(self._current_event.note, mode_cfg["midi_channel"])

                evt = NoteEvent(
                    note=midi_note,
                    frequency=freq,
                    confidence=conf,
                    velocity=velocity,
                    timestamp=now,
                    active=True
                )
                self._current_event = evt
                self.latest_note    = evt
                self.note_history.append(evt)
                if len(self.note_history) > 64:
                    self.note_history.pop(0)

                self._send_note_on(midi_note, velocity, mode_cfg["midi_channel"])
                self._fire_callbacks(evt)

    def _detect_pitch(self, audio: np.ndarray) -> tuple[float, float]:
        """Returns (frequency_hz, confidence)."""
        if self._crepe_model is None:
            # Fallback: autocorrelation-based (fast but less accurate)
            return self._autocorr_pitch(audio)
        try:
            times, freqs, confs, _ = self._crepe_model.predict(
                audio, SAMPLE_RATE,
                viterbi=True,
                verbose=0,
                step_size=10,
                model_capacity="small"
            )
            if len(freqs) == 0:
                return 0.0, 0.0
            # Take the most recent / highest confidence frame
            best_idx = int(np.argmax(confs))
            return float(freqs[best_idx]), float(confs[best_idx])
        except Exception as e:
            logger.debug(f"CREPE error: {e}")
            return self._autocorr_pitch(audio)

    def _autocorr_pitch(self, audio: np.ndarray) -> tuple[float, float]:
        """Simple autocorrelation pitch detector (CPU fallback)."""
        n = len(audio)
        windowed = audio * np.hanning(n)
        corr = np.correlate(windowed, windowed, mode='full')
        corr = corr[n-1:]

        # Find first minimum then first maximum after it
        d = np.diff(corr)
        # Find zero-crossings of derivative (peaks in autocorr)
        start = 1
        while start < len(d) and d[start] > 0:
            start += 1
        while start < len(d) and d[start] < 0:
            start += 1

        if start >= len(d) - 1:
            return 0.0, 0.0

        peak_idx = start + int(np.argmax(corr[start:]))
        if peak_idx == 0:
            return 0.0, 0.0

        freq = SAMPLE_RATE / peak_idx
        # Confidence = normalized autocorr peak
        conf = corr[peak_idx] / (corr[0] + 1e-9)
        conf = float(np.clip(conf, 0, 1))

        if not (50 <= freq <= 2000):
            return 0.0, 0.0

        return float(freq), conf * 0.8   # scale down (less reliable than CREPE)

    # ── Public API ────────────────────────────────────────────────────────────

    def register_callback(self, fn: Callable[[NoteEvent], None]):
        self._callbacks.append(fn)

    def _fire_callbacks(self, evt: NoteEvent):
        for fn in self._callbacks:
            try:
                fn(evt)
            except Exception as e:
                logger.debug(f"Callback error: {e}")

    def set_mode(self, mode: str):
        if mode in MODES:
            # Send all-notes-off on old channel before switching
            if self._current_event and self._midi_out:
                old_ch = MODES[self._mode]["midi_channel"]
                self._send_note_off(self._current_event.note, old_ch)
                self._current_event = None
            self._mode = mode

    def start(self):
        if self._running:
            return
        import sounddevice as sd

        self._running = True
        self._thread  = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

        self._stream = sd.InputStream(
            samplerate    = SAMPLE_RATE,
            blocksize     = BLOCK_SIZE,
            dtype         = "float32",
            channels      = 1,
            callback      = self._audio_callback,
            device        = self._device,
        )
        self._stream.start()
        logger.info("Audio engine started ▶️")

    def stop(self):
        self._running = False
        if hasattr(self, "_stream"):
            self._stream.stop()
            self._stream.close()
        if self._thread:
            self._thread.join(timeout=2)
        if self._current_event and self._midi_out:
            ch = MODES[self._mode]["midi_channel"]
            self._send_note_off(self._current_event.note, ch)
        logger.info("Audio engine stopped ⏹️")

    def list_input_devices(self) -> list[tuple[int, str]]:
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            return [
                (i, d["name"])
                for i, d in enumerate(devices)
                if d["max_input_channels"] > 0
            ]
        except Exception:
            return []

    def set_input_device(self, device_index: int):
        was_running = self._running
        if was_running:
            self.stop()
        self._device = device_index
        if was_running:
            self.start()

    @property
    def is_running(self) -> bool:
        return self._running
