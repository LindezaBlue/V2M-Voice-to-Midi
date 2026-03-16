# 🎙️ V2M - Voice to MIDI

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC--BY--NC--SA%204.0-orange.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Platform: Windows](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](https://www.microsoft.com/windows)
[![Platform: Linux](https://img.shields.io/badge/platform-Linux-brightgreen.svg)](https://www.kernel.org/)

Convert your voice into MIDI notes in real-time — export to GarageBand, Logic, Ableton, and more.

![preview](https://github.com/user-attachments/assets/e065df11-ea03-4d47-ba8a-88d9147323e6)

---
> [!IMPORTANT] 
> ## Requirements
>
> - **Python 3.8 or higher**  
>  Download from https://www.python.org/downloads/
> - **A microphone** (built-in or USB)
> - **Optional:** NVIDIA GPU with CUDA for faster pitch detection

---
> [!NOTE]
> Getting reliable wireless or wired communication working smoothly between iOS devices and Linux or Windows machines turned out to be a lot more complicated and time-consuming than I expected.
> To keep development manageable and make sure the program actually works consistently across different operating systems and platforms, I’ve decided to focus the app on recording audio locally on the device and saving the result straight to MIDI files only.
> This approach avoids all the cross-platform communication headaches and lets me get a solid, usable version out much faster.


---

## Quick Start

### macOS / Linux
```bash
# 1. Open Terminal in the voice_to_midi folder
chmod +x run.sh
./run.sh
```

### Windows
```
Double-click run.bat
```

The launcher will:
1. Create a Python virtual environment (`venv/`)
2. Auto-install all dependencies (takes ~2 min on first run)
3. Open the app at **http://127.0.0.1:7860** in your browser

---

## Features

### 🎛 Instrument Modes
| Mode | MIDI Channel | Description |
|---|---|---|
| Vocals / Melody | 1 | Direct pitch-to-MIDI, no correction |
| Bass | 2 | 2 octaves down, chromatic snap |
| Synth Lead | 3 | 1 octave up, quantized |
| Keys / Piano | 4 | Piano range, quantized |
| Drums / Perc | 10 | Maps pitch regions to drum hits |
| Strings | 5 | Smooth legato |
| Brass | 6 | Punchy hits |

### 🎹 Real-Time Visualizer
- **Piano roll** highlights the current note as you sing
- **VU meter** shows input level
- **Note history** shows the last 16 notes with frequency, velocity, and confidence

### 💾 MIDI Export
- Click **Export to .mid** to save your session
- Import the file into GarageBand: **File → Import**
- BPM is configurable before export

### 🔌 Virtual MIDI Output
When `python-rtmidi` is installed, the app creates a virtual MIDI port called `VoiceToMIDI`. You can route this to:
- **GarageBand** → Add external MIDI track → Input: VoiceToMIDI
- **Logic Pro** → same as above
- **Ableton** → Preferences → MIDI → Enable VoiceToMIDI

---

> [!TIP] 
> For Best Results:
>
> 1. **Sing clearly** with consistent pitch — the detector needs ~80ms of stability
> 2. **Quiet environment** reduces false triggers
> 3. **Adjust Sensitivity** slider: lower for noisy rooms, higher for quiet recording
> 4. For **Bass mode**, sing low "oooh" or "mmm" sounds
> 5. For **Drums mode**, use your mouth for "boom" (kick), "pssh" (snare), "tss" (hi-hat)

---

> [!NOTE]  
> ## Troubleshooting
>
> **No sound detected**
> - Check your microphone permissions in System Settings
> - Try selecting a different input device from the dropdown
> 
> **Laggy pitch detection**
> - Use `small` model capacity (default) — `full` is more accurate but 4× slower
> - If on CPU-only, the first few notes may have higher latency while the model warms up
>
> **MIDI not reaching GarageBand**
> - Make sure `python-rtmidi` installed successfully (check terminal output)
> - On macOS, enable the IAC Driver: Audio MIDI Setup → MIDI Studio → IAC Driver → Device is online
> 
> **`portaudio` error on Linux**
>  ```bash
> sudo apt-get install portaudio19-dev
> ```
> 
> **`python-rtmidi` build fails on Linux**
> ```bash
> sudo apt-get install libasound2-dev libjack-jackd2-dev
> ```

---

## File Structure
```
voice_to_midi/
├── app.py                # Entry point
├── audio_engine.py       # Pitch detection + MIDI output
├── ui.py                 # Gradio GUI
├── midi_export.py        # .mid file exporter
├── dependency_manager.py # Auto-install system
├── run.sh                # macOS/Linux launcher
├── run.bat               # Windows launcher
└── README.md
```

---

## 🛣️ Roadmap

- [x] Pitch Correction
- [x] Waveform Visualizer
- [x] Chord Detection
- [x] Scale/Key Quantization
- [ ] Extras

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report bugs** via [GitHub Issues](https://github.com/LindezaBlue/V2M-Voice-to-Midi/issues)
2. **Submit pull requests** for new features or fixes
3. **Share feedback** and suggestions for improvement
4. **Star this repo** if you find it useful!

---

## 📄 License

This project is licensed under the  
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.  
See the [LICENSE file](https://github.com/LindezaBlue/V2M-Voice-to-Midi?tab=License-1-ov-file) for full details.

You are free to **use, modify, and share** this project, **but only for non-commercial purposes**, and you must **give proper credit** to the original author.  

### If you remix or build upon this work, you must distribute your contributions under the **same license**!

---

## 💖 Support & Credits

**Created by:** [LindezaBlue](https://github.com/LindezaBlue)


If you find this tool helpful:
- ⭐ **Star this repository**
- 🐛 **Report issues** to help improve the tool
- 📢 **Share** with others who might benefit

---
