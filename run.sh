#!/bin/bash
# Voice to MIDI - Launcher Script
# Run this to start the application

echo "🎵 Voice to MIDI Converter"
echo "=========================="

# Check if Python 3.8+ is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python $PYTHON_VERSION found"

# Create venv if it doesn't exist
VENV_DIR="./venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✅ Virtual environment created"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Run the dependency checker + app
echo "🔍 Checking dependencies..."
python3 app.py

deactivate
