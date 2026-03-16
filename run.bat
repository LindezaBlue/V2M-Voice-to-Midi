@echo off
echo Voice to MIDI Converter
echo ==========================

:: Suppress TensorFlow / oneDNN noise
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0
set ABSL_MIN_LOG_LEVEL=3
set TF_FORCE_GPU_ALLOW_GROWTH=true

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)
echo Python found

:: Create venv if needed
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created
)

:: Activate and run
call venv\Scripts\activate.bat
echo Checking dependencies...
python app.py

pause
