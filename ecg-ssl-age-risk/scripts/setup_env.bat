@echo off
setlocal EnableDelayedExpansion

echo ==========================================
echo       ECG SSL Project Setup Script
echo ==========================================

REM Check for Conda
where conda >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [INFO] Conda found. Creating environment 'ecg_ssl' with Python 3.10...
    call conda create -n ecg_ssl python=3.10 -y
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create conda environment.
        exit /b 1
    )
    echo [INFO] Activating 'ecg_ssl'...
    call conda activate ecg_ssl
    goto :install_deps
)

REM Check for Python 3.10 via 'py' launcher
py -3.10 --version >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [INFO] Python 3.10 found via py launcher. Creating venv...
    if not exist ".venv" (
        py -3.10 -m venv .venv
    )
    echo [INFO] Activating .venv...
    call .venv\Scripts\activate.bat
    goto :install_deps
)

REM Check if 'python' is 3.10 (fallback)
for /f "tokens=2 delims= " %%i in ('python --version 2^>nul') do set PYTHON_VER=%%i
if "!PYTHON_VER:~0,4!"=="3.10" (
    echo [INFO] Python 3.10 found as default 'python'. Creating venv...
    if not exist ".venv" (
        python -m venv .venv
    )
    echo [INFO] Activating .venv...
    call .venv\Scripts\activate.bat
    goto :install_deps
)

echo [ERROR] Python 3.10 or Conda not found.
echo         Please install Anaconda or Python 3.10 and run this script again.
exit /b 1

:install_deps
echo [INFO] Installing PyTorch with CUDA 11.8 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install PyTorch.
    exit /b 1
)

echo [INFO] Installing project dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install dependencies.
    exit /b 1
)

echo [INFO] Verifying GPU setup...
python scripts/verify_gpu.py

echo ==========================================
echo         Setup Complete!
echo ==========================================
endlocal
pause
