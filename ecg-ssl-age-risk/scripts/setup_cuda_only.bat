@echo off
setlocal

echo ==========================================
echo    ECG SSL CUDA Setup (Fix & Retry)
echo ==========================================

echo [INFO] Ensuring build tools are up to date...
python -m pip install --upgrade pip setuptools wheel
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to upgrade pip/setuptools.
    exit /b 1
)

echo [INFO] Installing NumPy explicitly (binary only)...
REM Force binary install to avoid compilation errors
pip install "numpy>=1.26.0" --only-binary=:all:
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Binary install failed. Trying again with --prefer-binary...
    pip install "numpy>=1.26.0" --prefer-binary
)

echo [INFO] Installing PyTorch with CUDA 11.8 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install PyTorch.
    exit /b 1
)

echo [INFO] Installing remaining dependencies...
REM Installing dependencies one by one to isolate issues if any
for %%i in (pandas scipy scikit-learn wfdb matplotlib seaborn captum pyyaml tqdm pytorch-lightning mlflow) do (
    echo Installing %%i...
    pip install %%i --only-binary=:all:
)

echo [INFO] Verifying GPU setup...
python scripts/verify_gpu.py

echo ==========================================
echo       CUDA Setup Complete!
echo ==========================================
endlocal
pause
