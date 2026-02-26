@echo off
echo ============================================
echo ECG Training Progress Monitor
echo ============================================
echo.

:loop
cls
echo ============================================
echo ECG Training Progress Monitor
echo Updated: %date% %time%
echo ============================================
echo.

echo [1] Checking Running Python Processes...
echo.
tasklist /FI "IMAGENAME eq python.exe" /FO TABLE
echo.

echo [2] Latest Checkpoints...
echo.
echo SSL Checkpoints:
dir /B /O-D "experiments\checkpoints\ssl\*.ckpt" 2>nul | findstr /N "^"
echo.
echo Baseline Checkpoints:
dir /B /O-D "experiments\checkpoints\baseline\*.ckpt" 2>nul | findstr /N "^"
echo.
echo Multi-Task Checkpoints:
dir /B /O-D "experiments\checkpoints\multitask\*.ckpt" 2>nul | findstr /N "^"
echo.

echo [3] GPU Status...
echo.
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,memory.used,memory.total --format=csv
echo.

echo ============================================
echo Press Ctrl+C to stop monitoring
echo Refreshing in 30 seconds...
echo ============================================
timeout /t 30 /nobreak >nul
goto loop
