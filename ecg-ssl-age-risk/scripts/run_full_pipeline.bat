@echo off
REM Full End-to-End Training Pipeline
REM Estimated Total Time: 15-23 hours

echo ========================================
echo ECG SSL Age-Risk: Full Training Pipeline
echo ========================================
echo.
echo This will run:
echo 1. SSL Pre-training (50 epochs, ~3-5 hours)
echo 2. Baseline Training (50 epochs, ~2-3 hours)
echo 3. Multi-Task Training (30 epochs, ~2-3 hours)
echo 4. Ablation Study E1-E4 (~8-12 hours)
echo.
echo Total Estimated Time: 15-23 hours
echo ========================================
echo.

REM Step 1: SSL Pre-training
echo [STEP 1/4] Starting SSL Pre-training...
echo Start Time: %date% %time%
python scripts/train_ssl.py --max_epochs 50 --masking_strategy random
if %errorlevel% neq 0 (
    echo ERROR: SSL Pre-training failed!
    exit /b 1
)
echo SSL Pre-training completed at %date% %time%
echo.

REM Step 2: Baseline Training
echo [STEP 2/4] Starting Baseline Training...
echo Start Time: %date% %time%
python scripts/train_baseline.py --max_epochs 50
if %errorlevel% neq 0 (
    echo ERROR: Baseline training failed!
    exit /b 1
)
echo Baseline training completed at %date% %time%
echo.

REM Step 3: Multi-Task Training with SSL weights
echo [STEP 3/4] Starting Multi-Task Training (with SSL weights)...
echo Start Time: %date% %time%
REM Find the best SSL checkpoint
for /f "delims=" %%i in ('dir /b /o-d experiments\checkpoints\ssl\*.ckpt 2^>nul ^| findstr /v "last"') do (
    set SSL_CKPT=experiments\checkpoints\ssl\%%i
    goto :found_ssl
)
:found_ssl
if defined SSL_CKPT (
    echo Using SSL checkpoint: %SSL_CKPT%
    python scripts/train_multitask.py --max_epochs 30 --ssl_ckpt %SSL_CKPT%
) else (
    echo WARNING: No SSL checkpoint found, training from scratch
    python scripts/train_multitask.py --max_epochs 30
)
if %errorlevel% neq 0 (
    echo ERROR: Multi-Task training failed!
    exit /b 1
)
echo Multi-Task training completed at %date% %time%
echo.

REM Step 4: Ablation Study
echo [STEP 4/4] Starting Ablation Study...
echo Start Time: %date% %time%

echo Running E1 (Baseline) with 100%% data...
python scripts/run_ablation.py --experiment e1 --data_fraction 1.0

echo Running E1 (Baseline) with 25%% data...
python scripts/run_ablation.py --experiment e1 --data_fraction 0.25

echo Running E1 (Baseline) with 10%% data...
python scripts/run_ablation.py --experiment e1 --data_fraction 0.1

echo Running E3 (Multi-Task) with 100%% data...
python scripts/run_ablation.py --experiment e3 --data_fraction 1.0

echo Running E3 (Multi-Task) with 25%% data...
python scripts/run_ablation.py --experiment e3 --data_fraction 0.25

echo Running E3 (Multi-Task) with 10%% data...
python scripts/run_ablation.py --experiment e3 --data_fraction 0.1

echo Ablation Study completed at %date% %time%
echo.

REM Step 5: Analyze Results
echo ========================================
echo Analyzing Results...
echo ========================================
python scripts/analyze_results.py

echo.
echo ========================================
echo FULL PIPELINE COMPLETED!
echo End Time: %date% %time%
echo ========================================
echo.
echo Results saved to ablation_results.csv
echo Checkpoints saved in experiments/checkpoints/
echo.
pause
