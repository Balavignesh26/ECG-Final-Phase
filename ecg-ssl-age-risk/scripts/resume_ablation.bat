@echo off
REM Complete Remaining Ablation Experiments
REM Runs: E1 (25%, 10%) and E3 (100%, 25%, 10%)

echo ========================================
echo Resuming Ablation Study
echo ========================================
echo.
echo Running 5 remaining experiments:
echo   [1/5] E1 with 25%% data
echo   [2/5] E1 with 10%% data
echo   [3/5] E3 with 100%% data (50 epochs)
echo   [4/5] E3 with 25%% data
echo   [5/5] E3 with 10%% data
echo.
echo Estimated Duration: 10-12 hours
echo ========================================
echo.

REM E1 - Baseline with reduced data
echo [1/5] Running E1 with 25%% data...
python scripts\run_ablation.py --experiment e1 --data_fraction 0.25
if %errorlevel% neq 0 (
    echo ERROR: E1 25%% failed!
    exit /b %errorlevel%
)

echo [2/5] Running E1 with 10%% data...
python scripts\run_ablation.py --experiment e1 --data_fraction 0.1
if %errorlevel% neq 0 (
    echo ERROR: E1 10%% failed!
    exit /b %errorlevel%
)

REM E3 - Multi-Task with all data fractions
echo [3/5] Running E3 with 100%% data (50 epochs)...
python scripts\run_ablation.py --experiment e3 --data_fraction 1.0
if %errorlevel% neq 0 (
    echo ERROR: E3 100%% failed!
    exit /b %errorlevel%
)

echo [4/5] Running E3 with 25%% data...
python scripts\run_ablation.py --experiment e3 --data_fraction 0.25
if %errorlevel% neq 0 (
    echo ERROR: E3 25%% failed!
    exit /b %errorlevel%
)

echo [5/5] Running E3 with 10%% data...
python scripts\run_ablation.py --experiment e3 --data_fraction 0.1
if %errorlevel% neq 0 (
    echo ERROR: E3 10%% failed!
    exit /b %errorlevel%
)

echo.
echo ========================================
echo All experiments completed successfully!
echo ========================================
echo.
echo Running results analysis...
python scripts\analyze_results.py

echo.
echo Done! Check ablation_results.csv for complete comparison.
