@echo off
REM Full Ablation Study - Runs all E1 and E3 experiments
REM This will take approximately 6-8 hours total

echo ========================================
echo ECG SSL Age Risk - Full Ablation Study
echo ========================================
echo.
echo This will run 6 experiments:
echo   E1 (Baseline): 100%%, 25%%, 10%% data
echo   E3 (Multi-Task): 100%%, 25%%, 10%% data
echo.
echo Estimated Duration: 6-8 hours
echo ========================================
echo.

REM E1 - Baseline with different data fractions
echo [1/6] Running E1 with 100%% data...
python scripts\run_ablation.py --experiment e1 --data_fraction 1.0
if %errorlevel% neq 0 (
    echo ERROR: E1 100%% failed!
    exit /b %errorlevel%
)

echo [2/6] Running E1 with 25%% data...
python scripts\run_ablation.py --experiment e1 --data_fraction 0.25
if %errorlevel% neq 0 (
    echo ERROR: E1 25%% failed!
    exit /b %errorlevel%
)

echo [3/6] Running E1 with 10%% data...
python scripts\run_ablation.py --experiment e1 --data_fraction 0.1
if %errorlevel% neq 0 (
    echo ERROR: E1 10%% failed!
    exit /b %errorlevel%
)

REM E3 - Multi-Task with different data fractions
echo [4/6] Running E3 with 100%% data...
python scripts\run_ablation.py --experiment e3 --data_fraction 1.0
if %errorlevel% neq 0 (
    echo ERROR: E3 100%% failed!
    exit /b %errorlevel%
)

echo [5/6] Running E3 with 25%% data...
python scripts\run_ablation.py --experiment e3 --data_fraction 0.25
if %errorlevel% neq 0 (
    echo ERROR: E3 25%% failed!
    exit /b %errorlevel%
)

echo [6/6] Running E3 with 10%% data...
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
echo Check ablation_results.csv for final comparison!
