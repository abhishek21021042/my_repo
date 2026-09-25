@echo off
cd /d "%~dp0"
title Amazon ML Hackathon - Distributed Part Runner
color 0B
cls

REM 1. Find Python executable
set "PY_CMD="
if exist ".venv\Scripts\python.exe" set "PY_CMD=.venv\Scripts\python.exe"
if not defined PY_CMD if exist "venv\Scripts\python.exe" set "PY_CMD=venv\Scripts\python.exe"

if not defined PY_CMD (
    echo [INFO] Virtual environment not found. Checking system Python...
    where python >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python is not installed or not in PATH!
        pause
        exit /b 1
    )
    echo System Python found. Creating virtual environment...
    python -m venv .venv
    echo Installing requirements...
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    set "PY_CMD=.venv\Scripts\python.exe"
)

echo ===============================================================================
echo                AMAZON ML HACKATHON - 1-CLICK PART RUNNER
echo ===============================================================================
echo.
echo Total Test Entities to Process: 1,732,544 rows
echo Output format: Ground Truth TSV (source1_entity_id \t matched_entity_ids)
echo.
echo Please choose which part this laptop should run:
echo.
echo   [1] Run PART 1 of 4  (Records 1 to 433,136)
echo   [2] Run PART 2 of 4  (Records 433,137 to 866,272)
echo   [3] Run PART 3 of 4  (Records 866,273 to 1,299,408)
echo   [4] Run PART 4 of 4  (Records 1,299,409 to 1,732,544)
echo.
echo   [G] Open Graphical Desktop Window (GUI)
echo   [M] MERGE all finished parts into final matching_results.tsv
echo   [Q] Quit
echo.
echo ===============================================================================
set /p user_choice="Enter your choice (1, 2, 3, 4, G, M, Q): "

if "%user_choice%"=="1" goto run_p1
if "%user_choice%"=="2" goto run_p2
if "%user_choice%"=="3" goto run_p3
if "%user_choice%"=="4" goto run_p4
if /i "%user_choice%"=="G" goto run_gui
if /i "%user_choice%"=="M" goto run_merge
if /i "%user_choice%"=="Q" exit /b 0

echo Invalid choice.
pause
exit /b 1

:run_p1
echo.
echo Starting Part 1 of 4 across 8 Cores...
"%PY_CMD%" scripts\run_part.py --part 1 --total_parts 4 --workers 8
pause
exit /b 0

:run_p2
echo.
echo Starting Part 2 of 4 across 8 Cores...
"%PY_CMD%" scripts\run_part.py --part 2 --total_parts 4 --workers 8
pause
exit /b 0

:run_p3
echo.
echo Starting Part 3 of 4 across 8 Cores...
"%PY_CMD%" scripts\run_part.py --part 3 --total_parts 4 --workers 8
pause
exit /b 0

:run_p4
echo.
echo Starting Part 4 of 4 across 8 Cores...
"%PY_CMD%" scripts\run_part.py --part 4 --total_parts 4 --workers 8
pause
exit /b 0

:run_gui
echo.
echo Launching GUI...
start "" "%PY_CMD%" scripts\gui_launcher.py
exit /b 0

:run_merge
echo.
echo Merging all parts...
"%PY_CMD%" scripts\merge_parts.py
pause
exit /b 0
