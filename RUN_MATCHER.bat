@echo off
title Amazon ML Hackathon - Distributed Part Runner
color 0B
cls

echo ===============================================================================
echo                AMAZON ML HACKATHON - 1-CLICK PART RUNNER
echo ===============================================================================
echo.
echo Total Test Entities to Process: 1,732,544 rows
echo Output format: Exactly matches Ground Truth TSV (source1_entity_id \t matched_entity_ids)
echo.
echo Please choose which part this laptop should run:
echo.
echo   [1] Run PART 1 of 4  (Records 1 to 433,136)
echo   [2] Run PART 2 of 4  (Records 433,137 to 866,272)
echo   [3] Run PART 3 of 4  (Records 866,273 to 1,299,408)
echo   [4] Run PART 4 of 4  (Records 1,299,409 to 1,732,544)
echo.
echo   [G] Open Graphical Desktop Window (GUI with Buttons)
echo   [M] MERGE all finished parts into final matching_results.tsv
echo   [Q] Quit
echo.
echo ===============================================================================
set /p user_choice="Enter your choice (1, 2, 3, 4, G, M): "

if /i "%user_choice%"=="1" (
    echo.
    echo Starting Part 1 of 4...
    .venv\Scripts\python.exe scripts\run_part.py --part 1 --total_parts 4
    pause
    exit /b
)

if /i "%user_choice%"=="2" (
    echo.
    echo Starting Part 2 of 4...
    .venv\Scripts\python.exe scripts\run_part.py --part 2 --total_parts 4
    pause
    exit /b
)

if /i "%user_choice%"=="3" (
    echo.
    echo Starting Part 3 of 4...
    .venv\Scripts\python.exe scripts\run_part.py --part 3 --total_parts 4
    pause
    exit /b
)

if /i "%user_choice%"=="4" (
    echo.
    echo Starting Part 4 of 4...
    .venv\Scripts\python.exe scripts\run_part.py --part 4 --total_parts 4
    pause
    exit /b
)

if /i "%user_choice%"=="G" (
    echo.
    echo Launching Graphical User Interface...
    start .venv\Scripts\pythonw.exe scripts\gui_launcher.py
    exit /b
)

if /i "%user_choice%"=="M" (
    echo.
    echo Merging all parts into final matching_results.tsv...
    .venv\Scripts\python.exe scripts\merge_parts.py
    pause
    exit /b
)

echo Exiting...
