@echo off
cd /d "%~dp0"
title Amazon ML Hackathon GUI

set "PY_CMD="
if exist ".venv\Scripts\pythonw.exe" (
    set "PY_CMD=.venv\Scripts\pythonw.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PY_CMD=.venv\Scripts\python.exe"
) else (
    set "PY_CMD=python"
)

start "" "%PY_CMD%" "scripts\gui_launcher.py"
