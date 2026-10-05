@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" src\launch_demo.py
) else (
  py src\launch_demo.py
)
if errorlevel 1 pause
