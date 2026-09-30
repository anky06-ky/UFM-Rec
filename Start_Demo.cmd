@echo off
setlocal
cd /d "%~dp0"
if exist "tmp\ufm_checks_env\Scripts\python.exe" (
  "tmp\ufm_checks_env\Scripts\python.exe" src\launch_demo.py
) else if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" src\launch_demo.py
) else (
  py src\launch_demo.py
)
if errorlevel 1 pause
