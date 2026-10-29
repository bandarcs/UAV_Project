@echo off
cd /d "%~dp0"
if "%~1"=="" (
  echo Drag a recorded video onto run.bat or pass its path.
  pause
  exit /b 2
)
venv\Scripts\python.exe automation.py %*
if errorlevel 1 pause
