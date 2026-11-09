@echo off
cd /d "%~dp0"
setlocal
if not exist "venv\Scripts\python.exe" (
  echo Set up Python first. See docs\session_workflow.md.
  pause
  exit /b 2
)
if not "%~1"=="" goto analyse
"venv\Scripts\python.exe" uav_session.py --menu
set "run_status=%errorlevel%"
pause
exit /b %run_status%

:analyse
"venv\Scripts\python.exe" automation.py %*
set "run_status=%errorlevel%"
if not "%run_status%"=="0" pause
exit /b %run_status%
