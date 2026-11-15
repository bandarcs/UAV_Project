@echo off
REM Run from the project folder even when Windows starts this file elsewhere.
cd /d "%~dp0"
setlocal
if not exist "venv\Scripts\python.exe" (
  echo Set up Python first. See docs\session_workflow.md.
  pause
  exit /b 2
)
REM One Python entry point handles both the menu and arguments.
"venv\Scripts\python.exe" "%~dp0uav_project.py" %*
set "run_status=%errorlevel%"
if "%~1"=="" (
  pause
) else if not "%run_status%"=="0" (
  pause
)
exit /b %run_status%
