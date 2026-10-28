@echo off
chcp 65001 >nul
cd /d "%~dp0"
venv\Scripts\python.exe detect_plane_better.py %*
echo.
pause