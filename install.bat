@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 goto fallback
py -3 launch.py --background --setup
if errorlevel 1 pause
exit /b
:fallback
python launch.py --background --setup
if errorlevel 1 pause
