@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
    echo Error: Python virtual environment .venv was not found.
    echo Please run the setup script first: setup_project.bat
    pause
    exit /b 1
)

echo ===================================================
echo   DRDO VLM Threat Monitoring Dashboard Launching
echo ===================================================
echo.

.venv\Scripts\python.exe -u app.py
if %errorlevel% neq 0 (
    echo.
    echo Error: App exited with error code %errorlevel%.
    pause
)
