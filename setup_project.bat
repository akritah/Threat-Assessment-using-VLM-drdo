@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo   DRDO Threat Detection - Project Setup (Windows)
echo ===================================================
echo.

:: 1. Verify Python
echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in system PATH.
    echo Please install Python 3.9+ from https://python.org and try again.
    pause
    exit /b 1
)

:: 2. Initialize Virtual Environment
if not exist ".venv" (
    echo Creating Python virtual environment (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo Error: Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment (.venv) already exists.
)

:: 3. Upgrade Pip & Install Dependencies
echo Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip

echo Installing project dependencies from requirements.txt...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Dependency installation failed.
    pause
    exit /b 1
)

:: 4. Initialize Configuration (.env)
if not exist ".env" (
    echo Initializing configuration file (.env) from template...
    copy .env.example .env >nul
) else (
    echo Configuration file (.env) already exists.
)

:: 5. Check Ollama Service & Pull Model
echo Verifying Ollama service...
where ollama >nul 2>&1
if %errorlevel% equ 0 (
    :: Try querying Ollama local API tags using powershell
    powershell -Command "Invoke-RestMethod -Uri 'http://localhost:11434/api/tags' -ErrorAction Stop" >nul 2>&1
    if !errorlevel! equ 0 (
        echo Ollama local service detected and running.
        echo Pulling gemma3:4b model...
        ollama pull gemma3:4b
    ) else (
        echo Warning: Ollama client is installed, but the local background service is not running.
        echo Please launch the Ollama application, then run 'ollama pull gemma3:4b' manually.
    )
) else (
    echo Warning: Ollama CLI was not found.
    echo To use local Ollama VLM path, please install it from https://ollama.com.
)

:: 6. Run Diagnostics
echo.
echo Running capabilities diagnostics...
if exist "scripts\verify_environment.py" (
    .venv\Scripts\python.exe scripts\verify_environment.py
)

echo.
echo ===================================================
echo   Setup Complete!
echo   To launch the dashboard, run:  run_project.bat
echo ===================================================
echo.
pause
