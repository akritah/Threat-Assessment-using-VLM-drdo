@echo off
echo ===================================================
echo   DRDO Threat Detection - Project Setup Script
echo ===================================================
echo.

echo 1. Creating Python virtual environment (.venv)...
python -m venv .venv
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    pause
    exit /b 1
)

echo.
echo 2. Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip

echo.
echo 3. Installing dependencies from requirements.txt...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo 4. Verifying Ollama service and pulling Gemma 3...
where ollama >nul 2>nul
if %errorlevel% equ 0 (
    echo Pulling gemma3:4b model...
    ollama pull gemma3:4b
) else (
    echo Warning: Ollama is not installed or running. Please install it from https://ollama.com to use the local inference path.
)

echo.
echo ===================================================
echo   Setup Complete!
echo   To run the Video Q&A CLI Agent, run:
echo   .venv\Scripts\python.exe video_qa.py
echo ===================================================
echo.
pause
