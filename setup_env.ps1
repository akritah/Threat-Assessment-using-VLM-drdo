$ErrorActionPreference = "Stop"

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe scripts/verify_environment.py

Write-Host "Environment ready. Activate it with:"
Write-Host ".\.venv\Scripts\Activate.ps1"
