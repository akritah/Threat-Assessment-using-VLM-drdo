#!/usr/bin/env bash
# ===================================================
#   DRDO Threat Detection - Setup Script (Linux/macOS)
# ===================================================

set -euo pipefail

echo "==================================================="
echo "  DRDO Threat Detection - Project Setup"
echo "==================================================="
echo

# 1. Verify Python Installation
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH." >&2
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Found Python $PYTHON_VERSION"

# 2. Initialize Virtual Environment
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
else
    echo "Virtual environment (.venv) already exists."
fi

# 3. Upgrade Pip & Install Dependencies
echo "Upgrading pip..."
.venv/bin/pip install --upgrade pip

echo "Installing project dependencies..."
.venv/bin/pip install -r requirements.txt

# 4. Initialize Environment Configuration
if [ ! -f ".env" ]; then
    echo "Initializing configuration (.env) from template..."
    cp .env.example .env
else
    echo "Configuration file (.env) already exists."
fi

# 5. Verify Ollama Service and Models
echo "Verifying Ollama service..."
if command -v ollama &> /dev/null; then
    # Check if Ollama service is responsive
    if curl -s -f http://localhost:11434/api/tags > /dev/null; then
        echo "Ollama service detected and running."
        echo "Pulling gemma3:4b model..."
        ollama pull gemma3:4b
    else
        echo "Warning: Ollama client is installed, but the local service is not running."
        echo "Please start the Ollama application and run 'ollama pull gemma3:4b' manually."
    fi
else
    echo "Warning: Ollama command-line tool was not found."
    echo "To run with local Ollama inference, please download it from: https://ollama.com"
fi

# 6. Run Diagnostics
echo "Running system capabilities verification..."
if [ -f "scripts/verify_environment.py" ]; then
    .venv/bin/python scripts/verify_environment.py
fi

echo
echo "==================================================="
echo "  Setup Complete!"
echo "  To run the dashboard:  ./run.sh"
echo "==================================================="
echo
