#!/usr/bin/env bash
# ===================================================
#   DRDO Threat Detection - Launch Script (Linux/macOS)
# ===================================================

set -euo pipefail

if [ ! -d ".venv" ] || [ ! -f ".venv/bin/python" ]; then
    echo "Error: Virtual environment (.venv) not found." >&2
    echo "Please run the setup script first:  ./setup.sh" >&2
    exit 1
fi

echo "==================================================="
echo "  DRDO VLM Threat Monitoring Dashboard Launching"
echo "==================================================="
echo

# Run the Gradio entrypoint
.venv/bin/python app.py
