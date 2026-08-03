"""Root launcher for the Gradio Surveillance Dashboard app.
Performs startup diagnostics and starts the server.
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("drdo_launcher")

# 1. Ensure project directories exist
PROJECT_ROOT = Path(__file__).resolve().parent
for sub in ["outputs", "logs", "evaluation/outputs", "evaluation/reports", "evaluation/plots"]:
    (PROJECT_ROOT / sub).mkdir(parents=True, exist_ok=True)

# 2. Load environment variables
try:
    import env_loader
    env_loader.load_env()
except ImportError:
    logger.error("Error: env_loader.py module not found in root. Make sure you are launching from the root directory.")
    sys.exit(1)

# 3. Perform Startup Diagnostics
backend = os.getenv("INFERENCE_BACKEND", "ollama")
if backend == "ollama":
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    logger.info(f"Checking connection to Ollama local API at: {ollama_url}...")
    try:
        import requests
        r = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if r.status_code == 200:
            logger.info("Ollama connection verified successfully!")
        else:
            logger.warning(f"Ollama returned status code {r.status_code}. Make sure service is responsive.")
    except Exception as e:
        logger.error(
            "\n" + "="*60 + "\n"
            f"DIAGNOSTIC ERROR: Cannot connect to Ollama service at {ollama_url}.\n"
            "If using Ollama inference:\n"
            "  1. Make sure Ollama is installed and running.\n"
            "  2. If running inside Docker, set OLLAMA_URL=http://host.docker.internal:11434 in your .env file.\n"
            "  3. If running locally, start the Ollama background service.\n"
            "============================================================\n"
        )
else:
    logger.info(f"Using HuggingFace backend (Active Adapter: {os.getenv('ACTIVE_ADAPTER', 'None')})")

# 4. Import Gradio dashboard app
try:
    import gradio as gr
    from frontend.app import demo
except ImportError as e:
    logger.error(
        "\n" + "="*60 + "\n"
        f"DEPENDENCY ERROR: Missing package: {e}\n"
        "Please run the setup script to install all requirements:\n"
        "  Linux/macOS:  ./setup.sh\n"
        "  Windows:      setup_project.bat\n"
        "============================================================\n"
    )
    sys.exit(1)

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    try:
        port = int(os.getenv("PORT", "7860"))
    except ValueError:
        port = 7860
        
    logger.info(f"Launching Gradio Web Server at http://{host}:{port}...")
    demo.launch(server_name=host, server_port=port, share=True)
