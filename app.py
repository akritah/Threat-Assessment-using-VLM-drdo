"""Root stub for the Gradio Surveillance Dashboard app.
Redirects execution to frontend/app.py.
"""

from __future__ import annotations

import env_loader
env_loader.load_env()

from frontend.app import demo

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
