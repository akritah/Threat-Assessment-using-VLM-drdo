"""Root stub for the Video Analysis CLI.
Redirects execution to backend/main.py.
"""

from __future__ import annotations

import env_loader
env_loader.load_env()

import sys
from backend.main import main

if __name__ == "__main__":
    raise SystemExit(main())
