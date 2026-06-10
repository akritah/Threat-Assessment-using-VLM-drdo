"""Environment variable loader with zero-dependency fallback.

Loads environment variables from a local `.env` file into `os.environ`
and configures python paths for scripts in subdirectories.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def load_env() -> None:
    """Load environment variables from the project root's .env file."""
    # Ensure project root is in sys.path so subfolder scripts can import local packages
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    env_path = project_root / ".env"
    if not env_path.exists():
        return

    # Try using python-dotenv first
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path)
        return
    except ImportError:
        pass

    # Fallback: Zero-dependency parser
    try:
        with env_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    # Strip surrounding quotes and whitespace
                    val = val.strip().strip('"').strip("'").strip()
                    # Populate os.environ if not already set (replicates load_dotenv default)
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception as exc:
        logger.warning("Failed to manually parse .env file: %s", exc)
