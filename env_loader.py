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


def setup_logging(log_file_name: str = "run.log") -> None:
    """Configure structured logging to console and a local log file under logs/."""
    project_root = Path(__file__).resolve().parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = logs_dir / log_file_name

    # Create root logger
    root_logger = logging.getLogger()
    if root_logger.handlers:
        # Prevent adding handlers multiple times
        return

    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler
    try:
        file_handler = logging.FileHandler(str(log_file_path), encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except Exception as exc:
        print(f"Warning: Could not create log file at {log_file_path}: {exc}", file=sys.stderr)

    # Silence verbose third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def load_env() -> None:
    """Load environment variables from the project root's .env file."""
    # Ensure project root is in sys.path so subfolder scripts can import local packages
    project_root = Path(__file__).resolve().parent
    for path_to_add in [project_root, project_root / "backend", project_root / "frontend"]:
        path_str = str(path_to_add)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    env_path = project_root / ".env"
    if env_path.exists():
        # Try using python-dotenv first
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=env_path)
        except ImportError:
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
                print(f"Warning: Failed to parse .env file: {exc}", file=sys.stderr)

    # Automatically set up logging for the session
    setup_logging()
