"""High-reliability model downloader.
Uses stable sequential single-threaded downloads and clean text logging to prevent hangs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

def main() -> int:
    token = os.getenv("HF_TOKEN")
    if not token or "token" in token.lower():
        print("Error: HF_TOKEN environment variable is missing or invalid.")
        return 1

    repo_id = "google/gemma-3-4b-it"
    print(f"[*] Launching sequential downloader for: {repo_id}")
    print("[*] Threading: Single-threaded (max_workers=1) to prevent connection hangs.")
    
    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            token=token,
            max_workers=1,
            resume_download=True,
        )
        print(f"\n[+] Success! Model is cached locally at: {local_dir}")
        return 0
    except Exception as e:
        print(f"\n[-] Download failed: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
