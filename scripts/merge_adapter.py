"""Merge a LoRA adapter into the base model for standalone export.

This is a separate utility — the default workflow keeps adapters separate
and never overwrites base model files in models/gemma_base/.
"""

from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
import env_loader
env_loader.load_env()

import argparse
import logging

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def merge_adapter(
    adapter_path: Path,
    output_path: Path,
    *,
    model_id: str | None = None,
    save_merged: bool = True,
) -> Path:
    from peft import PeftModel

    from models.model_loader import load_base_model

    model, processor = load_base_model(model_id=model_id, use_4bit=False, device_map="cpu")
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    merged = model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    if save_merged:
        merged.save_pretrained(str(output_path))
        processor.save_pretrained(str(output_path))
        logging.info("Merged model saved to: %s", output_path)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model for export")
    parser.add_argument("--adapter", required=True, help="Path to adapter directory")
    parser.add_argument(
        "--output",
        default="outputs/merged_model",
        help="Directory for merged model export (does not touch models/gemma_base/)",
    )
    parser.add_argument("--model-id", help="Override HuggingFace base model ID")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    adapter_path = Path(args.adapter)
    if not adapter_path.is_absolute():
        adapter_path = _PROJECT_ROOT / adapter_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _PROJECT_ROOT / output_path

    try:
        merge_adapter(adapter_path, output_path, model_id=args.model_id)
        return 0
    except Exception as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
