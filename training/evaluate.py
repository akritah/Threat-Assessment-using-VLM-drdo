"""Evaluate a fine-tuned adapter against the base model."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
import env_loader
env_loader.load_env()

import argparse
import json
import logging
from typing import Any

import torch
from PIL import Image

from models.model_loader import load_base_model, load_finetuned_model
from training.config import TrainingConfig
from training.dataset import load_jsonl

logger = logging.getLogger(__name__)


def _generate(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    max_new_tokens: int = 512,
) -> str:
    import copy

    # Ensure all image paths are loaded as PIL Images
    messages_copy = copy.deepcopy(messages)
    for msg in messages_copy:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image" and isinstance(item.get("image"), str):
                    img_str = item["image"].replace("\\", "/")
                    img_path = Path(img_str)
                    if not img_path.is_absolute():
                        img_path = _PROJECT_ROOT / img_path
                    if img_path.exists():
                        item["image"] = Image.open(img_path).convert("RGB")
                    else:
                        logger.warning("Image path not found during generation: %s", img_path)

    inputs = processor.apply_chat_template(
        messages_copy,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    input_len = inputs["input_ids"].shape[-1]
    generated = output_ids[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def _record_to_messages(record: dict[str, Any], project_root: Path) -> list[dict[str, Any]]:
    if "messages" in record:
        return record["messages"]

    prompt = record.get("prompt", record.get("instruction", ""))
    image = record.get("image")
    if image:
        image_str = image.replace("\\", "/")
        image_path = Path(image_str)
        if not image_path.is_absolute():
            image_path = project_root / image_path
        user_content: list[dict[str, Any]] | str = [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt},
        ]
    else:
        user_content = prompt

    return [{"role": "user", "content": user_content}]


def get_losses_from_checkpoints(checkpoint_dir: Path) -> tuple[float | None, float | None]:
    """Search for trainer_state.json in the checkpoints directory and extract loss metrics."""
    train_loss = None
    val_loss = None
    state_files = sorted(checkpoint_dir.glob("**/trainer_state.json"), key=lambda p: p.stat().st_mtime)
    if state_files:
        latest_state = state_files[-1]
        try:
            with latest_state.open(encoding="utf-8") as handle:
                state_data = json.load(handle)
            log_history = state_data.get("log_history", [])
            for entry in reversed(log_history):
                if "loss" in entry and train_loss is None:
                    train_loss = entry["loss"]
                if "eval_loss" in entry and val_loss is None:
                    val_loss = entry["eval_loss"]
                if train_loss is not None and val_loss is not None:
                    break
        except Exception as exc:
            logger.warning("Could not parse trainer state file: %s", exc)
    return train_loss, val_loss


def evaluate_samples(
    eval_path: Path,
    adapter_path: Path | None,
    *,
    max_samples: int | None = None,
) -> tuple[list[dict[str, Any]], int, float]:
    records = load_jsonl(eval_path)
    if max_samples:
        records = records[:max_samples]

    base_model, processor = load_base_model()
    finetuned_model, _ = (
        load_finetuned_model(adapter_path) if adapter_path else (base_model, processor)
    )

    # Calculate adapter trainable parameters
    trainable_params = sum(p.numel() for n, p in finetuned_model.named_parameters() if "lora_" in n)

    # Calculate adapter size on disk
    adapter_size_bytes = 0
    if adapter_path and adapter_path.exists():
        for p in adapter_path.glob("**/*"):
            if p.is_file():
                adapter_size_bytes += p.stat().st_size
    adapter_size_mb = adapter_size_bytes / (1024 * 1024)

    results: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        messages = _record_to_messages(record, _PROJECT_ROOT)
        expected = record.get("response", "")

        base_output = _generate(base_model, processor, messages)
        finetuned_output = (
            _generate(finetuned_model, processor, messages) if adapter_path else base_output
        )

        results.append(
            {
                "sample": index,
                "expected": expected,
                "base_output": base_output,
                "finetuned_output": finetuned_output,
            }
        )

    return results, trainable_params, adapter_size_mb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned model outputs")
    parser.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "config" / "training_config.yaml"),
        help="Path to training YAML config",
    )
    parser.add_argument("--eval-dataset", help="Override eval dataset path")
    parser.add_argument("--adapter", help="Path to adapter directory")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument(
        "--output",
        default="outputs/evaluation_report.json",
        help="Where to write comparison results",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    config = TrainingConfig.from_yaml(args.config)
    eval_path = Path(args.eval_dataset) if args.eval_dataset else config.eval_dataset_path
    if eval_path is None or not eval_path.exists():
        logger.error("Eval dataset not found: %s", eval_path)
        return 1

    adapter_path = Path(args.adapter) if args.adapter else config.output_path
    checkpoint_dir = config.checkpoint_path

    if not adapter_path.exists():
        logger.warning("Adapter not found at %s; evaluating base model only.", adapter_path)
        adapter_path = None

    try:
        results, trainable_params, adapter_size_mb = evaluate_samples(
            eval_path,
            adapter_path,
            max_samples=args.max_samples,
        )

        # Retrieve losses from checkpoints if possible
        train_loss, val_loss = get_losses_from_checkpoints(checkpoint_dir)

        report = {
            "metrics": {
                "training_loss": train_loss,
                "validation_loss": val_loss,
                "adapter_size_mb": adapter_size_mb,
                "trainable_parameter_count": trainable_params,
            },
            "comparisons": results,
        }

        output_path = _PROJECT_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Evaluation report saved: %s", output_path)
        return 0
    except Exception as exc:
        logger.error("%s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
