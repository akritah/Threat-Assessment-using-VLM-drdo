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

from models.model_loader import load_base_model, load_finetuned_model
from training.config import TrainingConfig
from training.dataset import load_jsonl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _generate(model: Any, processor: Any, messages: list[dict[str, Any]], max_new_tokens: int = 512) -> str:
    inputs = processor.apply_chat_template(
        messages,
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

    prompt = record.get("prompt", "")
    image = record.get("image")
    if image:
        image_path = Path(image)
        if not image_path.is_absolute():
            image_path = project_root / image_path
        user_content: list[dict[str, Any]] | str = [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt},
        ]
    else:
        user_content = prompt

    return [{"role": "user", "content": user_content}]


def evaluate_samples(
    eval_path: Path,
    adapter_path: Path | None,
    *,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    records = load_jsonl(eval_path)
    if max_samples:
        records = records[:max_samples]

    base_model, processor = load_base_model()
    finetuned_model, _ = (
        load_finetuned_model(adapter_path) if adapter_path else (base_model, processor)
    )

    results: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        messages = _record_to_messages(record, _PROJECT_ROOT)
        expected = record.get("response", "")

        base_output = _generate(base_model, processor, messages)
        finetuned_output = _generate(finetuned_model, processor, messages) if adapter_path else base_output

        results.append(
            {
                "sample": index,
                "expected": expected,
                "base_output": base_output,
                "finetuned_output": finetuned_output,
            }
        )

    return results


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
        logging.error("Eval dataset not found: %s", eval_path)
        return 1

    adapter_path = Path(args.adapter) if args.adapter else config.output_path
    if not adapter_path.exists():
        logging.warning("Adapter not found at %s; evaluating base model only.", adapter_path)
        adapter_path = None

    try:
        results = evaluate_samples(eval_path, adapter_path, max_samples=args.max_samples)
        output_path = _PROJECT_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.info("Evaluation report saved: %s", output_path)
        return 0
    except Exception as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
