"""Dataset loading and formatting for Gemma 3 4B SFT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return records


def _record_to_messages(record: dict[str, Any], project_root: Path) -> list[dict[str, Any]]:
    if "messages" in record:
        return record["messages"]

    prompt = record.get("prompt", "")
    response = record.get("response", "")
    image = record.get("image")

    user_content: list[dict[str, Any]] | str
    if image:
        image_path = Path(image)
        if not image_path.is_absolute():
            image_path = project_root / image_path
        user_content = [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt},
        ]
    else:
        user_content = prompt

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response},
    ]


def format_dataset(records: list[dict[str, Any]], processor: Any, project_root: Path) -> Dataset:
    """Convert raw records into a HuggingFace Dataset with a 'text' column for SFTTrainer."""

    def _format_example(record: dict[str, Any]) -> dict[str, str]:
        messages = _record_to_messages(record, project_root)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    formatted = [_format_example(record) for record in records]
    return Dataset.from_list(formatted)


def load_training_datasets(
    train_path: Path,
    eval_path: Path | None,
    processor: Any,
    project_root: Path,
) -> tuple[Dataset, Dataset | None]:
    train_records = load_jsonl(train_path)
    if not train_records:
        raise ValueError(f"Training dataset is empty: {train_path}")

    train_dataset = format_dataset(train_records, processor, project_root)

    eval_dataset = None
    if eval_path and eval_path.exists():
        eval_records = load_jsonl(eval_path)
        if eval_records:
            eval_dataset = format_dataset(eval_records, processor, project_root)

    return train_dataset, eval_dataset


def create_sample_dataset(output_dir: Path) -> None:
    """Write example JSONL files demonstrating the expected training format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = [
        {
            "image": "outputs/frames/frame_001.jpg",
            "prompt": (
                "Analyze only what is visible in this CCTV/video frame. "
                "Return JSON with these keys: objects_present, people_present, actions_occurring, "
                "environment_description, description."
            ),
            "response": json.dumps(
                {
                    "objects_present": ["desk", "monitor"],
                    "people_present": ["one person seated"],
                    "actions_occurring": ["working at desk"],
                    "environment_description": "indoor office",
                    "description": "A person is seated at a desk using a computer.",
                }
            ),
        },
        {
            "prompt": "Summarize these frame observations as JSON.",
            "response": json.dumps(
                {
                    "timeline_of_events": ["Frame 1: person seated at desk"],
                    "overall_activity_summary": "Routine office activity.",
                    "key_observations": ["One person present", "Indoor office setting"],
                    "final_activity_description": "Person working at desk throughout.",
                }
            ),
        },
    ]

    train_path = output_dir / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for sample in train_samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    eval_path = output_dir / "eval.jsonl"
    with eval_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(train_samples[0], ensure_ascii=False) + "\n")
