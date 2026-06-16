"""Dataset loading and formatting for Gemma 3 4B SFT."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from PIL import Image
import torch


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
        # Standardize existing messages content types to prevent PyArrow conflicts
        messages = record["messages"]
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [{"type": "text", "text": content}]
        return messages

    # Support both "prompt" and "instruction" keys
    prompt = record.get("prompt", record.get("instruction", ""))
    response = record.get("response", "")
    image = record.get("image")

    user_content: list[dict[str, Any]]
    if image:
        image_str = image.replace("\\", "/")
        image_path = Path(image_str)
        if not image_path.is_absolute():
            image_path = project_root / image_path
        user_content = [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt},
        ]
    else:
        user_content = [
            {"type": "text", "text": prompt}
        ]

    # Standardize assistant content as a list of dicts to prevent PyArrow type conflicts
    assistant_content = [
        {"type": "text", "text": response}
    ]

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def format_dataset(records: list[dict[str, Any]], processor: Any, project_root: Path) -> Dataset:
    """Convert raw records into a HuggingFace Dataset with a 'messages' column."""
    formatted = []
    for record in records:
        messages = _record_to_messages(record, project_root)
        formatted.append({"messages": messages})
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


class Gemma3VLMDataCollator:
    """Collate multimodal examples for Gemma 3 Vision fine-tuning.

    Lazily loads PIL images from file paths inside raw messages and batches inputs
    using the model's processor, returning tokenized inputs and labels.
    """

    def __init__(self, processor: Any, project_root: Path) -> None:
        self.processor = processor
        self.project_root = project_root

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        processed_examples: list[list[dict[str, Any]]] = []
        for feat in features:
            messages = copy.deepcopy(feat["messages"])
            for msg in messages:
                content = msg.get("content")
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "image" and isinstance(item.get("image"), str):
                            img_str = item["image"].replace("\\", "/")
                            img_path = Path(img_str)
                            if not img_path.is_absolute():
                                img_path = self.project_root / img_path
                            item["image"] = Image.open(img_path).convert("RGB")
            processed_examples.append(messages)

        # Batch process using processor's apply_chat_template
        batch = self.processor.apply_chat_template(
            processed_examples,
            tokenize=True,
            return_dict=True,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()

        # Mask user query/instruction tokens in labels
        for i, messages in enumerate(processed_examples):
            prompt_messages = messages[:-1]
            prompt_inputs = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]
            # Since padding is 'right', user prompt is at the start of input_ids
            labels[i, :prompt_len] = -100

        # Mask padding tokens in labels
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch


def create_sample_dataset(output_dir: Path) -> None:
    """Write example JSONL files demonstrating the expected training format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = [
        {
            "image": "outputs/frames/frame_001.jpg",
            "instruction": "Describe what is happening in this scene.",
            "response": "A person is seated at a desk using a computer.",
        },
        {
            "image": "outputs/frames/frame_001.jpg",
            "instruction": "Describe what is happening in this scene.",
            "response": "The room is an indoor office containing a desk and a monitor.",
        },
    ]

    train_path = output_dir / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for sample in train_samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    eval_path = output_dir / "eval.jsonl"
    with eval_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(train_samples[0], ensure_ascii=False) + "\n")
