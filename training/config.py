"""Training configuration for QLoRA fine-tuning."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "training_config.yaml"


@dataclass
class TrainingConfig:
    base_model_id: str = "google/gemma-3-4b-it"
    output_dir: str = "adapters/threat_assessment"
    train_dataset: str = "training/data/train.jsonl"
    eval_dataset: str | None = "training/data/eval.jsonl"
    checkpoint_dir: str | None = None
    logging_dir: str | None = None

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2.0e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 3
    max_grad_norm: float = 0.0

    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    resume_from_checkpoint: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> TrainingConfig:
        raw_path = path or os.getenv("TRAINING_CONFIG_PATH", _DEFAULT_CONFIG_PATH)
        config_path = Path(raw_path)
        if not config_path.is_absolute():
            config_path = _PROJECT_ROOT / config_path
        if not config_path.exists():
            # Support loading configs/ prefixes dynamically
            alt_path = _PROJECT_ROOT / "configs" / config_path.name
            if alt_path.exists():
                config_path = alt_path
            else:
                return cls()

        with config_path.open(encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}

        return cls(**{key: raw[key] for key in raw if key in cls.__dataclass_fields__})

    def resolve_path(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute():
            return path
        return _PROJECT_ROOT / path

    @property
    def output_path(self) -> Path:
        return self.resolve_path(self.output_dir)

    @property
    def train_dataset_path(self) -> Path:
        return self.resolve_path(self.train_dataset)

    @property
    def eval_dataset_path(self) -> Path | None:
        if not self.eval_dataset:
            return None
        return self.resolve_path(self.eval_dataset)

    @property
    def checkpoint_path(self) -> Path:
        if self.checkpoint_dir:
            return self.resolve_path(self.checkpoint_dir)
        return self.output_path / "checkpoints"

    @property
    def logging_path(self) -> Path:
        if self.logging_dir:
            return self.resolve_path(self.logging_dir)
        return self.output_path / "logs"
