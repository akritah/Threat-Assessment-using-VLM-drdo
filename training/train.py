"""QLoRA fine-tuning entry point using TRL SFTTrainer."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Preload NVIDIA CUDA libraries dynamically to fix bitsandbytes loading errors on Linux/Colab
def _preload_cuda_libs():
    import platform
    if platform.system() != "Linux":
        return
    import ctypes
    import site

    # Collect all site-packages paths
    paths = list(site.getsitepackages())
    try:
        user_site = site.getusersitepackages()
        paths.append(user_site)
    except Exception:
        pass

    for sdir in paths:
        nv_path = Path(sdir) / "nvidia"
        if nv_path.exists():
            # Preload nvjitlink first since other libraries depend on it
            for sub in nv_path.iterdir():
                if sub.name == "nvjitlink":
                    lib_dir = sub / "lib"
                    if lib_dir.exists():
                        for so_file in lib_dir.glob("libnvJitLink.so*"):
                            try:
                                ctypes.CDLL(str(so_file), mode=ctypes.RTLD_GLOBAL)
                            except Exception:
                                pass

            # Preload other CUDA runtime libraries (cublas, cudart, etc.)
            for sub in nv_path.iterdir():
                if sub.name != "nvjitlink":
                    lib_dir = sub / "lib"
                    if lib_dir.exists():
                        for so_file in lib_dir.glob("*.so*"):
                            try:
                                ctypes.CDLL(str(so_file), mode=ctypes.RTLD_GLOBAL)
                            except Exception:
                                pass

try:
    _preload_cuda_libs()
except Exception:
    pass


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
import env_loader
env_loader.load_env()

import argparse
import logging
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

from training.config import TrainingConfig
from training.dataset import Gemma3VLMDataCollator, create_sample_dataset, load_training_datasets

logger = logging.getLogger(__name__)


def _build_bnb_config(config: TrainingConfig) -> BitsAndBytesConfig | None:
    if not config.load_in_4bit or not torch.cuda.is_available():
        return None
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype, torch.bfloat16)
    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _build_lora_config(config: TrainingConfig) -> LoraConfig:
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def train(config: TrainingConfig, resume: bool = False) -> Path:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # CPU Fallback check: Disable quantization if CUDA is not available
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available. Falling back to CPU. Disabling 4-bit quantization and mixed precision."
        )
        config.load_in_4bit = False
        config.bf16 = False
        config.fp16 = False

    logger.info("Loading base model: %s", config.base_model_id)
    bnb_config = _build_bnb_config(config)

    token = os.getenv("HF_TOKEN")
    load_kwargs: dict[str, Any] = {
        "device_map": "auto" if torch.cuda.is_available() else {"": "cpu"},
        "trust_remote_code": True,
        "token": token,
    }
    if bnb_config:
        load_kwargs["quantization_config"] = bnb_config

    model = AutoModelForImageTextToText.from_pretrained(
        config.base_model_id,
        **load_kwargs,
    )
    processor = AutoProcessor.from_pretrained(config.base_model_id, trust_remote_code=True, token=token)

    # Set padding token if not set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    # For training, ensure padding side is right
    processor.tokenizer.padding_side = "right"

    if config.load_in_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Freeze base model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA adapters
    model = get_peft_model(model, _build_lora_config(config))
    model.print_trainable_parameters()

    train_dataset, eval_dataset = load_training_datasets(
        config.train_dataset_path,
        config.eval_dataset_path,
        processor,
        _PROJECT_ROOT,
    )

    checkpoint_dir = config.checkpoint_path
    logging_dir = config.logging_path
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    eval_strategy = "epoch" if eval_dataset else "no"
    save_strategy = "epoch" if eval_dataset else "steps"

    training_args = SFTConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_steps=config.save_steps if not eval_dataset else None,
        save_total_limit=config.save_total_limit,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.bf16 if torch.cuda.is_available() else False,
        fp16=config.fp16 if torch.cuda.is_available() else False,
        logging_dir=str(logging_dir),
        logging_strategy="steps",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="loss" if eval_dataset else None,
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        max_length=config.max_seq_length,
        dataset_kwargs={"skip_prepare_dataset": True},
        loss_type="nll",
    )

    collator = Gemma3VLMDataCollator(processor, _PROJECT_ROOT)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    resume_checkpoint = config.resume_from_checkpoint
    if resume and resume_checkpoint is None:
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
        if checkpoints:
            resume_checkpoint = str(checkpoints[-1])
            logger.info("Resuming from checkpoint: %s", resume_checkpoint)

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save adapter weights only — never overwrite the base model.
    output_path = config.output_path
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_path))
    processor.save_pretrained(str(output_path))
    logger.info("Adapter saved to: %s", output_path)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Gemma 3 4B")
    parser.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "config" / "training_config.yaml"),
        help="Path to training YAML config",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument(
        "--create-sample-data",
        action="store_true",
        help="Create sample train/eval JSONL files and exit",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if args.create_sample_data:
        sample_dir = _PROJECT_ROOT / "training" / "data"
        create_sample_dataset(sample_dir)
        logging.info("Sample dataset created in: %s", sample_dir)
        return 0

    config = TrainingConfig.from_yaml(args.config)
    try:
        adapter_path = train(config, resume=args.resume)
        logging.info("Training complete. Adapter at: %s", adapter_path)
        return 0
    except Exception as exc:
        logger.error("%s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
