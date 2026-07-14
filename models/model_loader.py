"""HuggingFace model loading with optional QLoRA adapter support.

This module is independent of the Ollama inference path. It is used by the
optional HF backend and the training/evaluation scripts.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "adapter_config.yaml"


def _load_adapter_config(config_path: Path | None = None) -> dict[str, Any]:
    raw_path = config_path or Path(os.getenv("ADAPTER_CONFIG_PATH", str(_DEFAULT_CONFIG_PATH)))
    path = Path(raw_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    if not path.exists():
        alt_path = _PROJECT_ROOT / "configs" / path.name
        if alt_path.exists():
            path = alt_path
        else:
            return {}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_base_model_id(config: dict[str, Any]) -> str:
    return os.getenv(
        "HF_BASE_MODEL_ID",
        config.get("base_model_id", "google/gemma-3-4b-it"),
    )


def _resolve_base_model_path(config: dict[str, Any]) -> Path:
    raw = os.getenv("HF_BASE_MODEL_PATH", config.get("base_model_path", "models/gemma_base"))
    path = Path(raw)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def _resolve_adapters_dir(config: dict[str, Any]) -> Path:
    raw = os.getenv("ADAPTERS_DIR", config.get("adapters_dir", "adapters"))
    path = Path(raw)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def _resolve_adapter_path(adapter_name: str | None, config: dict[str, Any]) -> Path | None:
    if not adapter_name:
        return None
    adapters_dir = _resolve_adapters_dir(config)
    candidates = [
        adapters_dir / adapter_name,
        adapters_dir / adapter_name / "final_adapter",
    ]
    for adapter_path in candidates:
        if (adapter_path / "adapter_config.json").exists():
            return adapter_path
    logger.warning("Adapter not found for %r in %s", adapter_name, adapters_dir)
    return None


def _get_bnb_config() -> Any:
    import torch
    from transformers import BitsAndBytesConfig

    compute_dtype_name = os.getenv("BNB_4BIT_COMPUTE_DTYPE", "bfloat16")
    compute_dtype = getattr(torch, compute_dtype_name, torch.bfloat16)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=os.getenv("BNB_4BIT_QUANT_TYPE", "nf4"),
        bnb_4bit_use_double_quant=os.getenv("BNB_4BIT_USE_DOUBLE_QUANT", "true").lower() == "true",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _get_model_class():
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText


def _get_processor(model_id: str, local_path: Path | None = None):
    from transformers import AutoProcessor
    import os

    if local_path and local_path.exists() and (local_path / "config.json").exists():
        return AutoProcessor.from_pretrained(str(local_path), trust_remote_code=True)
    token = os.getenv("HF_TOKEN")
    return AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=token)


def load_base_model(
    model_id: str | None = None,
    *,
    config_path: Path | None = None,
    device_map: str | None = "auto",
    use_4bit: bool = True,
) -> tuple[Any, Any]:
    """Load the base Gemma 3 4B model (optionally 4-bit quantized).

    Returns (model, processor). Base model weights are never modified.
    """
    import torch
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Disabling 4-bit quantization and mapping device to CPU.")
        use_4bit = False
        device_map = {"": "cpu"}

    config = _load_adapter_config(config_path)
    resolved_id = model_id or _resolve_base_model_id(config)
    local_path = _resolve_base_model_path(config)

    model_cls = _get_model_class()
    processor = _get_processor(resolved_id, local_path)

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": device_map,
        "token": os.getenv("HF_TOKEN"),
    }
    if torch.cuda.is_available():
        load_kwargs["torch_dtype"] = torch.float16

    if use_4bit:
        load_kwargs["quantization_config"] = _get_bnb_config()

    if local_path.exists() and (local_path / "config.json").exists():
        logger.info("Loading base model from local path: %s", local_path)
        model = model_cls.from_pretrained(str(local_path), **load_kwargs)
    else:
        logger.info("Loading base model from HuggingFace Hub: %s", resolved_id)
        model = model_cls.from_pretrained(resolved_id, **load_kwargs)

    return model, processor


def load_finetuned_model(
    adapter_path: str | Path,
    model_id: str | None = None,
    *,
    config_path: Path | None = None,
    device_map: str | None = "auto",
    use_4bit: bool = True,
) -> tuple[Any, Any]:
    """Load base model with a LoRA adapter applied.

    Adapter weights are kept separate from the base model on disk.
    """
    from peft import PeftModel

    adapter = Path(adapter_path)
    if not adapter.is_absolute():
        adapter = _PROJECT_ROOT / adapter

    if not adapter.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter}")

    model, processor = load_base_model(
        model_id=model_id,
        config_path=config_path,
        device_map=device_map,
        use_4bit=use_4bit,
    )
    logger.info("Loading LoRA adapter from: %s", adapter)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    return model, processor


def load_adapter_if_available(
    model: Any,
    adapter_name: str | None = None,
    *,
    config_path: Path | None = None,
) -> Any:
    """Attach a LoRA adapter to an already-loaded model if available.

    Falls back to the base model when no adapter is configured or loading fails.
    """
    from peft import PeftModel

    config = _load_adapter_config(config_path)
    resolved_name = adapter_name or os.getenv("ACTIVE_ADAPTER", config.get("active_adapter"))
    adapter_path = _resolve_adapter_path(resolved_name, config)

    if adapter_path is None:
        logger.info("No adapter configured; using base model.")
        return model

    try:
        logger.info("Attempting to load adapter: %s", adapter_path)
        return PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    except Exception as exc:
        logger.warning("Adapter loading failed (%s); falling back to base model.", exc)
        return model
