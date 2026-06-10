"""Factory for creating inference backends without changing existing call sites.

Default behavior is unchanged: returns the Ollama-based GemmaAnalyzer.
Set INFERENCE_BACKEND=hf (or use --backend hf) to enable HuggingFace + adapter loading.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "adapter_config.yaml"


@runtime_checkable
class AnalyzerProtocol(Protocol):
    def analyze_frame(self, image_path: Path, frame_number: int) -> dict[str, Any]: ...

    def summarize(self, frame_results: list[dict[str, Any]]) -> dict[str, Any]: ...


def _load_config() -> dict[str, Any]:
    config_path = Path(os.getenv("ADAPTER_CONFIG_PATH", str(_DEFAULT_CONFIG_PATH)))
    if not config_path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def create_analyzer(
    *,
    model: str | None = None,
    base_url: str | None = None,
    backend: str | None = None,
    adapter: str | None = None,
) -> AnalyzerProtocol:
    """Create an analyzer instance based on configuration.

    All parameters are optional. When omitted, environment variables and
    config/adapter_config.yaml are consulted. Defaults preserve the original
    Ollama workflow.
    """
    config = _load_config()

    resolved_backend = (
        backend
        or os.getenv("INFERENCE_BACKEND", config.get("inference_backend", "ollama"))
    ).lower()

    if resolved_backend == "hf":
        from hf_gemma_analyzer import HFGemmaAnalyzer

        resolved_adapter = adapter or os.getenv("ACTIVE_ADAPTER", config.get("active_adapter"))
        return HFGemmaAnalyzer(adapter_name=resolved_adapter)

    from gemma_analyzer import GemmaAnalyzer

    resolved_model = model or os.getenv("OLLAMA_MODEL", config.get("ollama_model", "gemma4"))
    resolved_url = base_url or os.getenv("OLLAMA_URL", config.get("ollama_url", "http://localhost:11434"))
    return GemmaAnalyzer(model=resolved_model, base_url=resolved_url)
