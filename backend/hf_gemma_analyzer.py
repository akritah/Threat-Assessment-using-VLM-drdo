"""Optional HuggingFace inference backend with LoRA adapter support.

Implements the same public interface as GemmaAnalyzer (analyze_frame, summarize)
so it can be used as a drop-in replacement when INFERENCE_BACKEND=hf.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from models.model_loader import load_adapter_if_available, load_base_model
from prompts import FRAME_ANALYSIS_PROMPT, build_summary_prompt

logger = logging.getLogger(__name__)


class HFGemmaAnalyzer:
    """HuggingFace + optional QLoRA adapter backend."""

    def __init__(
        self,
        adapter_name: str | None = None,
        *,
        model_id: str | None = None,
        use_4bit: bool = True,
        max_new_tokens: int = 512,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self._model, self._processor = load_base_model(model_id=model_id, use_4bit=use_4bit)
        self._model = load_adapter_if_available(self._model, adapter_name=adapter_name)
        self._model.eval()

    def analyze_frame(self, image_path: Path, frame_number: int) -> dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": FRAME_ANALYSIS_PROMPT},
                ],
            }
        ]
        result = self._generate(messages)
        parsed = self._as_json(result)
        parsed["frame"] = frame_number
        parsed["image"] = str(image_path)
        return parsed

    def summarize(self, frame_results: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = build_summary_prompt(json.dumps(frame_results, indent=2))
        messages = [{"role": "user", "content": prompt}]
        result = self._generate(messages)
        return self._as_json(result)

    def _generate(self, messages: list[dict[str, Any]]) -> str:
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated = output_ids[0][input_len:]
        text = self._processor.decode(generated, skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError("HuggingFace model returned an empty response")
        return text

    @staticmethod
    def _as_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError:
            value = {"description": text}

        if not isinstance(value, dict):
            return {"description": text}
        return value
