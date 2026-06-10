from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import requests

from prompts import FRAME_ANALYSIS_PROMPT, build_summary_prompt


class GemmaAnalyzer:
    def __init__(self, model: str = "gemma4", base_url: str = "http://localhost:11434", timeout: int = 180) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def analyze_frame(self, image_path: Path, frame_number: int) -> dict[str, Any]:
        result = self._generate(prompt=FRAME_ANALYSIS_PROMPT, images=[image_path])
        parsed = self._as_json(result)
        parsed["frame"] = frame_number
        parsed["image"] = str(image_path)
        return parsed

    def summarize(self, frame_results: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = build_summary_prompt(json.dumps(frame_results, indent=2))
        result = self._generate(prompt=prompt)
        return self._as_json(result)

    def _generate(self, prompt: str, images: list[Path] | None = None) -> str:
        message: dict[str, Any] = {"role": "user", "content": prompt}

        if images:
            message["images"] = [self._image_to_base64(path) for path in images]

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
            },
        }

        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. Start Ollama and pull a local multimodal model named {self.model}."
            ) from exc

        data = response.json()
        message_data = data.get("message", {})
        text = message_data.get("content") if isinstance(message_data, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Ollama returned an empty response")
        return text.strip()

    @staticmethod
    def _image_to_base64(path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return base64.b64encode(path.read_bytes()).decode("utf-8")

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
