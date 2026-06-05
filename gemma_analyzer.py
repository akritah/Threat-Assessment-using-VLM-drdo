from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import requests


class GemmaAnalyzer:
    def __init__(self, model: str = "gemma4", base_url: str = "http://localhost:11434", timeout: int = 180) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def analyze_frame(self, image_path: Path, frame_number: int) -> dict[str, Any]:
        prompt = (
            "Analyze only what is visible in this CCTV/video frame. "
            "Return JSON with these keys: objects_present, people_present, actions_occurring, "
            "environment_description, description. "
            "Mention suspicious activity only if it is visible. Do not describe documents or forms unless they are visible."
        )
        result = self._generate(prompt=prompt, images=[image_path])
        parsed = self._as_json(result)
        parsed["frame"] = frame_number
        parsed["image"] = str(image_path)
        return parsed

    def summarize(self, frame_results: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = (
            "Create a JSON summary from these chronological CCTV frame observations. "
            "Return one JSON object only. Do not return an empty object. "
            "Use exactly these keys: timeline_of_events, overall_activity_summary, key_observations, final_activity_description. "
            "timeline_of_events and key_observations must be arrays of short strings. "
            "overall_activity_summary and final_activity_description must be short natural strings.\n\n"
            f"{json.dumps(frame_results, indent=2)}"
        )
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
