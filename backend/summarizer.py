from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from gemma_analyzer import GemmaAnalyzer


def summarize_video(analyzer: "GemmaAnalyzer", frame_results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = analyzer.summarize(frame_results)
    fallback = _fallback_summary(frame_results)
    return {
        "timeline_of_events": summary.get("timeline_of_events") or fallback["timeline_of_events"],
        "overall_activity_summary": summary.get("overall_activity_summary") or fallback["overall_activity_summary"],
        "key_observations": summary.get("key_observations") or fallback["key_observations"],
        "final_activity_description": summary.get("final_activity_description") or fallback["final_activity_description"],
    }


def build_text_report(frame_results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("Frame-by-frame analysis")
    lines.append("")
    for frame in frame_results:
        lines.append(f"Frame {frame.get('frame', '')}:")
        lines.append(str(frame.get("description") or frame.get("environment_description") or "No description available."))
        lines.append("")
        lines.append(f"Objects: {_join_value(frame.get('objects_present'))}")
        lines.append(f"People: {_join_value(frame.get('people_present'))}")
        lines.append(f"Actions: {_join_value(frame.get('actions_occurring'))}")
        lines.append(f"Environment: {_join_value(frame.get('environment_description'))}")
        lines.append("")

    lines.append("Chronological summary")
    lines.append("")
    lines.append(_join_value(summary.get("timeline_of_events")))
    lines.append("")

    lines.append("Overall activity summary")
    lines.append("")
    lines.append(_join_value(summary.get("overall_activity_summary")))
    lines.append("")

    lines.append("Key observations")
    lines.append("")
    lines.append(_join_value(summary.get("key_observations")))
    lines.append("")

    lines.append("Final activity description")
    lines.append("")
    lines.append(_join_value(summary.get("final_activity_description")))
    lines.append("")

    return "\n".join(lines)


def _join_value(value: Any) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, list):
        return "\n".join(f"- {_join_inline(item)}" for item in value) if value else "Not available"
    if isinstance(value, dict):
        return "\n".join(f"{key}: {item}" for key, item in value.items()) if value else "Not available"
    return str(value)


def _join_inline(value: Any) -> str:
    if isinstance(value, dict):
        parts = [str(item) for item in value.values() if item not in (None, "")]
        return " - ".join(parts) if parts else "Not available"
    return str(value)


def _fallback_summary(frame_results: list[dict[str, Any]]) -> dict[str, Any]:
    timeline = []
    actions: list[str] = []
    people: list[str] = []
    environments: list[str] = []
    descriptions: list[str] = []

    for frame in frame_results:
        frame_number = frame.get("frame", "")
        description = str(frame.get("description") or "No clear event described.")
        timeline.append({"frame": frame_number, "event": description})
        descriptions.append(description)
        actions.extend(_as_list(frame.get("actions_occurring")))
        people.extend(_as_list(frame.get("people_present")))
        environments.extend(_as_list(frame.get("environment_description")))

    unique_actions = _unique(actions)
    unique_people = _unique(people)
    unique_environments = _unique(environments)

    return {
        "timeline_of_events": timeline,
        "overall_activity_summary": " ".join(descriptions),
        "key_observations": [
            f"People observed: {', '.join(unique_people)}" if unique_people else "People are visible in the scene.",
            f"Actions observed: {', '.join(unique_actions)}" if unique_actions else "Actions are visible across the extracted frames.",
            f"Environment: {', '.join(unique_environments)}" if unique_environments else "The setting is visible across the extracted frames.",
        ],
        "final_activity_description": descriptions[-1] if descriptions else "No activity description available.",
    }


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip() and str(item).lower() != "none"]
    if isinstance(value, str) and value.strip() and value.lower() != "none":
        return [value]
    return []


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result
