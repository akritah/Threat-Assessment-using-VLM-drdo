"""Shared prompt templates for inference and fine-tuning."""

FRAME_ANALYSIS_PROMPT = (
    "Analyze only what is visible in this CCTV/video frame. "
    "Return JSON with these keys: objects_present, people_present, actions_occurring, "
    "environment_description, description. "
    "Mention suspicious activity only if it is visible. Do not describe documents or forms unless they are visible."
)

SUMMARY_PROMPT_PREFIX = (
    "Create a JSON summary from these chronological CCTV frame observations. "
    "Return one JSON object only. Do not return an empty object. "
    "Use exactly these keys: timeline_of_events, overall_activity_summary, key_observations, final_activity_description. "
    "timeline_of_events and key_observations must be arrays of short strings. "
    "overall_activity_summary and final_activity_description must be short natural strings.\n\n"
)


def build_summary_prompt(frame_results_json: str) -> str:
    return SUMMARY_PROMPT_PREFIX + frame_results_json
