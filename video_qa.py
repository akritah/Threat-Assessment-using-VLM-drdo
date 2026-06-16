"""Interactive Video Q&A Agent with Safety/Emergency Recommendation capabilities.

Queries the video analysis report (either pre-existing or generated on the fly)
to answer user questions and suggest response protocols for emergency cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import logging
from typing import Any

from tqdm import tqdm

from analyzer_factory import create_analyzer
from frame_extractor import extract_frames
from summarizer import summarize_video

logger = logging.getLogger(__name__)


def format_context(report_data: dict[str, Any]) -> str:
    """Format frame-by-frame details and summary as a single text context."""
    lines = []
    summary = report_data.get("summary", {})
    if summary:
        lines.append(f"Overall Activity Summary: {summary.get('overall_activity_summary', '')}")
        lines.append(f"Final Activity Description: {summary.get('final_activity_description', '')}")
        if summary.get("key_observations"):
            lines.append("Key Observations:")
            for obs in summary["key_observations"]:
                lines.append(f" - {obs}")

    lines.append("\nChronological Frame-by-Frame Log:")
    for frame in report_data.get("frames", []):
        num = frame.get("frame")
        desc = frame.get("description", "")
        objects = frame.get("objects_present", [])
        people = frame.get("people_present", [])
        actions = frame.get("actions_occurring", [])
        env = frame.get("environment_description", "")

        lines.append(f"Frame {num} (Image: {frame.get('image', 'N/A')}):")
        lines.append(f"  Environment: {env}")
        lines.append(f"  Description: {desc}")
        if objects:
            lines.append(f"  Objects: {', '.join(objects)}")
        if people:
            lines.append(f"  People: {', '.join(people)}")
        if actions:
            lines.append(f"  Actions: {', '.join(actions)}")

    return "\n".join(lines)


def ask_analyzer(analyzer: Any, prompt: str) -> str:
    """Send prompt to the loaded inference analyzer (Ollama or HF)."""
    # Hugging Face backend
    if hasattr(analyzer, "_model") and hasattr(analyzer, "_processor"):
        messages = [{"role": "user", "content": prompt}]
        return analyzer._generate(messages)
    # Ollama backend
    else:
        return analyzer._generate(prompt=prompt)


def build_qa_prompt(video_context: str, question: str) -> str:
    """Build the prompt instructing the model to answer and recommend safety measures."""
    return (
        "You are an AI security assistant analyzing a video activity log.\n\n"
        "Here is the chronological frame-by-frame analysis of the video:\n"
        "=========================================\n"
        f"{video_context}\n"
        "=========================================\n\n"
        f"User Question: {question}\n\n"
        "Instructions:\n"
        "1. Answer the user's question clearly and concisely based on the video context.\n"
        "2. Safety Action Protocol: If the question asks what to do, OR if the video/question "
        "indicates an emergency, crime, fire, theft, accident, medical situation, or suspicious "
        "security threat, you MUST explicitly suggest appropriate responses. Specifically:\n"
        "   - Call the Police (112, 911, or local emergency) if you observe crimes, fights, unauthorized entries, "
        "or suspicious security threats.\n"
        "   - Call an Ambulance / Medical Help if a person appears injured, unconscious, or in medical distress.\n"
        "   - Contact Fire Services if you see fire, smoke, or potential explosions.\n"
        "   - Recommend first-aid, evacuation, or safety precautions when relevant.\n\n"
        "Response:"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Video Q&A Agent")
    parser.add_argument("--video", help="Path to a local video file to analyze")
    parser.add_argument(
        "--report",
        default="outputs/report.json",
        help="Path to pre-generated report JSON file",
    )
    parser.add_argument("--frames", type=int, default=6, help="Number of frames to extract if analyzing video")
    parser.add_argument("--model", default="gemma3:4b", help="Ollama model name")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument(
        "--backend",
        choices=["ollama", "hf"],
        default=None,
        help="Inference backend (default: ollama)",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="LoRA adapter name when using HF backend",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    report_path = Path(args.report)
    report_data = None

    # Load pre-existing report if it exists and no video is explicitly provided
    if report_path.exists() and not args.video:
        logger.info("Loading pre-generated report from: %s", report_path)
        try:
            with report_path.open(encoding="utf-8") as handle:
                report_data = json.load(handle)
        except Exception as exc:
            logger.error("Failed to load report file: %s", exc)
            return 1
    elif args.video:
        # Generate analysis on the fly
        video_path = Path(args.video).resolve()
        if not video_path.exists():
            logger.error("Video file not found: %s", video_path)
            return 1

        output_dir = report_path.parent
        frames_dir = output_dir / "frames"

        try:
            logger.info("Extracting %d frames from video...", args.frames)
            frame_paths = extract_frames(video_path, frames_dir, args.frames)

            logger.info("Initializing analyzer (backend: %s)...", args.backend or "default")
            analyzer = create_analyzer(
                model=args.model,
                base_url=args.ollama_url,
                backend=args.backend,
                adapter=args.adapter,
            )

            frame_results = []
            for index, frame_path in enumerate(tqdm(frame_paths, desc="Analyzing frames"), start=1):
                frame_results.append(analyzer.analyze_frame(frame_path, index))

            video_summary = summarize_video(analyzer, frame_results)

            report_data = {
                "video": str(video_path),
                "model": args.model,
                "frames": frame_results,
                "summary": video_summary,
            }

            # Save the report for future QA sessions
            output_dir.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Analysis complete. Report saved to: %s", report_path)

        except Exception as exc:
            logger.error("Failed to analyze video: %s", exc, exc_info=True)
            return 1
    else:
        logger.error("Please provide either --video or a valid pre-generated report via --report.")
        return 1

    # Format the context from video data
    video_context = format_context(report_data)

    # Load analyzer for Q&A sessions
    try:
        logger.info("Loading Q&A engine...")
        analyzer = create_analyzer(
            model=args.model,
            base_url=args.ollama_url,
            backend=args.backend,
            adapter=args.adapter,
        )
    except Exception as exc:
        logger.error("Failed to load analyzer: %s", exc)
        return 1

    print("\n" + "=" * 60)
    print(" Video Q&A Assistant Loaded Successfully! ")
    print(" You can ask questions about the video contents.")
    print(" Type 'exit' or 'quit' to close the session.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("Q: ").strip()
            if not question:
                continue
            if question.lower() in ["exit", "quit"]:
                print("Exiting Video Q&A Session. Goodbye!")
                break

            prompt = build_qa_prompt(video_context, question)
            print("\nAnalyzing...", end="", flush=True)

            response = ask_analyzer(analyzer, prompt)

            print("\r" + " " * 12 + "\r", end="", flush=True)  # Clear 'Analyzing...'
            print(f"A: {response}\n")
            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\nExiting Video Q&A Session. Goodbye!")
            break
        except Exception as exc:
            print(f"\nError processing question: {exc}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
