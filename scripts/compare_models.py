"""Compare base Ollama/HF model output against a fine-tuned adapter on video frames.

Runs the existing frame extraction pipeline twice (base vs adapter) and writes
a side-by-side comparison report.
"""

from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
import env_loader
env_loader.load_env()

import argparse
import json
import logging

from analyzer_factory import create_analyzer
from frame_extractor import extract_frames

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def compare_on_video(
    video_path: Path,
    output_dir: Path,
    *,
    frames: int = 6,
    hf_backend: bool = True,
    adapter: str | None = None,
) -> dict:
    frames_dir = output_dir / "frames"
    frame_paths = extract_frames(video_path, frames_dir, frames)

    if hf_backend:
        base_analyzer = create_analyzer(backend="hf", adapter=None)
        finetuned_analyzer = create_analyzer(backend="hf", adapter=adapter)
        base_label = "hf_base"
        finetuned_label = f"hf_adapter_{adapter or 'none'}"
    else:
        base_analyzer = create_analyzer(backend="ollama")
        finetuned_analyzer = base_analyzer
        base_label = "ollama_base"
        finetuned_label = "ollama_base"

    comparisons = []
    for index, frame_path in enumerate(frame_paths, start=1):
        base_result = base_analyzer.analyze_frame(frame_path, index)
        finetuned_result = finetuned_analyzer.analyze_frame(frame_path, index)
        comparisons.append(
            {
                "frame": index,
                "image": str(frame_path),
                "base": base_result,
                "finetuned": finetuned_result,
            }
        )

    return {
        "video": str(video_path),
        "base_model": base_label,
        "finetuned_model": finetuned_label,
        "comparisons": comparisons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned model on video frames")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--frames", type=int, default=6, help="Number of frames to extract")
    parser.add_argument("--adapter", required=True, help="Adapter name (e.g. threat_assessment)")
    parser.add_argument(
        "--output",
        default="outputs/model_comparison.json",
        help="Output comparison report path",
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Compare using Ollama backend instead of HuggingFace",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    video_path = Path(args.video).expanduser().resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _PROJECT_ROOT / output_path

    try:
        report = compare_on_video(
            video_path,
            output_path.parent,
            frames=args.frames,
            hf_backend=not args.ollama,
            adapter=args.adapter,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.info("Comparison report saved: %s", output_path)
        return 0
    except Exception as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
