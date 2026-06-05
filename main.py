from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

from frame_extractor import extract_frames
from gemma_analyzer import GemmaAnalyzer
from summarizer import build_text_report, summarize_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline video activity understanding with OpenCV and Ollama")
    parser.add_argument("--video", required=True, help="Path to a local .mp4, .avi, or .mov video")
    parser.add_argument("--frames", type=int, default=6, help="Number of frames to extract")
    parser.add_argument("--model", default="gemma4", help="Ollama model name")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--output-dir", default="outputs", help="Directory for frames and reports")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    video_path = Path(args.video).expanduser().resolve()
    output_dir = Path(args.output_dir).resolve()
    frames_dir = output_dir / "frames"
    report_json_path = output_dir / "report.json"
    report_txt_path = output_dir / "report.txt"

    try:
        frame_paths = extract_frames(video_path, frames_dir, args.frames)
        analyzer = GemmaAnalyzer(model=args.model, base_url=args.ollama_url)

        frame_results = []
        for index, frame_path in enumerate(tqdm(frame_paths, desc="Analyzing frames"), start=1):
            frame_results.append(analyzer.analyze_frame(frame_path, index))

        video_summary = summarize_video(analyzer, frame_results)

        output_dir.mkdir(parents=True, exist_ok=True)
        report_data = {
            "video": str(video_path),
            "model": args.model,
            "frames": frame_results,
            "summary": video_summary,
        }

        report_json_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
        report_txt_path.write_text(build_text_report(frame_results, video_summary), encoding="utf-8")

        logging.info("Report saved: %s", report_txt_path)
        logging.info("JSON saved: %s", report_json_path)
        return 0
    except Exception as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
