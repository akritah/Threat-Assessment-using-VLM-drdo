import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import env_loader
env_loader.load_env()

import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

from analyzer_factory import create_analyzer
from frame_extractor import extract_frames
from summarizer import build_text_report, summarize_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline video activity understanding with OpenCV and Ollama")
    parser.add_argument("--video", required=True, help="Path to a local .mp4, .avi, or .mov video")
    parser.add_argument("--frames", type=int, default=6, help="Number of frames to extract")
    parser.add_argument("--model", default="gemma4", help="Ollama model name")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--output-dir", default="outputs", help="Directory for frames and reports")
    parser.add_argument(
        "--backend",
        choices=["ollama", "hf"],
        default=None,
        help="Inference backend (default: ollama — existing behavior)",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="LoRA adapter name when using HF backend (e.g. threat_assessment)",
    )
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

        # Retrieve video duration using OpenCV
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_sec = total_frames / fps if fps > 0 else None
        cap.release()

        # Build timeline and threat metrics
        from threat_engine import (
            generate_timestamps,
            build_event_timeline,
            export_timeline,
            calculate_threat_metrics,
            generate_explainable_report,
            infer_threat_level
        )
        
        timestamps = generate_timestamps(len(frame_paths), duration_sec)
        timeline = build_event_timeline(frame_results, timestamps)
        
        overall_text = str(video_summary.get("overall_activity_summary", ""))
        final_text = str(video_summary.get("final_activity_description", ""))
        combined_summary_text = f"{overall_text} {final_text}"
        
        threat_level = infer_threat_level(combined_summary_text)
        metrics = calculate_threat_metrics(threat_level, frame_results, combined_summary_text)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export timelines (timeline.json, timeline.csv, timeline.md)
        export_timeline(timeline, output_dir, prefix="timeline")
        
        # Export explainable reports (report.json, report.md)
        generate_explainable_report(
            video_name=video_path.name,
            action_caption=overall_text,
            threat_report=final_text,
            threat_level=threat_level,
            metrics=metrics,
            timeline=timeline,
            frame_results=frame_results,
            output_dir=output_dir,
            prefix="report"
        )

        # Legacy files for compatibility
        report_data = {
            "video": str(video_path),
            "model": args.model,
            "frames": frame_results,
            "summary": video_summary,
            "threat_assessment": {
                "threat_level": threat_level,
                "threat_score": metrics["threat_score"],
                "evidence_strength": metrics["evidence_strength"],
                "model_confidence": metrics["model_confidence"]
            }
        }

        report_json_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
        report_txt_path.write_text(build_text_report(frame_results, video_summary), encoding="utf-8")

        logging.info("Chronological event timeline saved to: %s", output_dir / "timeline.json")
        logging.info("Explainable reports generated successfully in: %s", output_dir)
        return 0
    except Exception as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
