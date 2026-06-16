"""Preprocessing script for ActivityNet Captions.

Extracts representative frames from video files corresponding to annotated segments
and formats them in the image-instruction-response JSONL schema for training.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)


def find_video_file(videos_dir: Path, video_id: str) -> Path | None:
    """Find a video file matching the video ID with various extensions."""
    extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    # ActivityNet IDs often have "v_" prefix or not. Check both.
    prefixes_and_names = [
        video_id,
        f"v_{video_id}",
        video_id.replace("v_", ""),
    ]

    for name in prefixes_and_names:
        for ext in extensions:
            candidate = videos_dir / f"{name}{ext}"
            if candidate.exists():
                return candidate

    return None


def extract_frame_at_time(video_path: Path, time_sec: float, output_path: Path) -> bool:
    """Extract a frame from a video at a specific time in seconds."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Failed to open video: %s", video_path)
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 29.97

    # Try setting position via frame index first
    frame_idx = int(time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()

    # Fallback to milliseconds if frame setting fails or returns empty
    if not success or frame is None:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
        success, frame = cap.read()

    if success and frame is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)
    else:
        logger.warning("Failed to extract frame at %.2fs from %s", time_sec, video_path)
        success = False

    cap.release()
    return success


def process_activitynet(
    annotations_path: str,
    videos_dir: Path,
    output_dir: Path,
    split_ratio: float = 0.9,
    num_samples: int | None = None,
    seed: int = 42,
) -> None:
    """Loads annotations (local or HF Hub), extracts frames, formats data, and splits into train/eval jsonl."""
    # Check if annotations is a Hugging Face dataset identifier
    if "/" in annotations_path and not Path(annotations_path).exists():
        from datasets import load_dataset

        logger.info("Loading annotations from Hugging Face Hub: %s", annotations_path)
        ds = load_dataset(annotations_path)
        database: dict[str, Any] = {}

        # Parse across all splits present in Hugging Face dataset
        for split in ds.keys():
            for row in ds[split]:
                video_id = row.get("video_id") or row.get("id")
                if not video_id:
                    continue

                timestamps = row.get("timestamps", [])
                sentences = row.get("sentences", [])

                # Check if Hugging Face represents row-per-segment (flat list of 2 elements)
                if (
                    isinstance(timestamps, list)
                    and len(timestamps) > 0
                    and not isinstance(timestamps[0], list)
                ):
                    if video_id not in database:
                        database[video_id] = {"timestamps": [], "sentences": []}
                    database[video_id]["timestamps"].append(timestamps)
                    database[video_id]["sentences"].append(row.get("sentence", ""))
                else:
                    # Hugging Face represents row-per-video (list of list of timestamps)
                    database[video_id] = {
                        "duration": row.get("duration"),
                        "timestamps": timestamps,
                        "sentences": sentences,
                    }
    else:
        # Local JSON file path
        path = Path(annotations_path)
        logger.info("Loading annotations from local path: %s", path)
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        database = data.get("database", data)

    records: list[dict[str, str]] = []
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    video_ids = list(database.keys())
    random.seed(seed)
    random.shuffle(video_ids)

    processed_count = 0
    skipped_count = 0

    for video_id in tqdm(video_ids, desc="Processing videos"):
        if num_samples and processed_count >= num_samples:
            logger.info("Reached target limit of %d samples. Stopping.", num_samples)
            break

        video_info = database[video_id]
        video_file = find_video_file(videos_dir, video_id)

        if not video_file:
            skipped_count += 1
            continue

        timestamps = video_info.get("timestamps", [])
        sentences = video_info.get("sentences", [])

        for i, (span, sentence) in enumerate(zip(timestamps, sentences)):
            if len(span) < 2:
                continue

            start, end = span[0], span[1]
            midpoint = (start + end) / 2.0

            frame_filename = f"{video_id}_segment_{i}.jpg"
            frame_path = frames_dir / frame_filename

            # Extract frame at segment midpoint
            if extract_frame_at_time(video_file, midpoint, frame_path):
                # Save relative path to output directory for portability
                relative_frame_path = Path("training/data/frames") / frame_filename
                records.append({
                    "image": relative_frame_path.as_posix(),
                    "instruction": "Describe what is happening in this scene.",
                    "response": sentence.strip(),
                })
                processed_count += 1

    logger.info(
        "Extracted %d samples. Skipped %d videos (not found in directory).",
        len(records),
        skipped_count,
    )

    if not records:
        logger.error("No samples successfully processed! Check if video files match annotations.")
        return

    # Train / Validation Split
    random.shuffle(records)
    split_idx = int(len(records) * split_ratio)
    train_records = records[:split_idx]
    eval_records = records[split_idx:]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write train.jsonl
    train_path = output_dir / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for record in train_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write eval.jsonl
    eval_path = output_dir / "eval.jsonl"
    with eval_path.open("w", encoding="utf-8") as handle:
        for record in eval_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Saved %d training samples to %s", len(train_records), train_path)
    logger.info("Saved %d evaluation samples to %s", len(eval_records), eval_path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="ActivityNet Captions Dataset Preprocessor")
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to annotations JSON file OR Hugging Face dataset name (e.g. friedrichor/ActivityNet_Captions)",
    )
    parser.add_argument("--videos-dir", required=True, help="Directory containing ActivityNet videos")
    parser.add_argument(
        "--output-dir", default="training/data", help="Output directory for frames and JSONL files"
    )
    parser.add_argument("--split-ratio", type=float, default=0.9, help="Train/eval split ratio (default: 0.9)")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit number of processed segments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility")

    args = parser.parse_args()

    try:
        process_activitynet(
            annotations_path=args.annotations,
            videos_dir=Path(args.videos_dir),
            output_dir=Path(args.output_dir),
            split_ratio=args.split_ratio,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        return 0
    except Exception as exc:
        logger.error("Error during preprocessing: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
