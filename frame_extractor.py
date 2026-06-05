from __future__ import annotations

import logging
from pathlib import Path

import cv2


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov"}


def extract_frames(video_path: Path, output_dir: Path, frame_count: int = 6) -> list[Path]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if video_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Supported video formats are .mp4, .avi, and .mov")

    if frame_count < 1:
        raise ValueError("Frame count must be at least 1")

    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError("Could not read frame count from video")

        count = min(frame_count, total_frames)
        positions = _even_positions(total_frames, count)
        saved_paths: list[Path] = []

        for number, position in enumerate(positions, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if not ok or frame is None:
                logging.warning("Skipping unreadable frame at position %s", position)
                continue

            frame_path = output_dir / f"frame_{number:02d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Could not save frame: {frame_path}")
            saved_paths.append(frame_path)

        if not saved_paths:
            raise RuntimeError("No frames were extracted")

        logging.info("Extracted %s frame(s)", len(saved_paths))
        return saved_paths
    finally:
        capture.release()


def _even_positions(total_frames: int, count: int) -> list[int]:
    if count == 1:
        return [max(0, total_frames // 2)]

    last = total_frames - 1
    return [round(i * last / (count - 1)) for i in range(count)]
