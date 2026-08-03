from __future__ import annotations

import logging
from pathlib import Path
import cv2

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov"}

def extract_frames(video_path: Path, output_dir: Path, frame_count: int = 6) -> list[Path]:
    """
    Extracts frames that capture the peak activity while keeping image sharpness high (filtering blur).
    Divides the video into equal segments and picks the optimal motion-sharpness frame from each.
    """
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
        
        # Get motion-and-sharpness aware optimal positions
        positions = _motion_and_sharpness_positions(video_path, total_frames, count)
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

        logging.info("Extracted %s optimized motion-sharpness frame(s)", len(saved_paths))
        return saved_paths
    finally:
        capture.release()

def _motion_and_sharpness_positions(video_path: Path, total_frames: int, count: int) -> list[int]:
    """
    Scans the video to compute temporal motion profiles and image sharpness indexes 
    (using Laplacian variance) to select the highest-quality active frames.
    """
    if count == 1:
        return [max(0, total_frames // 2)]

    capture = cv2.VideoCapture(str(video_path))
    
    # Sample every step-th frame to run this pre-pass at 200+ FPS on CPU
    step = max(1, total_frames // 150)
    frame_data = []
    
    prev_frame = None
    frame_idx = 0
    
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
            
        if frame_idx % step == 0:
            # Resize frame to small resolution (128x128) for fast matrix operations
            small_frame = cv2.resize(frame, (128, 128))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Compute Image Sharpness using Laplacian variance
            # High variance = sharp edges/details; Low variance = blurry images
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            
            # 2. Compute Frame-to-Frame Temporal Motion difference
            gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            if prev_frame is not None:
                diff = cv2.absdiff(gray_blurred, prev_frame)
                motion = float(diff.mean())
            else:
                motion = 0.0
                
            frame_data.append({
                "index": frame_idx,
                "motion": motion,
                "sharpness": sharpness
            })
            prev_frame = gray_blurred
            
        frame_idx += 1
        
    capture.release()
    
    if not frame_data:
        # Fallback to standard uniform positions
        last = total_frames - 1
        return [round(i * last / (count - 1)) for i in range(count)]
        
    # Segment video and optimize for joint motion + sharpness (penalizing motion blur)
    segment_size = total_frames / count
    positions = []
    
    for i in range(count):
        start_idx = int(i * segment_size)
        end_idx = int((i + 1) * segment_size)
        
        seg_items = [item for item in frame_data if start_idx <= item["index"] < end_idx]
        if seg_items:
            max_motion = max(item["motion"] for item in seg_items)
            max_sharpness = max(item["sharpness"] for item in seg_items)
            
            best_idx = seg_items[len(seg_items)//2]["index"]
            best_score = -1.0
            
            for item in seg_items:
                norm_motion = item["motion"] / max_motion if max_motion > 0 else 0.0
                norm_sharpness = item["sharpness"] / max_sharpness if max_sharpness > 0 else 0.0
                
                # Joint Quality Score: 40% motion weighting + 60% sharpness weighting
                # This ensures we favor active movement but strongly reject motion blur
                score = (norm_motion * 0.4) + (norm_sharpness * 0.6)
                
                # If there's virtually no motion in this segment, default entirely to the sharpest frame
                if max_motion < 1.0:
                    score = norm_sharpness
                    
                if score > best_score:
                    best_score = score
                    best_idx = item["index"]
                    
            positions.append(best_idx)
        else:
            positions.append(start_idx + (end_idx - start_idx) // 2)
            
    return sorted(positions)
