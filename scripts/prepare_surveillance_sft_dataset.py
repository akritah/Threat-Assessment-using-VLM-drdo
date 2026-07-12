import csv
import json
import logging
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
local_root = PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Ground-truth action mappings for XD-Violence domain alignment
ACTION_MAPPINGS = {
    "Abuse": "A person is performing physical abuse and assault on an individual.",
    "CarAccident": "A car accident or vehicle collision is taking place, causing damage.",
    "Explosion": "An explosion is occurring, destroying property and throwing debris.",
    "Fighting": "Multiple individuals are physically fighting, punching, and wrestling.",
    "Normal": "A normal surveillance scene showing regular street and pedestrian activity.",
    "Riot": "A riot or violent public demonstration is occurring, with individuals throwing objects.",
    "Shooting": "An active shooter is firing a weapon."
}

THREAT_REASONING = {
    "Abuse": (
        "**What is happening?**\nAn act of physical violence and abuse is occurring between individuals.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nActive physical abuse causes direct physical harm and requires security intervention."
    ),
    "CarAccident": (
        "**What is happening?**\nA vehicle collision or car accident is visible, showing damaged vehicles.\n\n"
        "**Threat Level**\nMedium\n\n"
        "**Reasoning**\nAccidents threaten public safety and require police and emergency medical dispatch."
    ),
    "Explosion": (
        "**What is happening?**\nA severe explosion is visible, generating smoke, fire, and structural damage.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nExplosions present catastrophic threats to life and property, requiring fire rescue and police."
    ),
    "Fighting": (
        "**What is happening?**\nMultiple individuals are engaged in a physical brawl or group fight.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nFighting threatens public order and safety, requiring police patrol dispatch."
    ),
    "Normal": (
        "**What is happening?**\nPedestrians and vehicles are moving through the frame normally without incident.\n\n"
        "**Threat Level**\nLow\n\n"
        "**Reasoning**\nNo threat indicators, suspicious behaviors, or hostile events are detected."
    ),
    "Riot": (
        "**What is happening?**\nA group of rioters is committing acts of violence and public destruction.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nRiots present severe public safety risks and widespread destruction, requiring SWAT or riot police containment."
    ),
    "Shooting": (
        "**What is happening?**\nAn individual is brandishing a firearm and active shooting is underway.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nActive shooting is a lethal threat to public safety, requiring immediate armed SWAT containment."
    )
}

SUSPICIOUS_INDICATORS = {
    "Abuse": "Yes. Aggressive posture and hostile physical contact.",
    "CarAccident": "Yes. Vehicle collision, sudden deceleration, and wreckage.",
    "Explosion": "Yes. Active fire, blast wave, and rising black smoke.",
    "Fighting": "Yes. Altercation with multiple actors throwing punches.",
    "Normal": "No. Normal everyday activity.",
    "Riot": "Yes. Group violence, vandalism, and throwing projectiles.",
    "Shooting": "Yes. Firearm active discharge and public panic."
}

EMERGENCY_DISPATCH = {
    "Abuse": "Police patrol and local security units should be dispatched.",
    "CarAccident": "Emergency medical services and traffic police should be dispatched.",
    "Explosion": "Fire rescue, SWAT units, and medical dispatch required.",
    "Fighting": "Police patrol units should be dispatched to clear the brawl.",
    "Normal": "No dispatch required. Normal operations.",
    "Riot": "SWAT units, riot police, and emergency medical teams required.",
    "Shooting": "SWAT units and emergency medical teams required."
}

def extract_video_frame(video_path: Path, output_image_path: Path) -> bool:
    """Extract the midpoint frame from a video clip using OpenCV."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return False
            
        midpoint = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, midpoint)
        ret, frame = cap.read()
        if ret:
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_image_path), frame)
            cap.release()
            return True
        cap.release()
    except Exception as e:
        logger.warning(f"Error extracting frame from {video_path}: {e}")
    return False

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="XD-Violence SFT Dataset Generator")
    parser.add_argument("--dataset-dir", default="datasets", help="Path containing test/ and train/ XD folders")
    parser.add_argument("--output-dir", default="training/data", help="Output path for JSONL files")
    parser.add_argument("--max-videos", type=int, default=500, help="Maximum unique videos to sample for training")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("Initializing XD-Violence SFT Dataset Generator...")
    
    dataset_root = Path(args.dataset_dir)
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root
        
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    # XD-Violence structure: train/ and test/ splits (case-insensitive checks)
    train_root = dataset_root / "train"
    if not train_root.exists():
        train_root = dataset_root / "Train"
    if not train_root.exists():
        train_root = local_root / "datasets" / "train"
        
    if not train_root.exists():
        logger.warning(f"Train directory not found at {train_root}. Generating conceptual/mock list.")
        mock_entries = []
        for cat in ACTION_MAPPINGS.keys():
            mock_entries.append((local_root / "datasets" / "train" / cat / "mock_video.mp4", cat))
        video_files = mock_entries
    else:
        # Scan category directories lazily
        import os
        video_files = []
        logger.info("Scanning train split directories lazily...")
        
        for category_dir in train_root.iterdir():
            if not category_dir.is_dir():
                continue
            category = category_dir.name
            logger.info(f"Scanning train category: {category}...")
            
            count = 0
            with os.scandir(category_dir) as it:
                for entry in it:
                    if entry.is_file() and entry.name.lower().endswith(('.mp4', '.avi', '.mkv', '.webm')):
                        # Stop scanning this folder once we have 80 videos (for even category balance)
                        if count >= 80:
                            break
                        video_files.append((Path(entry.path), category))
                        count += 1
                        
    # Sample unique videos up to max_videos
    random.seed(42)
    random.shuffle(video_files)
    selected_videos = video_files[:args.max_videos]
    
    logger.info(f"Selected {len(selected_videos)} videos for dataset generation. Extracting frames...")
    
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    sft_dataset = []
    success_count = 0
    
    for video_path, category in selected_videos:
        # For mock fallback check
        if "mock_video" in video_path.name:
            # Generate empty mock frame
            frame_path = frames_dir / f"{video_path.stem}_{category}.jpg"
            frame_path.touch()
            frame_path_str = str(frame_path).replace("\\", "/")
        else:
            frame_path = frames_dir / f"{video_path.stem}.jpg"
            # Extract midpoint frame
            if not extract_video_frame(video_path, frame_path):
                # Skip video if frame extraction fails
                continue
            frame_path_str = str(frame_path).replace("\\", "/")
            success_count += 1
            
        action_caption = ACTION_MAPPINGS.get(category, ACTION_MAPPINGS["Normal"])
        reasoning_report = THREAT_REASONING.get(category, THREAT_REASONING["Normal"])
        suspicious_ans = SUSPICIOUS_INDICATORS.get(category, SUSPICIOUS_INDICATORS["Normal"])
        dispatch_ans = EMERGENCY_DISPATCH.get(category, EMERGENCY_DISPATCH["Normal"])
        
        # Task A: Action Captioning
        sft_dataset.append({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame_path_str},
                        {"type": "text", "text": "Describe the exact activity happening in this video sequence as a concise caption."}
                    ]
                },
                {"role": "assistant", "content": [{"type": "text", "text": action_caption}]}
            ]
        })
        
        # Task B: Threat Report
        sft_dataset.append({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame_path_str},
                        {"type": "text", "text": "Analyze this surveillance scene video sequence.\nDescribe:\n* What is happening?\n* Estimate the threat level as Low, Medium, or High.\n* Explain your reasoning."}
                    ]
                },
                {"role": "assistant", "content": [{"type": "text", "text": reasoning_report}]}
            ]
        })
        
        # Task C: Suspicious indicators
        sft_dataset.append({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame_path_str},
                        {"type": "text", "text": "Are there any suspicious elements, weapons, or abnormal behaviors in this video sequence? Describe them."}
                    ]
                },
                {"role": "assistant", "content": [{"type": "text", "text": suspicious_ans}]}
            ]
        })
        
        # Task D: Emergency dispatch
        sft_dataset.append({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame_path_str},
                        {"type": "text", "text": "Which emergency services should be notified for this surveillance scene?"}
                    ]
                },
                {"role": "assistant", "content": [{"type": "text", "text": dispatch_ans}]}
            ]
        })
        
    random.shuffle(sft_dataset)
    split_idx = int(len(sft_dataset) * 0.8)
    train_set = sft_dataset[:split_idx]
    val_set = sft_dataset[split_idx:]
    
    train_file = output_dir / "surveillance_train.jsonl"
    val_file = output_dir / "surveillance_val.jsonl"
    
    with train_file.open("w", encoding="utf-8") as f:
        for item in train_set:
            f.write(json.dumps(item) + "\n")
    with val_file.open("w", encoding="utf-8") as f:
        for item in val_set:
            f.write(json.dumps(item) + "\n")
            
    logger.info("=========================================")
    logger.info("XD-Violence SFT Dataset Generation Complete!")
    logger.info(f" - Extracted frames successfully: {success_count}/{len(selected_videos)}")
    logger.info(f" - Train samples: {len(train_set)}")
    logger.info(f" - Validation samples: {len(val_set)}")
    logger.info(f" - Total dataset size: {len(sft_dataset)} samples")
    try:
        logger.info(f" - Outputs written to: {output_dir.relative_to(PROJECT_ROOT)}")
    except ValueError:
        logger.info(f" - Outputs written to: {output_dir}")
    logger.info("=========================================")

if __name__ == "__main__":
    main()
