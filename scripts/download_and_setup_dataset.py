import fiftyone.zoo as foz
import json
import cv2
import logging
from pathlib import Path
import os
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading FiftyOne ActivityNet validation dataset (50 samples)...")
    try:
        # Load 50 samples of validation split
        dataset = foz.load_zoo_dataset(
            "activitynet-200",
            split="validation",
            max_samples=50,
        )
        dataset.persistent = True
    except Exception as e:
        logger.error(f"Failed to load dataset from FiftyOne zoo: {e}")
        return

    database = {}
    
    # We will search for where FiftyOne saves the videos
    videos_dir = None
    
    for sample in dataset:
        video_path = Path(sample.filepath)
        video_id = video_path.stem
        
        if videos_dir is None:
            videos_dir = video_path.parent
            logger.info(f"Detected FiftyOne videos directory: {videos_dir}")
            
        # Get FPS and duration
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Failed to open video: {video_path}")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0:
            fps = 29.97
        duration = frame_count / fps
        cap.release()
        
        timestamps = []
        sentences = []
        
        # Get detections
        ground_truth = sample.ground_truth
        if ground_truth and ground_truth.detections:
            for det in ground_truth.detections:
                label = det.label
                support = det.support
                if len(support) == 2:
                    start_frame, end_frame = support
                    start_sec = start_frame / fps
                    end_sec = end_frame / fps
                    timestamps.append([start_sec, end_sec])
                    sentences.append(f"A person is performing {label.lower()}.")
                    
        if timestamps:
            database[video_id] = {
                "duration": duration,
                "timestamps": timestamps,
                "sentences": sentences,
                "subset": "validation"
            }
            
    # Save annotations to a json file
    output_annotations = project_root / "training" / "data" / "fiftyone_activitynet.json"
    output_annotations.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_annotations, "w") as f:
        json.dump({"database": database}, f, indent=2)
        
    logger.info(f"Saved {len(database)} annotations to {output_annotations}")
    
    if not videos_dir or not videos_dir.exists():
        logger.error("Could not determine videos directory or directory does not exist.")
        return
        
    output_dir = project_root / "training" / "data"
    
    logger.info("Extracting frames and creating train/eval splits...")
    from training.preprocess_activitynet import process_activitynet
    process_activitynet(
        annotations_path=str(output_annotations),
        videos_dir=videos_dir,
        output_dir=output_dir,
        split_ratio=0.0,  # All 50 samples will go to eval.jsonl
        seed=42
    )
    
    logger.info("Dataset setup completed successfully!")

if __name__ == "__main__":
    main()
