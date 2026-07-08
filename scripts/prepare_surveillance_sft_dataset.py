import csv
import json
import logging
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Ground-truth action mappings for surveillance domain alignment
ACTION_MAPPINGS = {
    "Abuse": "A person is performing physical abuse and assault on an individual.",
    "CarAccident": "A vehicle collision is happening on a public road.",
    "Explosion": "An explosion is occurring, destroying property and throwing debris.",
    "Fighting": "Multiple individuals are physically fighting, punching, and wrestling.",
    "Riot": "A crowd of people is rioting, throwing objects, and vandalizing street infrastructure.",
    "Shooting": "An active shooter is firing a weapon.",
    "Normal": "A normal surveillance scene showing regular street and pedestrian activity."
}

THREAT_REASONING = {
    "Abuse": (
        "**What is happening?**\nAn act of physical violence and assault is occurring between individuals.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nActive physical assault causes severe bodily harm and requires immediate security/police intervention."
    ),
    "CarAccident": (
        "**What is happening?**\nA vehicle collision has occurred on a road, causing potential traffic blockage and driver injury.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nTraffic collisions pose a risk of life-threatening injuries and fire, necessitating emergency rescue dispatch."
    ),
    "Explosion": (
        "**What is happening?**\nA severe explosion is visible, generating smoke, fire, and structural damage.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nExplosions indicate explosive hazards, active fire, and severe casualties, requiring fire rescue and military/police teams."
    ),
    "Fighting": (
        "**What is happening?**\nMultiple individuals are engaged in a physical brawl or group fight.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nGroup fighting escalates public disorder and results in injury, requiring security dispatch to restore order."
    ),
    "Riot": (
        "**What is happening?**\nA hostile crowd is marching, committing vandalism, and throwing missiles/objects.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nRiots represent large-scale public safety breakdowns and property destruction, requiring immediate anti-riot police deployment."
    ),
    "Shooting": (
        "**What is happening?**\nAn individual is brandishing a firearm and active shooting is underway.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nActive shooting presents a lethal, active threat to all bystanders, requiring immediate tactical SWAT and police containment."
    ),
    "Normal": (
        "**What is happening?**\nPeople and vehicles are moving through the frame normally without incident.\n\n"
        "**Threat Level**\nLow\n\n"
        "**Reasoning**\nNo threat indicators, suspicious activities, or hostile behaviors are detected. The scene is safe."
    )
}

def main():
    logger.info("Initializing Surveillance SFT Dataset Generator...")
    
    # 1. Scan for local videos to generate grounded SFT samples
    dataset_dir = PROJECT_ROOT / "datasets" / "XD-Violence"
    all_videos = list(dataset_dir.glob("**/*.mp4"))
    
    if not all_videos:
        logger.warning("No local XD-Violence videos found under datasets/XD-Violence. Generating conceptual dataset sample list.")
        # Create a mock list for testing if folders don't exist yet
        mock_videos = []
        for cat in ACTION_MAPPINGS.keys():
            mock_videos.append((f"{cat}_sample_video_01", cat, f"datasets/XD-Violence/{cat}/{cat}_sample.mp4"))
        video_entries = mock_videos
    else:
        video_entries = []
        for idx, v in enumerate(all_videos, 1):
            category = v.parent.name
            rel_path = v.relative_to(PROJECT_ROOT)
            video_entries.append((v.stem, category, str(rel_path).replace("\\", "/")))
            
    logger.info(f"Processing {len(video_entries)} videos for multi-task SFT...")
    
    sft_dataset = []
    
    # 2. Generate Multi-Task SFT samples for each video
    # We mix (A) Concise Action Captioning and (B) Detailed Threat Reports (Instruction Replay)
    for vid, category, rel_path in video_entries:
        action_caption = ACTION_MAPPINGS.get(category, ACTION_MAPPINGS["Normal"])
        reasoning_report = THREAT_REASONING.get(category, THREAT_REASONING["Normal"])
        
        # Task A: Action Captioning SFT Sample (Domain Alignment)
        task_a_sample = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the exact activity happening in this video sequence as a concise caption."}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": action_caption}
                    ]
                }
            ]
        }
        sft_dataset.append(task_a_sample)
        
        # Task B: Detailed Structured Threat Report (Instruction Replay / Prevent Collapse)
        task_b_sample = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                            "Analyze this surveillance scene video sequence.\n"
                            "Describe:\n"
                            "* What is happening?\n"
                            "* Estimate the threat level as Low, Medium, or High.\n"
                            "* Explain your reasoning."
                        )}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": reasoning_report}
                    ]
                }
            ]
        }
        sft_dataset.append(task_b_sample)

    # 3. Shuffle and split into Train (80%) and Val (20%)
    random.seed(42)
    random.shuffle(sft_dataset)
    
    split_idx = int(len(sft_dataset) * 0.8)
    train_set = sft_dataset[:split_idx]
    val_set = sft_dataset[split_idx:]
    
    # 4. Save to JSONL files
    data_dir = PROJECT_ROOT / "training" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = data_dir / "surveillance_train.jsonl"
    val_file = data_dir / "surveillance_val.jsonl"
    
    with train_file.open("w", encoding="utf-8") as f:
        for item in train_set:
            f.write(json.dumps(item) + "\n")
            
    with val_file.open("w", encoding="utf-8") as f:
        for item in val_set:
            f.write(json.dumps(item) + "\n")
            
    logger.info("=========================================")
    logger.info("Surveillance SFT Dataset Generation Complete!")
    logger.info(f" - Train samples: {len(train_set)}")
    logger.info(f" - Validation samples: {len(val_set)}")
    logger.info(f" - Total dataset size: {len(sft_dataset)} samples")
    logger.info(f" - Outputs written to: {data_dir.relative_to(PROJECT_ROOT)}")
    logger.info("=========================================")

if __name__ == "__main__":
    main()
