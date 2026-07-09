import csv
import json
import logging
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
local_root = PROJECT_ROOT
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

SUSPICIOUS_INDICATORS = {
    "Abuse": "Yes. Hostile physical contact, aggressive posture, and an individual attempting to overpower another.",
    "CarAccident": "Yes. Severe vehicular impact, smoke, and vehicle deformation or overturning on the road.",
    "Explosion": "Yes. Active blast wave, fire outbreak, rising black smoke, and flying structural debris.",
    "Fighting": "Yes. Multiple actors engaging in aggressive physical blows, wrestling, and violent contact.",
    "Riot": "Yes. Crowds throwing objects, active vandalism of public property, and chaotic group movements.",
    "Shooting": "Yes. Active brandishing of a firearm, shooting gestures, and people fleeing in panic.",
    "Normal": "No. Ordinary pedestrian movement, normal vehicle flow, and calm public environments."
}

EMERGENCY_DISPATCH = {
    "Abuse": "Police patrol and local security units should be dispatched immediately to de-escalate violence.",
    "CarAccident": "Traffic police, medical emergency teams (ambulance), and fire rescue should be dispatched.",
    "Explosion": "Fire rescue, SWAT units, emergency medical services, and structural safety inspectors are required.",
    "Fighting": "Security personnel and police units must be dispatched to break up the altercation.",
    "Riot": "Anti-riot police (crowd control), backup security forces, and emergency medical teams are required.",
    "Shooting": "Tactical SWAT teams, armed police units, and medical trauma teams must be dispatched immediately.",
    "Normal": "No emergency dispatch is required. Continue routine CCTV monitoring."
}


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Surveillance SFT Dataset Generator")
    parser.add_argument("--dataset-dir", default="datasets/XD-Violence", help="Path to input video folders")
    parser.add_argument("--output-dir", default="training/data", help="Output path for JSONL files")
    parser.add_argument("--extracted-dir", default="evaluation/extracted_frames", help="Path containing extracted JPG frames")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("Initializing Surveillance SFT Dataset Generator...")
    
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = PROJECT_ROOT / dataset_dir
        
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
        
    extracted_dir = Path(args.extracted_dir)
    if not extracted_dir.is_absolute():
        extracted_dir = PROJECT_ROOT / extracted_dir
        
    # 1. Scan for local videos to generate grounded SFT samples
    all_videos = list(dataset_dir.glob("**/*.mp4"))
    
    if not all_videos:
        logger.warning(f"No videos found under {dataset_dir}. Generating conceptual dataset sample list.")
        # Create a mock list for testing if folders don't exist yet
        mock_videos = []
        for cat in ACTION_MAPPINGS.keys():
            mock_videos.append((f"{cat}_sample_video_01", cat, f"datasets/XD-Violence/{cat}/{cat}_sample.mp4"))
        video_entries = mock_videos
    else:
        video_entries = []
        for idx, v in enumerate(all_videos, 1):
            category = v.parent.name
            try:
                rel_path = v.relative_to(PROJECT_ROOT)
            except ValueError:
                rel_path = v
            video_entries.append((v.stem, category, str(rel_path).replace("\\", "/")))
            
    logger.info(f"Processing {len(video_entries)} videos for multi-task SFT...")
    
    sft_dataset = []
    
    # 2. Generate Multi-Task SFT samples for each video
    # We mix (A) Concise Action Captioning and (B) Detailed Threat Reports (Instruction Replay)
    for vid, category, rel_path in video_entries:
        action_caption = ACTION_MAPPINGS.get(category, ACTION_MAPPINGS["Normal"])
        reasoning_report = THREAT_REASONING.get(category, THREAT_REASONING["Normal"])
        
        # Locate the extracted midpoint frame image
        frame_dir = extracted_dir / vid
        jpg_files = list(frame_dir.glob("*.jpg"))
        if jpg_files:
            jpg_files.sort()
            # Pick the middle frame to represent the video segment
            midpoint_jpg = jpg_files[len(jpg_files) // 2]
            try:
                frame_path_str = str(midpoint_jpg.relative_to(PROJECT_ROOT)).replace("\\", "/")
            except ValueError:
                frame_path_str = str(midpoint_jpg).replace("\\", "/")
        else:
            # Fallback path if frames are not yet extracted locally
            frame_path_str = f"evaluation/extracted_frames/{vid}/frame_04.jpg"
        
        # Task A: Action Captioning SFT Sample (Domain Alignment)
        task_a_sample = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame_path_str},
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
                        {"type": "image", "image": frame_path_str},
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

        # Task C: Suspicious Element Query SFT Sample (Domain Alignment)
        suspicious_ans = SUSPICIOUS_INDICATORS.get(category, SUSPICIOUS_INDICATORS["Normal"])
        task_c_sample = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame_path_str},
                        {"type": "text", "text": "Are there any suspicious elements, weapons, or abnormal behaviors in this video sequence? Describe them."}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": suspicious_ans}
                    ]
                }
            ]
        }
        sft_dataset.append(task_c_sample)

        # Task D: Emergency Action Dispatch SFT Sample (Instruction Replay)
        dispatch_ans = EMERGENCY_DISPATCH.get(category, EMERGENCY_DISPATCH["Normal"])
        task_d_sample = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame_path_str},
                        {"type": "text", "text": "Which emergency services should be notified for this surveillance scene?"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": dispatch_ans}
                    ]
                }
            ]
        }
        sft_dataset.append(task_d_sample)

    # 3. Shuffle and split into Train (80%) and Val (20%)
    random.seed(42)
    random.shuffle(sft_dataset)
    
    split_idx = int(len(sft_dataset) * 0.8)
    train_set = sft_dataset[:split_idx]
    val_set = sft_dataset[split_idx:]
    
    # 4. Save to JSONL files
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = output_dir / "surveillance_train.jsonl"
    val_file = output_dir / "surveillance_val.jsonl"
    
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
    try:
        logger.info(f" - Outputs written to: {output_dir.relative_to(PROJECT_ROOT)}")
    except ValueError:
        logger.info(f" - Outputs written to: {output_dir}")
    logger.info("=========================================")

if __name__ == "__main__":
    main()
