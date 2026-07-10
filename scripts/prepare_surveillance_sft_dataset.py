import csv
import json
import logging
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
local_root = PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Ground-truth action mappings for UCF-Crime domain alignment
ACTION_MAPPINGS = {
    "Abuse": "A person is performing physical abuse and assault on an individual.",
    "Arson": "An arsonist is starting a fire inside or outside a building.",
    "Assault": "A physical assault is occurring between individuals.",
    "Burglary": "A burglar is breaking into a property or building.",
    "Explosion": "An explosion is occurring, destroying property and throwing debris.",
    "Fighting": "Multiple individuals are physically fighting, punching, and wrestling.",
    "Robbery": "An armed robbery is happening, threatening victims with a weapon.",
    "Shooting": "An active shooter is firing a weapon.",
    "Shoplifting": "A shoplifter is stealing merchandise from a store retail shelf.",
    "Stealing": "A person is stealing property, a vehicle, or an item.",
    "Vandalism": "An individual is committing vandalism or destroying public property.",
    "Arrest": "Police officers are making an arrest and detaining a suspect.",
    "Normal": "A normal surveillance scene showing regular street and pedestrian activity."
}

THREAT_REASONING = {
    "Abuse": (
        "**What is happening?**\nAn act of physical violence and assault is occurring between individuals.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nActive physical assault causes severe bodily harm and requires immediate security/police intervention."
    ),
    "Arson": (
        "**What is happening?**\nAn individual is intentionally starting a fire on or near a building.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nArson threatens lives and properties with catastrophic fire damage, requiring fire department dispatch."
    ),
    "Assault": (
        "**What is happening?**\nA violent physical confrontation or assault is taking place.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nViolence poses an immediate threat to life, requiring police intervention."
    ),
    "Burglary": (
        "**What is happening?**\nAn unauthorized entry into a locked building or property is visible.\n\n"
        "**Threat Level**\nMedium\n\n"
        "**Reasoning**\nProperty crime is in progress. Law enforcement should be dispatched to apprehend suspects."
    ),
    "Explosion": (
        "**What is happening?**\nA severe explosion is visible, generating smoke, fire, and structural damage.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nExplosions indicate explosive hazards, active fire, and severe casualties, requiring fire rescue and police teams."
    ),
    "Fighting": (
        "**What is happening?**\nMultiple individuals are engaged in a physical brawl or group fight.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nGroup fighting escalates public disorder and results in injury, requiring security dispatch to restore order."
    ),
    "Robbery": (
        "**What is happening?**\nSuspects are threatening individuals with weapons to steal their belongings.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nRobberies involve active weapons and direct coercion, posing an immediate threat to life."
    ),
    "Shooting": (
        "**What is happening?**\nAn individual is brandishing a firearm and active shooting is underway.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nActive shooting presents a lethal, active threat to all bystanders, requiring immediate tactical SWAT containment."
    ),
    "Shoplifting": (
        "**What is happening?**\nAn individual is concealing store merchandise and attempting to leave without paying.\n\n"
        "**Threat Level**\nLow\n\n"
        "**Reasoning**\nNon-violent property crime. Store security should detain the suspect for police handoff."
    ),
    "Stealing": (
        "**What is happening?**\nA theft of a vehicle, bicycle, or property is occurring.\n\n"
        "**Threat Level**\nMedium\n\n"
        "**Reasoning**\nNon-violent property theft. Suspect is fleeing, police should be notified to track the stolen item."
    ),
    "Vandalism": (
        "**What is happening?**\nAn individual is spray-painting walls or destroying public property.\n\n"
        "**Threat Level**\nMedium\n\n"
        "**Reasoning**\nProperty damage in progress. Police should be dispatched to prevent further destruction."
    ),
    "Arrest": (
        "**What is happening?**\nLaw enforcement officers are detaining and handcuffing a suspect.\n\n"
        "**Threat Level**\nLow\n\n"
        "**Reasoning**\nActive police operation. The situation is under control of law enforcement. No threat to the general public."
    ),
    "Normal": (
        "**What is happening?**\nPeople and vehicles are moving through the frame normally without incident.\n\n"
        "**Threat Level**\nLow\n\n"
        "**Reasoning**\nNo threat indicators, suspicious activities, or hostile behaviors are detected. The scene is safe."
    )
}

SUSPICIOUS_INDICATORS = {
    "Abuse": "Yes. Hostile physical contact, aggressive posture, and abuse of an individual.",
    "Arson": "Yes. Starting a fire intentionally near a structure.",
    "Assault": "Yes. Active physical fight, hitting, or kicking.",
    "Burglary": "Yes. Breaking windows/doors or forced entry.",
    "Explosion": "Yes. Blast wave, fire, and rising black smoke.",
    "Fighting": "Yes. Altercation with multiple actors throwing punches.",
    "Robbery": "Yes. Brandishing weapons (knives/guns) and coercing victims.",
    "Shooting": "Yes. Firearm active discharge and public panic.",
    "Shoplifting": "Yes. Concealing goods in pockets or bags.",
    "Stealing": "Yes. Unauthorized moving of someone else's property.",
    "Vandalism": "Yes. Destroying public property or spray painting.",
    "Arrest": "No. Handcuffing suspect under police control.",
    "Normal": "No. Normal everyday activity."
}

EMERGENCY_DISPATCH = {
    "Abuse": "Police patrol and local security units should be dispatched.",
    "Arson": "Fire department and police should be dispatched immediately.",
    "Assault": "Police patrol and medical backup should be dispatched.",
    "Burglary": "Police dispatch to check the property alarm and secure suspect.",
    "Explosion": "Fire rescue, SWAT units, and medical dispatch required.",
    "Fighting": "Police patrol units should be dispatched to clear the brawl.",
    "Robbery": "Armed police dispatch required immediately.",
    "Shooting": "SWAT units and emergency medical teams required.",
    "Shoplifting": "Retail security alert. Non-emergency police log.",
    "Stealing": "Police log for stolen property tracking.",
    "Vandalism": "Police patrol dispatch to catch vandal in progress.",
    "Arrest": "None. Suspect already detained by officers.",
    "Normal": "No dispatch required. Normal operations."
}

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="UCF-Crime SFT Dataset Generator")
    parser.add_argument("--dataset-dir", default="datasets", help="Path containing Test/ and Train/ UCF folders")
    parser.add_argument("--output-dir", default="training/data", help="Output path for JSONL files")
    parser.add_argument("--max-videos", type=int, default=500, help="Maximum unique videos to sample for training")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("Initializing UCF-Crime SFT Dataset Generator...")
    
    dataset_root = Path(args.dataset_dir)
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root
        
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    # UCF-Crime structure: Train/ and Test/ splits
    train_root = dataset_root / "Train"
    if not train_root.exists():
        # Fallback search locally
        train_root = local_root / "datasets" / "Train"
        
    if not train_root.exists():
        logger.warning(f"Train directory not found at {train_root}. Generating conceptual/mock list.")
        # Generate mock entries for local validation checks
        mock_entries = []
        for cat in ACTION_MAPPINGS.keys():
            mock_entries.append(("Abuse028_x264", cat, [local_root / "datasets" / "Train" / cat / "Abuse028_x264_0.png"]))
        video_groups = mock_entries
    else:
        # Group PNG files by video ID prefix lazily using os.scandir
        import os
        video_groups_dict = {}
        logger.info("Scanning Train split directories lazily...")
        
        for category_dir in train_root.iterdir():
            if not category_dir.is_dir():
                continue
            category = category_dir.name
            logger.info(f"Scanning Train category: {category}...")
            
            video_prefixes_found = set()
            with os.scandir(category_dir) as it:
                for entry in it:
                    if entry.is_file() and entry.name.endswith(".png"):
                        f = Path(entry.path)
                        name_parts = f.stem.split("_")
                        if len(name_parts) > 1:
                            video_prefix = "_".join(name_parts[:-1])
                        else:
                            video_prefix = f.stem
                            
                        # Stop scanning this folder once we have 80 unique video segments
                        if len(video_prefixes_found) >= 80 and video_prefix not in video_prefixes_found:
                            continue
                            
                        video_prefixes_found.add(video_prefix)
                        group_key = (video_prefix, category)
                        if group_key not in video_groups_dict:
                            video_groups_dict[group_key] = []
                        video_groups_dict[group_key].append(f)
                        
        video_groups = []
        for (prefix, category), frames in video_groups_dict.items():
            video_groups.append((prefix, category, frames))
            
    # Sample unique videos up to max_videos
    random.seed(42)
    random.shuffle(video_groups)
    selected_groups = video_groups[:args.max_videos]
    
    logger.info(f"Selected {len(selected_groups)} unique video segments for dataset generation.")
    sft_dataset = []
    
    for prefix, category, frames in selected_groups:
        action_caption = ACTION_MAPPINGS.get(category, ACTION_MAPPINGS["Normal"])
        reasoning_report = THREAT_REASONING.get(category, THREAT_REASONING["Normal"])
        suspicious_ans = SUSPICIOUS_INDICATORS.get(category, SUSPICIOUS_INDICATORS["Normal"])
        dispatch_ans = EMERGENCY_DISPATCH.get(category, EMERGENCY_DISPATCH["Normal"])
        
        # Pick the middle frame to represent the video segment
        frames.sort()
        midpoint_frame = frames[len(frames) // 2]
        
        frame_path_str = str(midpoint_frame).replace("\\", "/")
            
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
    logger.info("UCF-Crime SFT Dataset Generation Complete!")
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
