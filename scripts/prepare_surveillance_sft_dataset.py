from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
local_root = PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}

# Normalized category names support both XD-Violence and UCF-Crime folder labels.
ACTION_MAPPINGS = {
    "abuse": "A person is performing physical abuse and assault on an individual.",
    "arrest": "Police officers are arresting a suspect in a public scene.",
    "arson": "A person is intentionally setting a fire or causing deliberate damage.",
    "assault": "A person is attacking or assaulting another individual.",
    "burglary": "A person is breaking into a property or attempting a burglary.",
    "caraccident": "A car accident or vehicle collision is taking place, causing damage.",
    "explosion": "An explosion is occurring, destroying property and throwing debris.",
    "fighting": "Multiple individuals are physically fighting, punching, and wrestling.",
    "normal": "A normal surveillance scene showing regular street and pedestrian activity.",
    "roadaccident": "A road accident or traffic collision is visible in the scene.",
    "roadaccidents": "A road accident or traffic collision is visible in the scene.",
    "robbery": "A robbery or theft-related confrontation is occurring.",
    "riot": "A riot or violent public demonstration is occurring, with individuals throwing objects.",
    "shooting": "An active shooter is firing a weapon.",
    "shoplifting": "A person is covertly stealing goods from a store.",
    "stealing": "A person is stealing or concealing property without permission.",
    "vandalism": "A person is damaging property or committing vandalism.",
}

THREAT_REASONING = {
    "abuse": (
        "**What is happening?**\nAn act of physical violence and abuse is occurring between individuals.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nActive physical abuse causes direct physical harm and requires security intervention."
    ),
    "arrest": (
        "**What is happening?**\nPolice officers are detaining a suspect.\n\n"
        "**Threat Level**\nLow\n\n"
        "**Reasoning**\nAn arrest often indicates a controlled law-enforcement response rather than an active threat."
    ),
    "arson": (
        "**What is happening?**\nA deliberate fire-setting event is underway, with visible smoke or flames.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nArson is a severe threat to life and property and requires immediate emergency response."
    ),
    "assault": (
        "**What is happening?**\nA physical assault is occurring between individuals.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nAssault creates immediate bodily harm risk and requires police intervention."
    ),
    "burglary": (
        "**What is happening?**\nA burglary or break-in attempt is visible.\n\n"
        "**Threat Level**\nMedium\n\n"
        "**Reasoning**\nBurglary is a property crime with security implications, though it may not be immediately violent."
    ),
    "caraccident": (
        "**What is happening?**\nA vehicle collision or car accident is visible, showing damaged vehicles.\n\n"
        "**Threat Level**\nMedium\n\n"
        "**Reasoning**\nAccidents threaten public safety and require police and emergency medical dispatch."
    ),
    "explosion": (
        "**What is happening?**\nA severe explosion is visible, generating smoke, fire, and structural damage.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nExplosions present catastrophic threats to life and property, requiring fire rescue and police."
    ),
    "fighting": (
        "**What is happening?**\nMultiple individuals are engaged in a physical brawl or group fight.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nFighting threatens public order and safety, requiring police patrol dispatch."
    ),
    "normal": (
        "**What is happening?**\nPedestrians and vehicles are moving through the frame normally without incident.\n\n"
        "**Threat Level**\nLow\n\n"
        "**Reasoning**\nNo threat indicators, suspicious behaviors, or hostile events are detected."
    ),
    "roadaccident": (
        "**What is happening?**\nA traffic collision or road accident is visible.\n\n"
        "**Threat Level**\nMedium\n\n"
        "**Reasoning**\nRoad accidents create safety hazards and often need medical and traffic response."
    ),
    "roadaccidents": (
        "**What is happening?**\nA traffic collision or road accident is visible.\n\n"
        "**Threat Level**\nMedium\n\n"
        "**Reasoning**\nRoad accidents create safety hazards and often need medical and traffic response."
    ),
    "robbery": (
        "**What is happening?**\nA robbery is underway, with theft or forceful taking of property.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nRobbery is a violent property crime and should be treated as a high threat."
    ),
    "riot": (
        "**What is happening?**\nA group of rioters is committing acts of violence and public destruction.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nRiots present severe public safety risks and widespread destruction, requiring SWAT or riot police containment."
    ),
    "shooting": (
        "**What is happening?**\nAn individual is brandishing a firearm and active shooting is underway.\n\n"
        "**Threat Level**\nHigh\n\n"
        "**Reasoning**\nActive shooting is a lethal threat to public safety, requiring immediate armed SWAT containment."
    ),
    "shoplifting": (
        "**What is happening?**\nA person is covertly stealing goods from a store.\n\n"
        "**Threat Level**\nMedium\n\n"
        "**Reasoning**\nShoplifting is a property crime that needs loss-prevention or police intervention."
    ),
    "stealing": (
        "**What is happening?**\nA person is stealing property without permission.\n\n"
        "**Threat Level**\nMedium\n\n"
        "**Reasoning**\nStealing is a theft-related incident that warrants security attention."
    ),
    "vandalism": (
        "**What is happening?**\nA person is damaging or defacing property.\n\n"
        "**Threat Level**\nMedium\n\n"
        "**Reasoning**\nVandalism is a disruptive property crime and should be addressed by security or police."
    ),
}

SUSPICIOUS_INDICATORS = {
    "abuse": "Yes. Aggressive posture and hostile physical contact.",
    "arrest": "Yes. Police intervention is present, but the scene is controlled.",
    "arson": "Yes. Active fire, accelerant-like behavior, or deliberate ignition.",
    "assault": "Yes. Aggressive contact and bodily harm risk.",
    "burglary": "Yes. Unauthorized entry, suspicious movement, or forced access.",
    "caraccident": "Yes. Vehicle collision, sudden deceleration, and wreckage.",
    "explosion": "Yes. Active fire, blast wave, and rising black smoke.",
    "fighting": "Yes. Altercation with multiple actors throwing punches.",
    "normal": "No. Normal everyday activity.",
    "roadaccident": "Yes. Vehicle collision, wreckage, and traffic disruption.",
    "roadaccidents": "Yes. Vehicle collision, wreckage, and traffic disruption.",
    "robbery": "Yes. Coercive theft behavior, confrontation, or property seizure.",
    "riot": "Yes. Group violence, vandalism, and throwing projectiles.",
    "shooting": "Yes. Firearm active discharge and public panic.",
    "shoplifting": "Yes. Concealment of merchandise or suspicious loitering.",
    "stealing": "Yes. Unauthorised removal or concealment of property.",
    "vandalism": "Yes. Property damage, graffiti, or destruction.",
}

EMERGENCY_DISPATCH = {
    "abuse": "Police patrol and local security units should be dispatched.",
    "arrest": "No urgent dispatch required beyond standard police supervision.",
    "arson": "Fire rescue, SWAT units, and medical dispatch required.",
    "assault": "Police patrol and emergency medical units should be dispatched.",
    "burglary": "Police patrol and property security units should be dispatched.",
    "caraccident": "Emergency medical services and traffic police should be dispatched.",
    "explosion": "Fire rescue, SWAT units, and medical dispatch required.",
    "fighting": "Police patrol units should be dispatched to clear the brawl.",
    "normal": "No dispatch required. Normal operations.",
    "roadaccident": "Emergency medical services and traffic police should be dispatched.",
    "roadaccidents": "Emergency medical services and traffic police should be dispatched.",
    "robbery": "Police patrol and emergency response units should be dispatched.",
    "riot": "SWAT units, riot police, and emergency medical teams required.",
    "shooting": "SWAT units and emergency medical teams required.",
    "shoplifting": "Loss-prevention staff or police should be dispatched.",
    "stealing": "Security or police should be dispatched.",
    "vandalism": "Police patrol or property security should be dispatched.",
}


def normalize_category(category: str) -> str:
    return category.replace(" ", "").replace("-", "").replace("_", "").lower()


def lookup(mapping: dict[str, str], category: str) -> str:
    normalized = normalize_category(category)
    return mapping.get(normalized, mapping["normal"])


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
    except Exception as exc:
        logger.warning("Error extracting frame from %s: %s", video_path, exc)

    return False


def save_representative_image(image_path: Path, output_image_path: Path) -> bool:
    """Copy or re-encode a still image into the output frames directory."""
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            img.convert("RGB").save(output_image_path, format="JPEG", quality=95)
        return True
    except Exception as exc:
        logger.warning("Error copying image from %s: %s", image_path, exc)
        return False


def _is_media_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS.union(VIDEO_EXTENSIONS)


def find_representative_media(root: Path) -> Path | None:
    """Find the first usable still image or video inside a folder tree."""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    entry_path = Path(entry.path)
                    if entry.is_file() and _is_media_file(entry_path):
                        return entry_path
                    if entry.is_dir():
                        stack.append(entry_path)
        except FileNotFoundError:
            continue
    return None


def write_representative_frame(source_path: Path, output_image_path: Path) -> bool:
    suffix = source_path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return save_representative_image(source_path, output_image_path)
    if suffix in VIDEO_EXTENSIONS:
        return extract_video_frame(source_path, output_image_path)
    logger.warning("Unsupported media type: %s", source_path)
    return False


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="XD-Violence / UCF-Crime SFT Dataset Generator")
    parser.add_argument("--dataset-dir", default="datasets", help="Path containing train/ and test/ folders")
    parser.add_argument("--output-dir", default="training/data", help="Output path for JSONL files")
    parser.add_argument("--max-videos", type=int, default=500, help="Maximum unique samples to select for training")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing surveillance SFT dataset generator...")

    dataset_root = Path(args.dataset_dir)
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    train_root = dataset_root / "train"
    if not train_root.exists():
        train_root = dataset_root / "Train"
    if not train_root.exists():
        train_root = local_root / "datasets" / "train"

    if not train_root.exists():
        logger.warning("Train directory not found at %s. Generating conceptual/mock list.", train_root)
        mock_entries = []
        for cat in ACTION_MAPPINGS.keys():
            mock_entries.append((local_root / "datasets" / "train" / cat / "mock_video.mp4", cat))
        media_entries = mock_entries
    else:
        media_entries = []
        logger.info("Scanning train split directories lazily...")

        for category_dir in train_root.iterdir():
            if not category_dir.is_dir():
                continue
            category = category_dir.name
            logger.info("Scanning train category: %s...", category)

            count = 0
            with os.scandir(category_dir) as it:
                for entry in it:
                    if count >= 80:
                        break

                    entry_path = Path(entry.path)
                    if entry.is_file() and _is_media_file(entry_path):
                        media_entries.append((entry_path, category))
                        count += 1
                        continue

                    if entry.is_dir():
                        representative = find_representative_media(entry_path)
                        if representative is not None:
                            media_entries.append((representative, category))
                            count += 1

    random.seed(42)
    random.shuffle(media_entries)
    selected_entries = media_entries[: args.max_videos]

    logger.info("Selected %d media samples for dataset generation. Extracting frames...", len(selected_entries))

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    sft_dataset = []
    success_count = 0

    for media_path, category in selected_entries:
        frame_path = frames_dir / f"{media_path.stem}.jpg"

        if "mock_video" in media_path.name:
            from PIL import Image

            img = Image.new("RGB", (100, 100), color="white")
            img.save(frame_path, "JPEG")
            frame_path_str = str(frame_path).replace("\\", "/")
            success_count += 1
        else:
            if not write_representative_frame(media_path, frame_path):
                continue
            frame_path_str = str(frame_path).replace("\\", "/")
            success_count += 1

        action_caption = lookup(ACTION_MAPPINGS, category)
        reasoning_report = lookup(THREAT_REASONING, category)
        suspicious_ans = lookup(SUSPICIOUS_INDICATORS, category)
        dispatch_ans = lookup(EMERGENCY_DISPATCH, category)

        sft_dataset.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": frame_path_str},
                            {
                                "type": "text",
                                "text": "Describe the exact activity happening in this video sequence as a concise caption.",
                            },
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": action_caption}]},
                ]
            }
        )

        sft_dataset.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": frame_path_str},
                            {
                                "type": "text",
                                "text": "Analyze this surveillance scene video sequence.\nDescribe:\n* What is happening?\n* Estimate the threat level as Low, Medium, or High.\n* Explain your reasoning.",
                            },
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": reasoning_report}]},
                ]
            }
        )

        sft_dataset.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": frame_path_str},
                            {
                                "type": "text",
                                "text": "Are there any suspicious elements, weapons, or abnormal behaviors in this video sequence? Describe them.",
                            },
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": suspicious_ans}]},
                ]
            }
        )

        sft_dataset.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": frame_path_str},
                            {
                                "type": "text",
                                "text": "Which emergency services should be notified for this surveillance scene?",
                            },
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": dispatch_ans}]},
                ]
            }
        )

    random.shuffle(sft_dataset)
    split_idx = int(len(sft_dataset) * 0.8)
    train_set = sft_dataset[:split_idx]
    val_set = sft_dataset[split_idx:]

    train_file = output_dir / "surveillance_train.jsonl"
    val_file = output_dir / "surveillance_val.jsonl"

    with train_file.open("w", encoding="utf-8") as handle:
        for item in train_set:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    with val_file.open("w", encoding="utf-8") as handle:
        for item in val_set:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("=========================================")
    logger.info("Surveillance SFT Dataset Generation Complete!")
    logger.info(" - Extracted frames successfully: %d/%d", success_count, len(selected_entries))
    logger.info(" - Train samples: %d", len(train_set))
    logger.info(" - Validation samples: %d", len(val_set))
    logger.info(" - Total dataset size: %d samples", len(sft_dataset))
    try:
        logger.info(" - Outputs written to: %s", output_dir.relative_to(PROJECT_ROOT))
    except ValueError:
        logger.info(" - Outputs written to: %s", output_dir)
    logger.info("=========================================")


if __name__ == "__main__":
    main()