import argparse
import csv
import gc
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any
import matplotlib.pyplot as plt
from PIL import Image
import torch

# 1. Automatically find and mount frame_extractor.py's directory
PROJECT_ROOT = Path("/content/project")
frame_extractor_dir = None
for p in PROJECT_ROOT.glob("**/frame_extractor.py"):
    frame_extractor_dir = p.parent
    break

if frame_extractor_dir:
    sys.path.insert(0, str(frame_extractor_dir))
else:
    sys.path.insert(0, str(PROJECT_ROOT))

from frame_extractor import extract_frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="/content/xd_violence_subset")
    parser.add_argument("--adapter-path", default="/content/project/adapters/activitynet_v1")
    parser.add_argument("--output-dir", default="/content/project/evaluation")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()

def load_gemma_model(model_id, adapter_path=None, device="cpu"):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    load_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16, "device_map": "auto"}
    model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
    
    # If the adapter is also nested, find it automatically
    actual_adapter_path = adapter_path
    if adapter_path and not Path(adapter_path).exists():
        for p in PROJECT_ROOT.glob("**/adapter_model.safetensors"):
            actual_adapter_path = str(p.parent)
            break

    if actual_adapter_path and Path(actual_adapter_path).exists():
        logger.info(f"Loading LoRA adapter from: {actual_adapter_path}")
        model = PeftModel.from_pretrained(model, actual_adapter_path, is_trainable=False)
    else:
        logger.warning("LoRA adapter not found. Running base model.")
        
    model.eval()
    return model, processor

def run_inference(model, processor, image_path, prompt, device):
    img = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs = {k: (v.to(torch.float16) if v.dtype == torch.float32 else v) for k, v in inputs.items()}
    start_time = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    latency = time.perf_counter() - start_time
    input_len = inputs["input_ids"].shape[-1]
    response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    return response, latency

def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    adapter_path = Path(args.adapter_path)
    output_dir = Path(args.output_dir)

    selected_videos_dir = output_dir / "selected_videos"
    extracted_frames_dir = output_dir / "extracted_frames"
    outputs_dir = output_dir / "outputs"
    reports_dir = output_dir / "reports"
    csv_dir = output_dir / "csv"
    plots_dir = output_dir / "plots"

    for d in [selected_videos_dir, extracted_frames_dir, outputs_dir, reports_dir, csv_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_videos = list(dataset_dir.glob("**/*.mp4"))
    selected_list = []
    for v in all_videos:
        video_id = v.name.replace(".mp4", "")
        category = v.parent.name
        selected_list.append((video_id, category, str(v)))

    selected_csv_path = PROJECT_ROOT / "selected_videos.csv"
    with selected_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Video ID", "Category", "Path"])
        for row in selected_list:
            writer.writerow(row)

    extracted_data = []
    for video_id, category, path in selected_list:
        v_path = Path(path)
        v_frame_dir = extracted_frames_dir / video_id
        try:
            frames = extract_frames(v_path, v_frame_dir, frame_count=1)
            if frames:
                extracted_data.append({"video_id": video_id, "category": category, "video_path": v_path, "frame_path": frames[0]})
        except Exception as e:
            logger.warning("Failed to extract frames for %s: %s", video_id, e)

    prompt = "Analyze this surveillance scene.\nDescribe:\n* What is happening?\n* Which activities are visible?\n* Is there any suspicious behaviour?\n* Are there any threat indicators?\n* Estimate the threat level as Low, Medium, or High.\n* Explain your reasoning."

    base_model_id = "google/gemma-3-4b-it"
    base_results = {}
    model, processor = load_gemma_model(base_model_id, device=args.device)
    for idx, sample in enumerate(extracted_data, 1):
        vid = sample["video_id"]
        res, latency = run_inference(model, processor, sample["frame_path"], prompt, args.device)
        base_results[vid] = (res, latency)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    ft_results = {}
    model, processor = load_gemma_model(base_model_id, adapter_path=str(adapter_path), device=args.device)
    for idx, sample in enumerate(extracted_data, 1):
        vid = sample["video_id"]
        res, latency = run_inference(model, processor, sample["frame_path"], prompt, args.device)
        ft_results[vid] = (res, latency)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    eval_csv_path = PROJECT_ROOT / "evaluation_results.csv"
    eval_headers = ["Video ID", "Ground Truth Category", "Frame Used", "Base Gemma Output", "Fine-Tuned Gemma Output", "Video-LLaVA Output", "Predicted Activity", "Threat Assessment", "Threat Level", "Inference Time", "Notes"]
    records = []
    for sample in extracted_data:
        vid = sample["video_id"]
        cat = sample["category"]
        frame_rel = sample["frame_path"]
        base_out, base_time = base_results.get(vid, ("N/A", 0.0))
        ft_out, ft_time = ft_results.get(vid, ("N/A", 0.0))
        predicted_activity = "Unknown"
        threat_level = "Low"
        for line in ft_out.split("\n"):
            if "activities" in line.lower() or "happening" in line.lower():
                predicted_activity = line.split(":")[-1].strip()
            if "threat level" in line.lower():
                raw_level = line.split(":")[-1].strip().lower()
                if "high" in raw_level:
                    threat_level = "High"
                elif "medium" in raw_level:
                    threat_level = "Medium"
        records.append({"Video ID": vid, "Ground Truth Category": cat, "Frame Used": str(frame_rel), "Base Gemma Output": base_out, "Fine-Tuned Gemma Output": ft_out, "Video-LLaVA Output": "N/A - Skip (Not installed/configured on local CPU)", "Predicted Activity": predicted_activity, "Threat Assessment": ft_out, "Threat Level": threat_level, "Inference Time": f"Base: {base_time:.2f}s | FT: {ft_time:.2f}s", "Notes": f"Executed on {args.device.upper()}"})

    with eval_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=eval_headers)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    base_levels = {"Low": 0, "Medium": 0, "High": 0}
    ft_levels = {"Low": 0, "Medium": 0, "High": 0}
    for vid, (base_out, _) in base_results.items():
        raw_level = "low"
        for line in base_out.split("\n"):
            if "threat level" in line.lower():
                raw_level = line.split(":")[-1].strip().lower()
                break
        if "high" in raw_level:
            base_levels["High"] += 1
        elif "medium" in raw_level:
            base_levels["Medium"] += 1
        else:
            base_levels["Low"] += 1
    for r in records:
        ft_levels[r["Threat Level"]] += 1

    categories_plot = ["Low", "Medium", "High"]
    base_counts = [base_levels[c] for c in categories_plot]
    ft_counts = [ft_levels[c] for c in categories_plot]
    x = range(len(categories_plot))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width/2 for i in x], base_counts, width, label="Base Gemma 3", color="#90caf9")
    ax.bar([i + width/2 for i in x], ft_counts, width, label="Fine-Tuned Gemma 3", color="#1565c0")
    ax.set_ylabel("Count")
    ax.set_title("Threat Level Estimation Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(categories_plot)
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "threat_distribution.png", dpi=300)
    plt.close()

    base_avg_time = sum(base_results[vid][1] for vid in base_results) / len(base_results)
    ft_avg_time = sum(ft_results[vid][1] for vid in ft_results) / len(ft_results)
    cat_distribution = {}
    for r in records:
        cat_distribution[r["Ground Truth Category"]] = cat_distribution.get(r["Ground Truth Category"], 0) + 1
    dist_lines = [f"*   **{cat}**: {count} videos" for cat, count in cat_distribution.items()]

    # Load template from file
    with open("/content/project/evaluation/report_template.md", "r") as f:
        report_template = f.read()

    # Perform standard string replacements
    report_content = report_template.replace("{len_records}", str(len(records)))
    report_content = report_content.replace("{dist_lines}", "\n".join(dist_lines))
    report_content = report_content.replace("{base_avg_time}", f"{base_avg_time:.2f}")
    report_content = report_content.replace("{ft_avg_time}", f"{ft_avg_time:.2f}")

    with Path("/content/project/evaluation_report.md").open("w") as f:
        f.write(report_content)
    
    # Copy files to category subdirs
    import shutil
    shutil.copy("/content/project/selected_videos.csv", csv_dir / "selected_videos.csv")
    shutil.copy("/content/project/evaluation_results.csv", csv_dir / "evaluation_results.csv")
    shutil.copy("/content/project/evaluation_report.md", reports_dir / "evaluation_report.md")
    print("Done!")

if __name__ == "__main__":
    main()
