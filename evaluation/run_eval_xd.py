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

# Also search locally for fallback
local_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(local_root))

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
    from models.model_loader import load_base_model
    from peft import PeftModel
    
    use_4bit = (device == "cuda")
    device_map = "auto" if device == "cuda" else {"": "cpu"}
    
    logger.info(f"Loading base model {model_id} via load_base_model (use_4bit={use_4bit})...")
    model, processor = load_base_model(model_id=model_id, use_4bit=use_4bit, device_map=device_map)
    
    # Resolve adapter path if nested
    actual_adapter_path = adapter_path
    if adapter_path and not Path(adapter_path).exists():
        # Search in project folder
        for p in PROJECT_ROOT.glob("**/adapter_model.safetensors"):
            actual_adapter_path = str(p.parent)
            break
        # Fallback to local search
        if not actual_adapter_path or not Path(actual_adapter_path).exists():
            for p in local_root.glob("**/adapter_model.safetensors"):
                actual_adapter_path = str(p.parent)
                break

    if actual_adapter_path and Path(actual_adapter_path).exists():
        logger.info(f"Attaching LoRA adapter from: {actual_adapter_path}")
        model = PeftModel.from_pretrained(model, actual_adapter_path, is_trainable=False)
    else:
        logger.warning("LoRA adapter not found or not specified. Running base model only.")
        
    model.eval()
    return model, processor

def run_inference(model, processor, image_paths, prompt, device):
    """Run inference over a list of image keyframes (multi-frame VLM processing)."""
    content_list = []
    # Support up to 8 frames for temporal window processing
    for path in image_paths[:8]:
        img = Image.open(path).convert("RGB")
        content_list.append({"type": "image", "image": img})
    content_list.append({"type": "text", "text": prompt})
    
    messages = [{"role": "user", "content": content_list}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    if device == "cuda":
        inputs = {k: (v.to(torch.float16) if v.dtype == torch.float32 else v) for k, v in inputs.items()}
    else:
        inputs = {k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v) for k, v in inputs.items()}
        
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
    if not selected_csv_path.parent.exists():
        selected_csv_path = local_root / "selected_videos.csv"

    with selected_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Video ID", "Category", "Path"])
        for row in selected_list:
            writer.writerow(row)

    # Extract 8 frames per video to enable temporal windowing
    extracted_data = []
    for video_id, category, path in selected_list:
        v_path = Path(path)
        v_frame_dir = extracted_frames_dir / video_id
        try:
            frames = extract_frames(v_path, v_frame_dir, frame_count=8)
            if frames:
                extracted_data.append({
                    "video_id": video_id,
                    "category": category,
                    "video_path": v_path,
                    "frame_paths": frames
                })
        except Exception as e:
            logger.warning("Failed to extract frames for %s: %s", video_id, e)

    base_model_id = "google/gemma-3-4b-it"

    # --- STAGE 1: Fine-Tuned Action Captioning (8-Frame Input) ---
    logger.info("Starting Stage 1 SFT inference (Action caption extraction)...")
    ft_action_captions = {}
    
    # Load FT model to extract captions
    model, processor = load_gemma_model(base_model_id, adapter_path=str(adapter_path), device=args.device)
    caption_prompt = "Describe the exact activity happening in this video sequence as a concise caption (e.g., A person is performing...)."
    
    for idx, sample in enumerate(extracted_data, 1):
        vid = sample["video_id"]
        logger.info(f"[{idx}/{len(extracted_data)}] FT captioning: {vid}")
        caption, _ = run_inference(model, processor, sample["frame_paths"], caption_prompt, args.device)
        ft_action_captions[vid] = caption
        logger.info(f"FT Caption: {caption}")
        
    del model, processor
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # --- STAGE 2: Base VLM Evaluation & Guided Two-Stage Inference ---
    logger.info("Starting Stage 2 inference (Baseline vs. Guided Reasoning)...")
    base_results = {}
    ft_results = {}
    
    # Load Base model once
    model, processor = load_gemma_model(base_model_id, device=args.device)
    
    # Loop A: Base Model alone (Baseline)
    baseline_prompt = (
        "Analyze this surveillance scene video sequence.\n"
        "Describe:\n"
        "* What is happening?\n"
        "* Which activities are visible?\n"
        "* Is there any suspicious behaviour?\n"
        "* Are there any threat indicators?\n"
        "* Estimate the threat level as Low, Medium, or High.\n"
        "* Explain your reasoning."
    )
    
    for idx, sample in enumerate(extracted_data, 1):
        vid = sample["video_id"]
        logger.info(f"[{idx}/{len(extracted_data)}] Base Baseline: {vid}")
        res, latency = run_inference(model, processor, sample["frame_paths"], baseline_prompt, args.device)
        base_results[vid] = (res, latency)

    # Loop B: Base Model guided by FT Caption (Two-Stage Hybrid)
    for idx, sample in enumerate(extracted_data, 1):
        vid = sample["video_id"]
        action_caption = ft_action_captions.get(vid, "Unknown activity")
        guided_prompt = (
            "Analyze this surveillance scene video sequence.\n"
            f"You are given the following pre-extracted activity class: '{action_caption}'\n\n"
            "Using this action class and the visual evidence from the frames, describe:\n"
            "* What is happening?\n"
            "* Which activities are visible?\n"
            "* Is there any suspicious behaviour?\n"
            "* Are there any threat indicators?\n"
            "* Estimate the threat level as Low, Medium, or High.\n"
            "* Explain your reasoning."
        )
        logger.info(f"[{idx}/{len(extracted_data)}] Guided Two-Stage: {vid}")
        res, latency = run_inference(model, processor, sample["frame_paths"], guided_prompt, args.device)
        ft_results[vid] = (res, latency)

    del model, processor
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # --- Compile Deliverables ---
    eval_csv_path = PROJECT_ROOT / "evaluation_results.csv"
    if not eval_csv_path.parent.exists():
        eval_csv_path = local_root / "evaluation_results.csv"

    eval_headers = [
        "Video ID", "Ground Truth Category", "Frames Used", "Base Gemma Output", 
        "Fine-Tuned Gemma Output", "Video-LLaVA Output", "Predicted Activity", 
        "Threat Assessment", "Threat Level", "Inference Time", "Notes"
    ]
    
    records = []
    for sample in extracted_data:
        vid = sample["video_id"]
        cat = sample["category"]
        frames_str = ";".join([str(p) for p in sample["frame_paths"]])
        
        base_out, base_time = base_results.get(vid, ("N/A", 0.0))
        ft_guided_out, ft_time = ft_results.get(vid, ("N/A", 0.0))
        predicted_activity = ft_action_captions.get(vid, "Unknown")
        
        # Parse Guided Output for metrics
        threat_level = "Low"
        for line in ft_guided_out.split("\n"):
            if "threat level" in line.lower():
                raw_level = line.split(":")[-1].strip().lower()
                if "high" in raw_level:
                    threat_level = "High"
                elif "medium" in raw_level:
                    threat_level = "Medium"
                break
                
        records.append({
            "Video ID": vid,
            "Ground Truth Category": cat,
            "Frames Used": frames_str,
            "Base Gemma Output": base_out,
            "Fine-Tuned Gemma Output": ft_guided_out,
            "Video-LLaVA Output": "N/A - Skip (Not installed/configured)",
            "Predicted Activity": predicted_activity,
            "Threat Assessment": ft_guided_out,
            "Threat Level": threat_level,
            "Inference Time": f"Base: {base_time:.2f}s | FT Guided: {ft_time:.2f}s",
            "Notes": f"Two-Stage Hybrid (FT Caption + Base Reasoning)"
        })

    with eval_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=eval_headers)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    # Plot Distribution
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
    ax.bar([i + width/2 for i in x], ft_counts, width, label="Fine-Tuned Guided (Two-Stage)", color="#1565c0")
    ax.set_ylabel("Count")
    ax.set_title("Threat Level Estimation Comparison (Two-Stage)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories_plot)
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "threat_distribution.png", dpi=300)
    plt.close()

    # Load and compile report
    report_template_path = output_dir / "report_template.md"
    if not report_template_path.exists():
        report_template_path = local_root / "evaluation" / "report_template.md"
        
    with open(report_template_path, "r", encoding="utf-8") as f:
        report_template = f.read()

    cat_distribution = {}
    for r in records:
        cat_distribution[r["Ground Truth Category"]] = cat_distribution.get(r["Ground Truth Category"], 0) + 1
    dist_lines = [f"*   **{cat}**: {count} videos" for cat, count in cat_distribution.items()]

    # Calculate Binary Classification Metrics
    base_tp = base_fp = base_tn = base_fn = 0
    ft_tp = ft_fp = ft_tn = ft_fn = 0
    
    for r in records:
        gt_category = r["Ground Truth Category"]
        is_gt_anomaly = (gt_category != "Normal")
        
        # Base Model Prediction Parsing
        base_out = r["Base Gemma Output"].lower()
        base_threat = "Low"
        for line in base_out.split("\n"):
            if "threat level" in line:
                if "high" in line:
                    base_threat = "High"
                elif "medium" in line:
                    base_threat = "Medium"
                break
        is_base_threat = (base_threat in ["High", "Medium"])
        
        if is_gt_anomaly and is_base_threat:
            base_tp += 1
        elif not is_gt_anomaly and is_base_threat:
            base_fp += 1
        elif not is_gt_anomaly and not is_base_threat:
            base_tn += 1
        elif is_gt_anomaly and not is_base_threat:
            base_fn += 1
            
        # Fine-Tuned Model Prediction Parsing
        is_ft_threat = (r["Threat Level"] in ["High", "Medium"])
        
        if is_gt_anomaly and is_ft_threat:
            ft_tp += 1
        elif not is_gt_anomaly and is_ft_threat:
            ft_fp += 1
        elif not is_gt_anomaly and not is_ft_threat:
            ft_tn += 1
        elif is_gt_anomaly and not is_ft_threat:
            ft_fn += 1

    def calc_metrics(tp, fp, tn, fn):
        acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return acc * 100, prec * 100, rec * 100, f1 * 100

    base_acc, base_prec, base_rec, base_f1 = calc_metrics(base_tp, base_fp, base_tn, base_fn)
    ft_acc, ft_prec, ft_rec, ft_f1 = calc_metrics(ft_tp, ft_fp, ft_tn, ft_fn)

    report_content = report_template.replace("{len_records}", str(len(records)))
    report_content = report_content.replace("{dist_lines}", "\n".join(dist_lines))
    report_content = report_content.replace("{base_avg_time}", f"{sum(b[1] for b in base_results.values())/len(base_results):.2f}")
    report_content = report_content.replace("{ft_avg_time}", f"{sum(f[1] for f in ft_results.values())/len(ft_results):.2f}")
    
    # Classification metrics replacements
    report_content = report_content.replace("{base_acc}", f"{base_acc:.1f}")
    report_content = report_content.replace("{base_prec}", f"{base_prec:.1f}")
    report_content = report_content.replace("{base_rec}", f"{base_rec:.1f}")
    report_content = report_content.replace("{base_f1}", f"{base_f1:.1f}")
    
    report_content = report_content.replace("{ft_acc}", f"{ft_acc:.1f}")
    report_content = report_content.replace("{ft_prec}", f"{ft_prec:.1f}")
    report_content = report_content.replace("{ft_rec}", f"{ft_rec:.1f}")
    report_content = report_content.replace("{ft_f1}", f"{ft_f1:.1f}")
    
    # Confusion matrix replacements
    report_content = report_content.replace("{base_tp}", str(base_tp))
    report_content = report_content.replace("{base_fp}", str(base_fp))
    report_content = report_content.replace("{base_tn}", str(base_tn))
    report_content = report_content.replace("{base_fn}", str(base_fn))
    
    report_content = report_content.replace("{ft_tp}", str(ft_tp))
    report_content = report_content.replace("{ft_fp}", str(ft_fp))
    report_content = report_content.replace("{ft_tn}", str(ft_tn))
    report_content = report_content.replace("{ft_fn}", str(ft_fn))

    report_out_path = PROJECT_ROOT / "evaluation_report.md"
    if not report_out_path.parent.exists():
        report_out_path = local_root / "evaluation_report.md"
        
    with report_out_path.open("w", encoding="utf-8") as f:
        f.write(report_content)
        
    # Copy files
    try:
        import shutil
        shutil.copy(str(selected_csv_path), csv_dir / "selected_videos.csv")
        shutil.copy(str(eval_csv_path), csv_dir / "evaluation_results.csv")
        shutil.copy(str(report_out_path), reports_dir / "evaluation_report.md")
    except Exception as e:
        logger.warning(f"Could not copy files to subdir: {e}")
        
    print("Done!")

if __name__ == "__main__":
    main()
