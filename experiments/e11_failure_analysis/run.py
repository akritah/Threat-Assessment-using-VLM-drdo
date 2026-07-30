import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.config import get_config
from experiments.common.eval_engine import (
    load_dataset_and_extract_frames,
    run_vlm_inference
)
from experiments.common.plotting import (
    plot_pie_chart
)
from experiments.common.reporting import (
    save_json_results,
    generate_latex_table,
    generate_markdown_report
)

def run_experiment():
    config = get_config("E11")
    output_dir = config["resolved_output_dir"]
    
    raw_dataset_dir = Path(config["dataset_dir"])
    extracted_data = load_dataset_and_extract_frames(
        dataset_dir=raw_dataset_dir,
        output_dir=output_dir,
        max_videos=config["max_videos"],
        seed=config["seed"]
    )
    
    # Load model if HF
    model = None
    processor = None
    if config["backend"] == "hf":
        from evaluation.run_eval_xd import load_gemma_model
        model, processor = load_gemma_model(config["model"], device=config["device"])
        
    failures = []
    
    # Failure categories count initializer
    failure_counts = {
        "Occlusion": 0,
        "Low Light": 0,
        "Motion Blur": 0,
        "Wrong Reasoning": 0,
        "Temporal Miss": 0,
        "Ambiguous Event": 0,
        "False Alarm": 0
    }

    for idx, sample in enumerate(extracted_data, 1):
        vid = sample["video_id"]
        cat = sample["category"]
        frames = sample["frame_paths"]
        print(f"[{idx}/{len(extracted_data)}] Screening video: {vid}")
        
        gt_is_anomaly = 1 if cat.lower() != "normal" else 0

        # Run Two-Stage Hybrid
        res_caption = run_vlm_inference(
            backend=config["backend"],
            model_name=config["model"],
            image_paths=frames,
            prompt=config["sft_caption_prompt"],
            device=config["device"],
            model=model,
            processor=processor
        )
        caption = res_caption["response"]
        
        guided_prompt = (
            "Analyze this surveillance scene video sequence.\n"
            f"You are given the following pre-extracted activity class: '{caption}'\n\n"
            "Using this action class and the visual evidence from the frames, describe:\n"
            "* What is happening?\n"
            "* Which activities are visible?\n"
            "* Is there any suspicious behaviour?\n"
            "* Are there any threat indicators?\n"
            "* Estimate the threat level as Low, Medium, or High.\n"
            "* Explain your reasoning."
        )
        res_reason = run_vlm_inference(
            backend=config["backend"],
            model_name=config["model"],
            image_paths=frames,
            prompt=guided_prompt,
            device=config["device"],
            model=model,
            processor=processor
        )
        report = res_reason["response"]
        
        text_reason = report.lower()
        threat = "Low"
        for line in text_reason.split("\n"):
            if "threat level" in line:
                if "high" in line:
                    threat = "High"
                elif "medium" in line:
                    threat = "Medium"
                break
        pred_is_anomaly = 1 if threat in ["High", "Medium"] else 0

        # Check if classification was a failure
        if gt_is_anomaly != pred_is_anomaly:
            print(f"Prediction mismatch for {vid}. Querying diagnostic analyzer...")
            
            # Query VLM-as-Judge to identify failure category
            diagnostic_prompt = (
                f"Surveillance Video Sequence: {vid}\n"
                f"Ground Truth Category: {cat}\n"
                f"Model Output Report: \"{report}\"\n\n"
                "This video sequence was misclassified. Analyze the keyframes and determine the primary reason.\n"
                "Respond with exactly one category from this list:\n"
                "- Occlusion\n"
                "- Low Light\n"
                "- Motion Blur\n"
                "- Wrong Reasoning\n"
                "- Temporal Miss\n"
                "- Ambiguous Event\n"
                "- False Alarm"
            )
            res_diag = run_vlm_inference(
                backend=config["backend"],
                model_name=config.get("judge_model", "gemma3:4b"),
                image_paths=frames,
                prompt=diagnostic_prompt,
                device=config["device"],
                model=model,
                processor=processor
            )
            diag_reason = res_diag["response"].strip()
            
            # Map response to category
            matched_cat = "Wrong Reasoning"
            for k in failure_counts.keys():
                if k.lower() in diag_reason.lower():
                    matched_cat = k
                    break
                    
            failure_counts[matched_cat] += 1
            failures.append({
                "video_id": vid,
                "category": cat,
                "predicted_threat": threat,
                "reason": matched_cat,
                "details": diag_reason,
                "first_frame_path": str(frames[0]) if frames else ""
            })

    # Clean up model
    if model is not None:
        del model
        import gc
        gc.collect()
        if config["device"] == "cuda":
            torch.cuda.empty_cache()

    # 4. Plot figures
    print("Generating failure analysis pie chart...")
    labels = list(failure_counts.keys())
    counts = list(failure_counts.values())
    plot_pie_chart(labels, counts, "Analysis of Pipeline Prediction Failures", output_dir, "failure_analysis_pie")

    # 6. Reporting
    save_json_results(failures, output_dir, "failure_analysis_results")

    # Generate LaTeX Table
    headers = ["Failure Reason Category", "Frequency Count", "Distribution (%)"]
    rows = []
    tot_failures = sum(counts)
    for label, count in failure_counts.items():
        pct = (count / tot_failures * 100) if tot_failures > 0 else 0.0
        rows.append([
            label,
            str(count),
            f"{pct:.1f}%"
        ])
    rows.append([
        "\\textbf{Total Failures}",
        f"\\textbf{{ {tot_failures} }}",
        "\\textbf{100.0%}"
    ])
    
    latex_code = generate_latex_table(
        headers, rows,
        label="failure_analysis",
        caption="Taxonomy and distribution of VLM prediction failures categorized using self-diagnostics.",
        output_dir=output_dir,
        name="failure_table"
    )

    # Markdown Report
    gallery_lines = []
    for f in failures:
        gallery_lines.append(f"* **Video:** `{f['video_id']}` ({f['category']}) $\\to$ Reason: **{f['reason']}**")
        
    sections = {
        "Failure Taxonomy Analysis": (
            f"This experiment logs and analyzes predictions that mismatch ground-truth annotations.\n\n"
            f"**Total prediction failures screened:** {len(failures)}.\n\n"
            "Classification failures are categorized to isolate system gaps (like low-light visibility or reasoning flaws)."
        ),
        "Failed Case Log": "\n".join(gallery_lines) if gallery_lines else "No failures detected in this run."
    }
    generate_markdown_report(
        "Experiment E11: Failure Analysis Study Report",
        sections,
        {"Failure Distribution Table": latex_code},
        output_dir,
        "E11_Failure_Analysis_Report"
    )
    print("Experiment E11 failure analysis complete.")

if __name__ == "__main__":
    run_experiment()
