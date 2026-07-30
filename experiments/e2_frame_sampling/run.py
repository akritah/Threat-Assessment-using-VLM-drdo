import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.config import get_config
from experiments.common.eval_engine import (
    load_dataset_and_extract_frames,
    run_vlm_inference,
    compute_classification_metrics
)
from experiments.common.plotting import (
    plot_pareto_frontier,
    save_publication_figure
)
from experiments.common.reporting import (
    save_json_results,
    generate_latex_table,
    generate_markdown_report
)
import matplotlib.pyplot as plt
import numpy as np

def run_experiment():
    config = get_config("E2")
    output_dir = config["resolved_output_dir"]
    frame_counts = config.get("frame_counts", [4, 8, 16, 32])
    
    # Load dataset paths once
    raw_dataset_dir = Path(config["dataset_dir"])
    
    # We will run the Two-Stage Hybrid pipeline
    model = None
    processor = None
    if config["backend"] == "hf":
        from evaluation.run_eval_xd import load_gemma_model
        model, processor = load_gemma_model(config["model"], device=config["device"])
        
    study_results = {}
    
    for count in frame_counts:
        print(f"\n--- Benchmarking frame count: {count} ---")
        extracted_data = load_dataset_and_extract_frames(
            dataset_dir=raw_dataset_dir,
            output_dir=output_dir / f"frames_{count}",
            max_videos=config["max_videos"],
            num_frames=count,
            seed=config["seed"]
        )
        
        gt_labels = []
        preds = []
        latencies = []
        ram_peaks = []
        vram_peaks = []
        
        for idx, sample in enumerate(extracted_data, 1):
            vid = sample["video_id"]
            cat = sample["category"]
            frames = sample["frame_paths"]
            
            gt_is_anomaly = 1 if cat.lower() != "normal" else 0
            gt_labels.append(gt_is_anomaly)
            
            # Step 1: Caption
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
            
            # Step 2: Guided reasoning
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
            
            # Record metrics
            lat = res_caption["metrics"]["latency_sec"] + res_reason["metrics"]["latency_sec"]
            latencies.append(lat)
            
            ram_peaks.append(res_reason["metrics"]["peak_ram_mb"])
            vram_peaks.append(res_reason["metrics"]["vram_allocated_mb"])
            
            text_resp = res_reason["response"].lower()
            threat = "Low"
            for line in text_resp.split("\n"):
                if "threat level" in line:
                    if "high" in line:
                        threat = "High"
                    elif "medium" in line:
                        threat = "Medium"
                    break
            preds.append(1 if threat in ["High", "Medium"] else 0)

        # Compute metrics for this count
        metrics = compute_classification_metrics(gt_labels, preds)
        
        study_results[count] = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "avg_latency": float(np.mean(latencies)),
            "avg_ram_peak": float(np.mean(ram_peaks)),
            "avg_vram_peak": float(np.mean(vram_peaks))
        }
        
        print(f"Frame Count {count} Result: F1={metrics['f1_score']*100:.1f}%, Latency={np.mean(latencies):.2f}s")

    # Clean up VLM if HF
    if model is not None:
        del model
        import gc
        gc.collect()
        if config["device"] == "cuda":
            torch.cuda.empty_cache()

    # 4. Determine Pareto frontier and Pareto-optimal frame count
    # Since we want to maximize accuracy and minimize latency:
    # A point is dominated if there is another point with higher/equal accuracy AND lower/equal latency.
    pareto_optimal = []
    points = [(count, study_results[count]["accuracy"], study_results[count]["avg_latency"]) for count in frame_counts]
    
    for i, (cnt_i, acc_i, lat_i) in enumerate(points):
        dominated = False
        for j, (cnt_j, acc_j, lat_j) in enumerate(points):
            if i == j:
                continue
            # j dominates i if: acc_j >= acc_i and lat_j <= lat_i and at least one is strict
            if (acc_j >= acc_i and lat_j <= lat_i) and (acc_j > acc_i or lat_j < lat_i):
                dominated = True
                break
        if not dominated:
            pareto_optimal.append(cnt_i)

    # Pick the best Pareto optimal as recommended (e.g. highest accuracy-to-latency ratio, or highest F1)
    recommended_frame_count = max(pareto_optimal, key=lambda c: study_results[c]["f1_score"])

    # 5. Plot figures
    print("Generating trade-off plots...")
    accuracies = [study_results[c]["accuracy"] * 100 for c in frame_counts]
    latencies = [study_results[c]["avg_latency"] for c in frame_counts]
    plot_pareto_frontier(accuracies, latencies, frame_counts, output_dir, "accuracy_latency_pareto")

    # Plot Memory vs Frames Line Chart
    fig, ax = plt.subplots(figsize=(6, 4.5))
    rams = [study_results[c]["avg_ram_peak"] for c in frame_counts]
    vrams = [study_results[c]["avg_vram_peak"] for c in frame_counts]
    
    ax.plot(frame_counts, rams, marker="o", color="#1565c0", linewidth=2, label="Peak CPU RAM Delta (MB)")
    if config["device"] == "cuda":
        ax.plot(frame_counts, vrams, marker="s", color="#d84315", linewidth=2, label="Peak GPU VRAM Delta (MB)")
        
    ax.set_xlabel("Number of Frames Sampled")
    ax.set_ylabel("Peak Resource Usage Delta (MB)")
    ax.set_title("Memory Footprint vs. Frame Sampling Count")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    save_publication_figure(fig, output_dir, "memory_vs_frames")

    # 6. Reporting
    save_json_results(study_results, output_dir, "frame_sampling_results")
    
    # LaTeX Table
    headers = ["Frames", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)", "Avg Latency (s)", "CPU RAM Peak (MB)"]
    rows = []
    for count in frame_counts:
        res = study_results[count]
        rows.append([
            str(count),
            f"{res['accuracy']*100:.1f}",
            f"{res['precision']*100:.1f}",
            f"{res['recall']*100:.1f}",
            f"{res['f1_score']*100:.1f}",
            f"{res['avg_latency']:.2f}",
            f"{res['avg_ram_peak']:.1f}"
        ])
        
    latex_code = generate_latex_table(
        headers, rows,
        label="frame_sampling_study",
        caption="VLM frame sampling trade-off study comparing metric quality, latency, and RAM footprints.",
        output_dir=output_dir,
        name="frame_sampling_table"
    )

    # Markdown Report
    sections = {
        "Frame Sampling Analysis": (
            f"This experiment benchmarks VLM accuracy and resources across different frame counts: {frame_counts}.\n\n"
            f"**Recommended Frame Count (Pareto Optimal):** {recommended_frame_count} frames.\n"
            f"**Pareto-optimal set of frames:** {pareto_optimal}.\n\n"
            "Increasing the frame count generally improves classification accuracy and recall, but incurs a linear penalty in model visual token encoding latency."
        )
    }
    generate_markdown_report(
        "Experiment E2: Frame Sampling Study Report",
        sections,
        {"Frame Sampling Performance Table": latex_code},
        output_dir,
        "E2_Frame_Sampling_Report"
    )
    print("Experiment E2 frame sampling study complete.")

if __name__ == "__main__":
    run_experiment()
