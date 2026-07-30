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
    compute_classification_metrics,
    run_mcnemar_test,
    run_paired_bootstrap
)
from experiments.common.reporting import (
    save_json_results,
    generate_latex_table,
    generate_markdown_report
)
import numpy as np

def run_experiment():
    config = get_config("E3")
    output_dir = config["resolved_output_dir"]
    seeds = config.get("seeds", [42, 100, 2026])
    
    raw_dataset_dir = Path(config["dataset_dir"])
    
    # Load model if HF
    model = None
    processor = None
    if config["backend"] == "hf":
        from evaluation.run_eval_xd import load_gemma_model
        model, processor = load_gemma_model(config["model"], device=config["device"])
        
    seed_metrics = []
    
    # For significance testing, we will collect predictions of first seed
    first_gt = []
    first_base_preds = []
    first_hybrid_preds = []

    for s_idx, seed in enumerate(seeds):
        print(f"\n--- Running evaluation with Seed: {seed} ---")
        extracted_data = load_dataset_and_extract_frames(
            dataset_dir=raw_dataset_dir,
            output_dir=output_dir / f"seed_{seed}",
            max_videos=config["max_videos"],
            seed=seed
        )
        
        gt_labels = []
        base_preds = []
        hybrid_preds = []
        
        for idx, sample in enumerate(extracted_data, 1):
            vid = sample["video_id"]
            cat = sample["category"]
            frames = sample["frame_paths"]
            
            gt_is_anomaly = 1 if cat.lower() != "normal" else 0
            gt_labels.append(gt_is_anomaly)
            
            # 1. Base VLM Baseline
            base_threat = "Low"
            if not config["skip_baseline"]:
                res_a = run_vlm_inference(
                    backend=config["backend"],
                    model_name=config["model"],
                    image_paths=frames,
                    prompt=config["baseline_prompt"],
                    device=config["device"],
                    model=model,
                    processor=processor
                )
                text_base = res_a["response"].lower()
                for line in text_base.split("\n"):
                    if "threat level" in line:
                        if "high" in line:
                            base_threat = "High"
                        elif "medium" in line:
                            base_threat = "Medium"
                        break
            base_preds.append(1 if base_threat in ["High", "Medium"] else 0)
            
            # 2. Two-Stage Hybrid
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
            
            text_hybrid = res_reason["response"].lower()
            hybrid_threat = "Low"
            for line in text_hybrid.split("\n"):
                if "threat level" in line:
                    if "high" in line:
                        hybrid_threat = "High"
                    elif "medium" in line:
                        hybrid_threat = "Medium"
                    break
            hybrid_preds.append(1 if hybrid_threat in ["High", "Medium"] else 0)
            
        metrics_base = compute_classification_metrics(gt_labels, base_preds)
        metrics_hybrid = compute_classification_metrics(gt_labels, hybrid_preds)
        
        seed_metrics.append({
            "seed": seed,
            "base": metrics_base,
            "hybrid": metrics_hybrid
        })
        
        # Save first seed outputs for significance tests
        if s_idx == 0:
            first_gt = gt_labels
            first_base_preds = base_preds
            first_hybrid_preds = hybrid_preds

    # Clean up VLM if HF
    if model is not None:
        del model
        import gc
        gc.collect()
        if config["device"] == "cuda":
            torch.cuda.empty_cache()

    # 4. Compute statistical properties (Mean, Std, 95% Confidence Interval)
    def aggregate_metric(key: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        # Collect base scores and hybrid scores
        base_vals = [m["base"][key] * 100 for m in seed_metrics]
        hybrid_vals = [m["hybrid"][key] * 100 for m in seed_metrics]
        
        base_stats = {
            "mean": float(np.mean(base_vals)),
            "std": float(np.std(base_vals)),
            "ci": 1.96 * float(np.std(base_vals)) / np.sqrt(len(seeds))
        }
        
        hybrid_stats = {
            "mean": float(np.mean(hybrid_vals)),
            "std": float(np.std(hybrid_vals)),
            "ci": 1.96 * float(np.std(hybrid_vals)) / np.sqrt(len(seeds))
        }
        return base_stats, hybrid_stats

    metrics_keys = ["accuracy", "precision", "recall", "f1_score"]
    stats_summary = {}
    for k in metrics_keys:
        base_st, hybrid_st = aggregate_metric(k)
        stats_summary[k] = {
            "base": base_st,
            "hybrid": hybrid_st
        }

    # Run McNemar's Test
    mcnemar_res = run_mcnemar_test(first_gt, first_base_preds, first_hybrid_preds)
    
    # Run Paired Bootstrap
    bootstrap_res = run_paired_bootstrap(first_gt, first_base_preds, first_hybrid_preds)

    # 5. Reporting
    report_data = {
        "runs": seed_metrics,
        "statistical_analysis": stats_summary,
        "mcnemar_test": mcnemar_res,
        "paired_bootstrap": bootstrap_res
    }
    save_json_results(report_data, output_dir, "expanded_eval_results")

    # Generate LaTeX Table
    headers = ["Metric", "Base Gemma Mean", "Base Gemma CI (95%)", "Hybrid Mean", "Hybrid CI (95%)"]
    rows = []
    metric_labels = {
        "accuracy": "Accuracy (%)",
        "precision": "Precision (%)",
        "recall": "Recall (%)",
        "f1_score": "F1-Score (%)"
    }
    for k in metrics_keys:
        st = stats_summary[k]
        rows.append([
            metric_labels[k],
            f"{st['base']['mean']:.2f} ± {st['base']['std']:.2f}",
            f"[{st['base']['mean'] - st['base']['ci']:.2f}, {st['base']['mean'] + st['base']['ci']:.2f}]",
            f"{st['hybrid']['mean']:.2f} ± {st['hybrid']['std']:.2f}",
            f"[{st['hybrid']['mean'] - st['hybrid']['ci']:.2f}, {st['hybrid']['mean'] + st['hybrid']['ci']:.2f}]"
        ])
        
    latex_code = generate_latex_table(
        headers, rows,
        label="expanded_eval",
        caption="Expanded VLM statistical validation showing performance mean and standard deviation across seeds.",
        output_dir=output_dir,
        name="expanded_eval_table"
    )

    # Markdown Report
    sections = {
        "Expanded Evaluation & Significance Tests": (
            f"This experiment ran VLM benchmarking across {len(seeds)} different random seeds to model classification stability.\n\n"
            f"**McNemar Chi-Square Statistic:** {mcnemar_res['chi2_statistic']:.4f}\n"
            f"**McNemar p-value:** {mcnemar_res['p_value']:.4f} "
            f"({'Statistically Significant (p < 0.05)' if mcnemar_res['significant_05'] else 'Not Statistically Significant (p >= 0.05)'}).\n\n"
            f"**Bootstrap Mean Difference (Hybrid - Base):** {bootstrap_res['mean_difference']*100:.2f}%\n"
            f"**Bootstrap 95% CI of Difference:** [{bootstrap_res['confidence_interval_95'][0]*100:.2f}%, {bootstrap_res['confidence_interval_95'][1]*100:.2f}%]."
        )
    }
    generate_markdown_report(
        "Experiment E3: Expanded Evaluation Study Report",
        sections,
        {"Statistical Metrics Table": latex_code},
        output_dir,
        "E3_Expanded_Evaluation_Report"
    )
    print("Experiment E3 expanded evaluation complete.")

if __name__ == "__main__":
    run_experiment()
