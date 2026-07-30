import os
import sys
import csv
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Ensure matplotlib backend is headless
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.plotting import (
    plot_confusion_matrix,
    plot_bar_comparison,
    save_publication_figure
)
from experiments.common.reporting import (
    save_json_results,
    generate_latex_table,
    generate_markdown_report
)

def load_real_data() -> list[dict[str, str]]:
    csv_path = PROJECT_ROOT / "evaluation" / "csv" / "evaluation_results.csv"
    if not csv_path.exists():
        csv_path = PROJECT_ROOT / "evaluation" / "outputs" / "evaluation_results.csv"
        
    records = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
    return records

def parse_threat_binary(out_text: str, is_hybrid: bool = False, raw_level: str = None) -> int:
    if is_hybrid:
        return 1 if raw_level in ["High", "Medium"] else 0
        
    out_text = out_text.lower()
    threat = "Low"
    for line in out_text.split("\n"):
        if "threat level" in line:
            if "high" in line:
                threat = "High"
            elif "medium" in line:
                threat = "Medium"
            break
    return 1 if threat in ["High", "Medium"] else 0

def main():
    print("Loading real VLM evaluation dataset...")
    records = load_real_data()
    n_total = len(records)
    print(f"Loaded {n_total} records from evaluation_results.csv.")

    # 1. Separate subset where both Base and Hybrid ran
    valid_comparison_subset = []
    for r in records:
        base_out = r.get("Base Gemma Output", "")
        hybrid_level = r.get("Threat Level", "")
        has_base = base_out and "N/A" not in base_out and "skip" not in base_out.lower()
        has_hybrid = hybrid_level and "N/A" not in hybrid_level and "skip" not in hybrid_level.lower()
        if has_base and has_hybrid:
            valid_comparison_subset.append(r)
            
    n_subset = len(valid_comparison_subset)
    print(f"Sub-population where both models executed: N = {n_subset} videos.")

    # =========================================================================
    # E1: Core Ablation Study (using real N=42 and N=18 subset)
    # =========================================================================
    print("Generating E1: Core Ablation results...")
    e1_dir = PROJECT_ROOT / "evaluation" / "results" / "e1"
    e1_dir.mkdir(parents=True, exist_ok=True)

    # Compute metrics for Hybrid on full set (N=42)
    gt_all = []
    hybrid_pred_all = []
    for r in records:
        gt_cat = r.get("Ground Truth Category", "Normal")
        gt_is_anomaly = 1 if gt_cat.lower() != "normal" else 0
        gt_all.append(gt_is_anomaly)
        
        ph = parse_threat_binary("", is_hybrid=True, raw_level=r.get("Threat Level", "Low"))
        hybrid_pred_all.append(ph)

    # Compute metrics on the intersection subset (N=18)
    gt_sub = []
    base_pred_sub = []
    hybrid_pred_sub = []
    for r in valid_comparison_subset:
        gt_cat = r.get("Ground Truth Category", "Normal")
        gt_is_anomaly = 1 if gt_cat.lower() != "normal" else 0
        gt_sub.append(gt_is_anomaly)
        
        pb = parse_threat_binary(r.get("Base Gemma Output", ""))
        ph = parse_threat_binary("", is_hybrid=True, raw_level=r.get("Threat Level", "Low"))
        base_pred_sub.append(pb)
        hybrid_pred_sub.append(ph)

    def compute_metrics(gt, pred):
        gt_np = np.array(gt)
        pred_np = np.array(pred)
        tp = int(np.sum((gt_np == 1) & (pred_np == 1)))
        fp = int(np.sum((gt_np == 0) & (pred_np == 1)))
        tn = int(np.sum((gt_np == 0) & (pred_np == 0)))
        fn = int(np.sum((gt_np == 1) & (pred_np == 0)))
        
        total = len(gt)
        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {
            "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1,
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
        }

    metrics_hybrid_full = compute_metrics(gt_all, hybrid_pred_all)
    metrics_base_sub = compute_metrics(gt_sub, base_pred_sub)
    metrics_hybrid_sub = compute_metrics(gt_sub, hybrid_pred_sub)

    # Save real confusion matrices
    plot_confusion_matrix(
        metrics_base_sub["confusion_matrix"]["tp"], metrics_base_sub["confusion_matrix"]["fp"],
        metrics_base_sub["confusion_matrix"]["tn"], metrics_base_sub["confusion_matrix"]["fn"],
        e1_dir, "confusion_matrix_base_subset"
    )
    plot_confusion_matrix(
        metrics_hybrid_full["confusion_matrix"]["tp"], metrics_hybrid_full["confusion_matrix"]["fp"],
        metrics_hybrid_full["confusion_matrix"]["tn"], metrics_hybrid_full["confusion_matrix"]["fn"],
        e1_dir, "confusion_matrix_hybrid_full"
    )

    # Grouped Bar Plot Comparison (strictly over the 18-video comparison subset where both ran)
    plot_bar_comparison(
        ["Accuracy", "Precision", "Recall", "F1-Score"],
        {
            f"Base Gemma (N={n_subset})": [metrics_base_sub["accuracy"]*100, metrics_base_sub["precision"]*100, metrics_base_sub["recall"]*100, metrics_base_sub["f1_score"]*100],
            f"Hybrid Model (N={n_subset})": [metrics_hybrid_sub["accuracy"]*100, metrics_hybrid_sub["precision"]*100, metrics_hybrid_sub["recall"]*100, metrics_hybrid_sub["f1_score"]*100]
        },
        ylabel="Score (%)", title="Head-to-Head Comparison on Common Executed Subset", output_dir=e1_dir, name="metrics_comparison_bar"
    )

    save_json_results({
        "full_hybrid_evaluation": metrics_hybrid_full,
        "subset_base_evaluation": metrics_base_sub,
        "subset_hybrid_evaluation": metrics_hybrid_sub,
        "dataset_metadata": {"total_count": n_total, "subset_count": n_subset}
    }, e1_dir, "ablation_results")

    headers = ["Model Configuration", "Sample Size (N)", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]
    rows = [
        ["Base Gemma (Common Subset)", str(n_subset), f"{metrics_base_sub['accuracy']*100:.1f}", f"{metrics_base_sub['precision']*100:.1f}", f"{metrics_base_sub['recall']*100:.1f}", f"{metrics_base_sub['f1_score']*100:.1f}"],
        ["Two-Stage Hybrid (Common Subset)", str(n_subset), f"{metrics_hybrid_sub['accuracy']*100:.1f}", f"{metrics_hybrid_sub['precision']*100:.1f}", f"{metrics_hybrid_sub['recall']*100:.1f}", f"{metrics_hybrid_sub['f1_score']*100:.1f}"],
        ["Two-Stage Hybrid (Full Dataset)", str(n_total), f"{metrics_hybrid_full['accuracy']*100:.1f}", f"{metrics_hybrid_full['precision']*100:.1f}", f"{metrics_hybrid_full['recall']*100:.1f}", f"{metrics_hybrid_full['f1_score']*100:.1f}"]
    ]
    latex_e1 = generate_latex_table(headers, rows, "ablation_study", "Core Ablation comparison over actual VLM runs.", e1_dir, "ablation_table")
    generate_markdown_report("Experiment E1: Core Ablation Study Report", {"Summary": "Core ablation benchmarks."}, {"LaTeX Table": latex_e1}, e1_dir, "E1_Core_Ablation_Report")

    # =========================================================================
    # E3: Statistical Significance (strictly over the 18 paired predictions)
    # =========================================================================
    print("Generating E3: Statistical Significance results...")
    e3_dir = PROJECT_ROOT / "evaluation" / "results" / "e3"
    e3_dir.mkdir(parents=True, exist_ok=True)

    # Contingency table calculation
    # Both correct (a), Base correct & Hybrid wrong (b), Base wrong & Hybrid correct (c), Both wrong (d)
    b_arr = np.array(base_pred_sub) == np.array(gt_sub)
    h_arr = np.array(hybrid_pred_sub) == np.array(gt_sub)
    
    a = int(np.sum(b_arr & h_arr))
    b = int(np.sum(b_arr & ~h_arr))
    c = int(np.sum(~b_arr & h_arr))
    d = int(np.sum(~b_arr & ~h_arr))

    # McNemar chi-square
    if (b + c) > 0:
        chi2 = (abs(b - c) - 1)**2 / (b + c)
        p_val = float(chi2_survival(chi2))
    else:
        chi2 = 0.0
        p_val = 1.0  # Identical predictions

    mcnemar = {"both_correct": a, "base_correct_only": b, "hybrid_correct_only": c, "both_wrong": d, "chi2_statistic": chi2, "p_value": p_val}

    # Bootstrap iterations over real data to get true standard error of accuracy difference
    np.random.seed(42)
    diffs = []
    for _ in range(1000):
        idx = np.random.choice(n_subset, size=n_subset, replace=True)
        boot_gt = np.array(gt_sub)[idx]
        boot_base = np.array(base_pred_sub)[idx]
        boot_hybrid = np.array(hybrid_pred_sub)[idx]
        
        base_acc = np.mean(boot_base == boot_gt)
        hybrid_acc = np.mean(boot_hybrid == boot_gt)
        diffs.append(hybrid_acc - base_acc)
        
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs))
    ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]

    bootstrap = {"mean_difference": mean_diff, "std_difference": std_diff, "confidence_interval_95": ci}

    save_json_results({"mcnemar": mcnemar, "bootstrap": bootstrap}, e3_dir, "expanded_eval_results")

    headers = ["Statistical Parameter", "Measured Value"]
    rows = [
        ["Contingency Cell: Both Correct (a)", str(a)],
        ["Contingency Cell: Base Correct Only (b)", str(b)],
        ["Contingency Cell: Hybrid Correct Only (c)", str(c)],
        ["Contingency Cell: Both Incorrect (d)", str(d)],
        ["McNemar Chi-Square Statistic", f"{chi2:.3f}"],
        ["McNemar p-value", f"{p_val:.4f}"],
        ["Bootstrap Accuracy Difference Mean", f"{mean_diff:.4f}"],
        ["Bootstrap Difference 95% Confidence Interval", f"[{ci[0]:.4f}, {ci[1]:.4f}]"]
    ]
    latex_e3 = generate_latex_table(headers, rows, "expanded_eval", "Statistical tests over the common executed subset.", e3_dir, "expanded_eval_table")
    generate_markdown_report("Experiment E3: Statistical Significance Report", {"Summary": "Statistical checks over real paired predictions."}, {"LaTeX Table": latex_e3}, e3_dir, "E3_Expanded_Evaluation_Report")

    # =========================================================================
    # E5: Pipeline Profiling & Latency (excluding outliers)
    # =========================================================================
    print("Generating E5: Latency Profiling results...")
    e5_dir = PROJECT_ROOT / "evaluation" / "results" / "e5"
    e5_dir.mkdir(parents=True, exist_ok=True)

    base_latencies = []
    hybrid_latencies_subset = []
    hybrid_latencies_all = []

    for r in records:
        vid = r.get("Video ID", "")
        raw_lat = r.get("Inference Time", "")
        if "|" not in raw_lat:
            continue
            
        parts = raw_lat.split("|")
        b_part = float(parts[0].replace("Base:", "").replace("s", "").strip())
        h_part = float(parts[1].replace("FT Guided:", "").replace("s", "").strip())
        
        # Exclude Base outlier (Burglary089_x264)
        if b_part > 0 and vid != "Burglary089_x264":
            base_latencies.append(b_part)
            # Exclude Hybrid outlier (Fighting013_x264)
            if vid != "Fighting013_x264":
                hybrid_latencies_subset.append(h_part)
                
        # Exclude Hybrid outlier (Fighting013_x264)
        if vid != "Fighting013_x264":
            hybrid_latencies_all.append(h_part)

    avg_base = float(np.mean(base_latencies)) if base_latencies else 0.0
    avg_hybrid_sub = float(np.mean(hybrid_latencies_subset)) if hybrid_latencies_subset else 0.0
    avg_hybrid_all = float(np.mean(hybrid_latencies_all)) if hybrid_latencies_all else 0.0

    save_json_results({
        "average_base_latency_sec": avg_base,
        "average_hybrid_latency_subset_sec": avg_hybrid_sub,
        "average_hybrid_latency_all_sec": avg_hybrid_all,
        "base_valid_runs": len(base_latencies),
        "hybrid_valid_runs": len(hybrid_latencies_all)
    }, e5_dir, "pipeline_profiling_results")

    # Generate real comparison plot
    fig, ax = plt.subplots(figsize=(6, 4))
    categories = [f"Base Gemma\n(N={len(base_latencies)})", f"Hybrid Model (Subset)\n(N={len(hybrid_latencies_subset)})", f"Hybrid Model (All)\n(N={len(hybrid_latencies_all)})"]
    lat_values = [avg_base, avg_hybrid_sub, avg_hybrid_all]
    ax.bar(categories, lat_values, color=["#e53935", "#1e88e5", "#0d47a1"], width=0.5)
    ax.set_ylabel("Average Inference Latency (seconds)")
    ax.set_title("Real Inference Latency Profile (Outliers Excluded)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    
    fig.savefig(e5_dir / "pipeline_flame_breakdown.png", dpi=300)
    fig.savefig(e5_dir / "pipeline_flame_breakdown.pdf")
    fig.savefig(e5_dir / "pipeline_flame_breakdown.svg")
    plt.close(fig)

    headers = ["Configuration", "Active Runs (N)", "Average Latency (s)", "Outlier Status"]
    rows = [
        ["Base Gemma", str(len(base_latencies)), f"{avg_base:.2f}", "Excluded Burglary089 (10527s)"],
        ["Two-Stage Hybrid (Common Subset)", str(len(hybrid_latencies_subset)), f"{avg_hybrid_sub:.2f}", "Excluded Fighting013 (2534s)"],
        ["Two-Stage Hybrid (Full Dataset)", str(len(hybrid_latencies_all)), f"{avg_hybrid_all:.2f}", "Excluded Fighting013 (2534s)"]
    ]
    latex_e5 = generate_latex_table(headers, rows, "pipeline_profiling", "Clean latency measurements.", e5_dir, "pipeline_profiling_table")
    generate_markdown_report("Experiment E5: Pipeline Profiling Report", {"Summary": "Pipeline latency metrics excluding hangs."}, {"LaTeX Table": latex_e5}, e5_dir, "E5_Pipeline_Profiling_Report")

    print("\nAll real statistical results compiled successfully!")

def chi2_survival(chi2):
    # Simple approximation of chi-squared CDF survival function (1 degree of freedom)
    # for p-values estimation
    import math
    if chi2 <= 0: return 1.0
    try:
        return math.erfc(math.sqrt(chi2) / math.sqrt(2.0))
    except Exception:
        return 0.0

if __name__ == "__main__":
    main()
