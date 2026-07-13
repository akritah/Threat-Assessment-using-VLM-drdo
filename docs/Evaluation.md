# Evaluation Methodology & Performance Metrics

This document details the quantitative and qualitative evaluation framework used to test the model's threat assessment capabilities.

---

## 1. Evaluation Architecture

### Evaluation Module
*   **Path:** `evaluation/run_eval_xd.py`
*   **Purpose:** Compares the performance of the **Base Gemma 3** model (without adapter) against our **Two-Stage Fine-Tuned Gemma 3** pipeline.
*   **Dual-Mode Scanning:** Dynamically scans either flat folder frames (UCF-Crime) or raw MP4 clips (XD-Violence), extracting 8 representative frames.

---

## 2. Quantitative Classification Metrics

The system maps categories into binary security decisions:
*   **Ground Truth:**
    *   `Normal` $\rightarrow$ Negative (Low Threat)
    *   Any anomaly category (e.g. `Riot`, `Fighting`, `Abuse`) $\rightarrow$ Positive (Medium/High Threat)
*   **Predictions:**
    *   `Low` Threat $\rightarrow$ Negative
    *   `Medium` or `High` Threat $\rightarrow$ Positive

Using these outcomes, it computes:
*   **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$
*   **Precision:** $\frac{TP}{TP + FP}$
*   **Recall:** $\frac{TP}{TP + FN}$
*   **F1-Score:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

---

## 3. Visualizations & Charts

At the end of the evaluation run, `run_eval_xd.py` automatically generates comparison charts saved in `outputs/plots/` (using matplotlib):
1.  **Threat Distribution Chart (`threat_distribution.png`):** Shows the bar chart comparison of predicted threat categories (Low, Medium, High) between the Base model and the Fine-Tuned model.
2.  **Inference Latency:** Measures and plots the average inference time (seconds) per segment.

---

## 4. How to Execute Evaluation

Run the evaluation script by pointing it to the dataset directory and the trained adapter folder:

```bash
HF_TOKEN="your_token" python evaluation/run_eval_xd.py \
  --dataset-dir /kaggle/input/datasets/odins0n/ucf-crime-dataset \
  --adapter-path adapters/surveillance_colab \
  --device cuda \
  --max-eval-videos 100
```

### Deliverables Saved under `outputs/`:
*   `outputs/evaluation_results.csv`: Log of every video's ground truth, predicted activity, base/fine-tuned outputs, latencies, and threat levels.
*   `outputs/evaluation_report.md`: Markdown report summarizing overall metrics, confusion matrices, and representative failure/success cases.
*   `outputs/plots/`: Directory containing generated comparison plots.
