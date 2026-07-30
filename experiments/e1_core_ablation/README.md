# Experiment E1: Core Ablation Study

This experiment evaluates the core architectural decision of the two-stage VLM pipeline. It contrasts three different configurations to isolate the contributions of the fine-tuned action captioner and the base reasoning model.

## Model Configurations Checked
1.  **Base Gemma (Baseline):** The frozen pretrained base VLM queried with a direct, comprehensive threat analysis prompt.
2.  **Single-Stage Fine-Tuned:** The fine-tuned Gemma model queried with a simple activity description prompt (this model suffers from *instruction collapse* under complex reasoning prompts).
3.  **Two-Stage Hybrid (Proposed):**
    *   *Stage 1:* The Fine-Tuned captioner outputs a concise action caption.
    *   *Stage 2:* The Base model uses the caption as a semantic constraint alongside the visual frames to perform threat reasoning.

## How to Run
Trigger this experiment using the global orchestrator CLI:
```bash
python experiments/run.py --experiment E1
```

## Outputs Produced
*   `ablation_runs.json`: Raw VLM outputs, threat assessments, and latencies per video.
*   `ablation_table.tex`: Copy-pasteable LaTeX table showing the F1, Recall, Precision, and Accuracy for each of the configurations.
*   `confusion_matrix_*.png`: Confusion matrices (Base, SFT, Hybrid).
*   `metrics_comparison_bar.png`: Publication-ready 300 DPI bar chart.
*   `E1_Core_Ablation_Report.md`: Consolidated summary report.
