# Experiment E11: Failure Analysis Study

This experiment automatically screens VLM predictions that mismatch ground-truth annotations, self-diagnosing the failure causes (e.g. low-light, occlusion, motion blur, temporal miss, wrong reasoning) using a VLM-as-Judge setup, and compiles the taxonomy distribution.

## Failure Categories
*   **Occlusion:** Target subject is hidden behind barriers.
*   **Low Light:** Scene has low visibility or night-time shadow issues.
*   **Motion Blur:** Fast movement causes camera frames to blur.
*   **Wrong Reasoning:** Model saw the visual cue but inferred incorrect intent.
*   **Temporal Miss:** Critical action was not captured in sampled keyframes.
*   **Ambiguous Event:** Visual event is unclear or open to interpretation.
*   **False Alarm:** Normal movement misclassified as threat.

## How to Run
Run via the global orchestrator CLI:
```bash
python experiments/run.py --experiment E11
```

## Outputs Produced
*   `failure_analysis_results.json`: Diagnostic reason, prediction, and details for each failed case.
*   `failure_table.tex`: LaTeX code formatting the failure categories.
*   `failure_analysis_pie.png`: High-resolution pie chart of failure reasons (300 DPI).
*   `E11_Failure_Analysis_Report.md`: Consolidated summary report and failed case gallery.
