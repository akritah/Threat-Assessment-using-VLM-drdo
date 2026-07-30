# Experiment E5: Pipeline Profiling

This experiment profiles the processing time and system resource overhead for each phase of the VLM surveillance pipeline.

## Stages Profiled
1.  **Frame Extraction:** Sampling evenly-spaced keyframes via OpenCV.
2.  **Stage 1 (VLM Captioning):** Fine-Tuned caption model execution.
3.  **Stage 2 (VLM Guided Reasoning):** Base model reasoning execution.
4.  **Post-Processing:** Compiling the outputs, checking schema compliance, and logging.

## How to Run
Run via the CLI:
```bash
python experiments/run.py --experiment E5
```

## Outputs Produced
*   `pipeline_profiling_results.json`: Execution time and percentage share per stage.
*   `pipeline_profiling_table.tex`: LaTeX code for publication.
*   `pipeline_flame_breakdown.png`: Stacked horizontal profiling bar chart (300 DPI).
*   `E5_Pipeline_Profiling_Report.md`: Consolidated summary report.
