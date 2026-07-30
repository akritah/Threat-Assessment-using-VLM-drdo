# Experiment E2: Frame Sampling Study

This experiment evaluates the optimal number of keyframes required for the two-stage VLM surveillance pipeline. It establishes the accuracy-to-latency trade-off and automatically identifies the Pareto optimal configurations.

## Frame Counts Checked
*   4 frames
*   8 frames
*   16 frames
*   32 frames

## How to Run
Trigger the experiment via the global runner:
```bash
python experiments/run.py --experiment E2
```

## Outputs Produced
*   `frame_sampling_results.json`: Summary stats for each frame count configuration.
*   `frame_sampling_table.tex`: LaTeX code for publication-ready results.
*   `accuracy_latency_pareto.png`: Accuracy vs. Latency scatter plot with the Pareto frontier curves.
*   `memory_vs_frames.png`: Peak CPU RAM / GPU VRAM utilization line chart.
*   `E2_Frame_Sampling_Report.md`: Consolidated summary report.
