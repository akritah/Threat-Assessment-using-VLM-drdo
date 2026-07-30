# Experiment E4: Quantization Study

This experiment evaluates the resource utilization and execution latency of the Gemma-3-4B model under three key quantization precisions: FP16, INT8, and NF4.

## Precision Modes Benchmarked
*   **FP16:** Standard 16-bit floating point precision.
*   **INT8:** 8-bit quantization.
*   **NF4:** Normal Float 4-bit quantization (standard QLoRA setup).

## How to Run
Run via the global orchestrator CLI:
```bash
python experiments/run.py --experiment E4
```

## Outputs Produced
*   `quantization_results.json`: Memory, latency, load time, and accuracy for each mode.
*   `quantization_table.tex`: LaTeX table summarizing the metrics.
*   `quantization_latency.png`: Inference latency bar chart.
*   `quantization_memory.png`: Memory footprint bar chart.
*   `E4_Quantization_Report.md`: Consolidated summary report.
