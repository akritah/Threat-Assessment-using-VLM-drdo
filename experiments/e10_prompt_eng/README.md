# Experiment E10: Prompt Engineering Study

This experiment evaluates the performance of different prompt engineering strategies on the Gemma-3-4B model. It measures formatting compliance, response latency, semantic quality, and hallucination rates across five prompt templates.

## Prompt Types Benchmarked
1.  **Baseline:** Basic descriptive prompt.
2.  **Few-Shot:** Context-guided examples inside the prompt.
3.  **JSON:** Strict instructions for structured JSON output.
4.  **Chain-of-Thought (CoT):** Step-by-step thinking instructions before final output.
5.  **Structured Reasoning (Proposed):** Multilayered description, threat indicator search, and threat level assessment.

## How to Run
Run via the CLI:
```bash
python experiments/run.py --experiment E10
```

## Outputs Produced
*   `prompt_engineering_results.json`: Output scores for all prompt variants.
*   `prompt_engineering_table.tex`: LaTeX table for publication.
*   `prompt_radar_comparison.png`: Radar / spider chart comparing dimensions of prompt styles (300 DPI).
*   `E10_Prompt_Engineering_Report.md`: Consolidated summary report.
