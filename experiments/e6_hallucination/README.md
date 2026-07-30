# Experiment E6: Hallucination Study

This experiment evaluates the factual reliability of the Two-Stage VLM pipeline reports. It extracts factual claims from generated VLM threat reports and automatically verifies them against the raw keyframes using a VLM-as-Judge setup.

## Metrics Computed
*   **Total Claims Extracted:** Number of testable assertions generated.
*   **Verified Factual Claims:** Number of assertions confirmed by the judge VLM.
*   **Contradicted / Hallucinated Claims:** Number of assertions contradicted by visual frames.
*   **Hallucination Rate:** Percentage of false claims generated.
*   **Factual Precision:** Accuracy of descriptive reports.

## How to Run
Run the experiment using the orchestrator:
```bash
python experiments/run.py --experiment E6
```

## Outputs Produced
*   `hallucination_study_results.json`: Extracted claims, verification decisions, and summary scores.
*   `hallucination_table.tex`: LaTeX code formatting the summary statistics.
*   `E6_Hallucination_Study_Report.md`: Consolidated summary report.
