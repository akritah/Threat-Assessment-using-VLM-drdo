# Experiment E3: Expanded Evaluation

This experiment evaluates the Two-Stage VLM pipeline's statistical stability across multiple random seeds, generating mean and standard errors. It conducts McNemar's significance tests and paired bootstrap analyses to verify if the performance gains are statistically significant.

## Seeds Checked
*   42
*   100
*   2026

## How to Run
Run the experiment using the CLI:
```bash
python experiments/run.py --experiment E3
```

## Outputs Produced
*   `expanded_eval_results.json`: Raw results, mean, std, CI, McNemar statistics, and bootstrap confidence intervals.
*   `expanded_eval_table.tex`: Copy-pasteable LaTeX table showing the statistical performance bounds.
*   `E3_Expanded_Evaluation_Report.md`: Markdown summary report.
