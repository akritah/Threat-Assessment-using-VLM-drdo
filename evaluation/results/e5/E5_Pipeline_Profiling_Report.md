# Experiment E5: Pipeline Profiling Report

*Generated dynamically by the DRDO VLM Research framework*

## Summary
Pipeline latency bottlenecks.


## LaTeX Tables code (Copy-Paste to Thesis)
### Table: LaTeX Table
```latex
\begin{table}[htbp]
  \centering
  \caption{Pipeline execution bottlenecks.}
  \label{tab:pipeline_profiling}
  \begin{tabular}{lcc}
    \toprule
    Pipeline Stage & Avg Duration (s) & Execution Share (%) \\
    \midrule
    Frame Extraction & 0.1200 & 2.28% \\
    Stage 1 (VLM Caption) & 1.2500 & 23.76% \\
    Stage 2 (VLM Reason) & 3.8400 & 73.00% \\
    Post-Processing & 0.0500 & 0.95% \\
    Total Pipeline & 5.2600 & 100.00% \\
    \bottomrule
  \end{tabular}
\end{table}
```
