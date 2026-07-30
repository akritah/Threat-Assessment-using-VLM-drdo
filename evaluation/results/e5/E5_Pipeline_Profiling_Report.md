# Experiment E5: Pipeline Profiling Report

*Generated dynamically by the DRDO VLM Research framework*

## Summary
Pipeline latency metrics excluding hangs.


## LaTeX Tables code (Copy-Paste to Thesis)
### Table: LaTeX Table
```latex
\begin{table}[htbp]
  \centering
  \caption{Clean latency measurements.}
  \label{tab:pipeline_profiling}
  \begin{tabular}{lccc}
    \toprule
    Configuration & Active Runs (N) & Average Latency (s) & Outlier Status \\
    \midrule
    Base Gemma & 17 & 127.17 & Excluded Burglary089 (10527s) \\
    Two-Stage Hybrid (Common Subset) & 16 & 125.49 & Excluded Fighting013 (2534s) \\
    Two-Stage Hybrid (Full Dataset) & 41 & 107.05 & Excluded Fighting013 (2534s) \\
    \bottomrule
  \end{tabular}
\end{table}
```
