# Experiment E4: Quantization Study Report

*Generated dynamically by the DRDO VLM Research framework*

## Summary
Quantization results.


## LaTeX Tables code (Copy-Paste to Thesis)
### Table: LaTeX Table
```latex
\begin{table}[htbp]
  \centering
  \caption{Gemma precision quantization benchmarks.}
  \label{tab:quantization_study}
  \begin{tabular}{lcccc}
    \toprule
    Precision Mode & Load Time (s) & Memory Footprint (MB) & Avg Latency (s) & Accuracy (%) \\
    \midrule
    FP16 & 42.50 & 8600.0 & 3.84 & 80.0 \\
    INT8 & 24.20 & 4800.0 & 5.12 & 79.5 \\
    NF4 & 12.80 & 2800.0 & 2.10 & 78.5 \\
    \bottomrule
  \end{tabular}
\end{table}
```
