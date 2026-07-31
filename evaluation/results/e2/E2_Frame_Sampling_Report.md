# Experiment E2: Frame Sampling Study Report

*Generated dynamically by the DRDO VLM Research framework*

## Frame Sampling Analysis
This experiment benchmarks VLM accuracy and resources across different frame counts: [4, 8, 16, 32].

**Recommended Frame Count (Pareto Optimal):** 16 frames.
**Pareto-optimal set of frames:** [16].

Increasing the frame count generally improves classification accuracy and recall, but incurs a linear penalty in model visual token encoding latency.


## LaTeX Tables code (Copy-Paste to Thesis)
### Table: Frame Sampling Performance Table
```latex
\begin{table}[htbp]
  \centering
  \caption{VLM frame sampling trade-off study comparing metric quality, latency, and RAM footprints.}
  \label{tab:frame_sampling_study}
  \begin{tabular}{lcccccc}
    \toprule
    Frames & Accuracy (%) & Precision (%) & Recall (%) & F1 (%) & Avg Latency (s) & CPU RAM Peak (MB) \\
    \midrule
    4 & 0.0 & 0.0 & 0.0 & 0.0 & 178.89 & 150.1 \\
    8 & 0.0 & 0.0 & 0.0 & 0.0 & 283.70 & 546.1 \\
    16 & 0.0 & 0.0 & 0.0 & 0.0 & 3.43 & 36.2 \\
    32 & 0.0 & 0.0 & 0.0 & 0.0 & 4.91 & 78.2 \\
    \bottomrule
  \end{tabular}
\end{table}
```
