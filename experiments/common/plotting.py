import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Enable publication-grade font settings
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "font.family": "sans-serif"
})

# central color palettes
PRIMARY_COLOR = "#1565c0"  # Dark Blue
SECONDARY_COLOR = "#90caf9"  # Light Blue
ACCENT_COLOR = "#d84315"  # Burnt Orange
NEUTRAL_DARK = "#212121"
NEUTRAL_LIGHT = "#f5f5f5"

def save_publication_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Save the figure as PNG, PDF, and vector SVG at 300 DPI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save standard formats
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)

def plot_confusion_matrix(tp: int, fp: int, tn: int, fn: int, output_dir: Path, name: str = "confusion_matrix") -> None:
    """Plot an annotated 2x2 confusion matrix heatmap."""
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    
    classes = ["Non-Threat", "Threat"]
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title="Confusion Matrix",
           ylabel="Ground Truth",
           xlabel="VLM Prediction")
    
    # Text annotations in the cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold")
            
    fig.tight_layout()
    save_publication_figure(fig, output_dir, name)

def plot_bar_comparison(categories: List[str], data_series: Dict[str, List[float]], ylabel: str, title: str, output_dir: Path, name: str) -> None:
    """Plot grouped bar charts for model comparison studies."""
    x = np.arange(len(categories))
    width = 0.8 / len(data_series)
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    colors = [PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, "#4caf50", "#9c27b0"]
    for idx, (label, values) in enumerate(data_series.items()):
        color = colors[idx % len(colors)]
        offset = x + (idx - len(data_series)/2 + 0.5) * width
        ax.bar(offset, values, width, label=label, color=color)
        
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    fig.tight_layout()
    save_publication_figure(fig, output_dir, name)

def plot_pareto_frontier(accuracies: List[float], latencies: List[float], frame_counts: List[int], output_dir: Path, name: str) -> None:
    """Plot Latency vs Accuracy scatter with Pareto optimal frontier curve."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    ax.scatter(latencies, accuracies, color=ACCENT_COLOR, s=80, zorder=3, label="Evaluated Sample Points")
    for i, count in enumerate(frame_counts):
        ax.annotate(f"{count} Frames", (latencies[i], accuracies[i]), textcoords="offset points", xytext=(5,5), ha="left", fontweight="bold")

    # Sort to draw frontier line
    sorted_idx = np.argsort(latencies)
    sorted_latencies = np.array(latencies)[sorted_idx]
    sorted_accuracies = np.array(accuracies)[sorted_idx]
    
    # Trace Pareto frontier
    p_latencies = [sorted_latencies[0]]
    p_accuracies = [sorted_accuracies[0]]
    for i in range(1, len(sorted_idx)):
        if sorted_accuracies[i] > p_accuracies[-1]:
            p_latencies.append(sorted_latencies[i])
            p_accuracies.append(sorted_accuracies[i])
            
    ax.plot(p_latencies, p_accuracies, linestyle="-.", color=PRIMARY_COLOR, alpha=0.8, label="Pareto Frontier")
    
    ax.set_xlabel("Average Latency (seconds)")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_title("Accuracy vs. Latency Trade-Off Study")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    
    fig.tight_layout()
    save_publication_figure(fig, output_dir, name)

def plot_radar_chart(categories: List[str], data: Dict[str, List[float]], output_dir: Path, name: str) -> None:
    """Plot multi-variate radar / spider charts for prompt comparison studies."""
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # close the loop
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    colors = [PRIMARY_COLOR, ACCENT_COLOR, "#4caf50", "#9c27b0"]
    for idx, (label, values) in enumerate(data.items()):
        color = colors[idx % len(colors)]
        val_list = list(values)
        val_list += val_list[:1]  # close the loop
        
        ax.plot(angles, val_list, linewidth=2, linestyle="solid", label=label, color=color)
        ax.fill(angles, val_list, color=color, alpha=0.15)
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    
    fig.tight_layout()
    save_publication_figure(fig, output_dir, name)

def plot_flame_profile(stages: List[str], time_slots: List[float], output_dir: Path, name: str) -> None:
    """Plot horizontal timing profile stacked bar chart (flame timing breakdown)."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    
    y_pos = np.arange(len(stages))
    
    # Beautiful horizontal bar
    bars = ax.barh(y_pos, time_slots, align="center", color=PRIMARY_COLOR, height=0.55)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages)
    ax.invert_yaxis()  # top-down view
    ax.set_xlabel("Time (seconds)")
    ax.set_title("VLM Pipeline Execution Profile Breakdown")
    
    # Value annotations on the right of bars
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f"{width:.2f}s",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),  # 5 points horizontal offset
                    textcoords="offset points",
                    ha="left", va="center", fontweight="bold")
                    
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    
    fig.tight_layout()
    save_publication_figure(fig, output_dir, name)

def plot_pie_chart(labels: List[str], counts: List[int], title: str, output_dir: Path, name: str) -> None:
    """Plot a clean, readable failure analysis pie chart."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    colors = ["#e53935", "#fb8c00", "#fdd835", "#43a047", "#1e88e5", "#8e24aa", "#546e7a"]
    
    # Remove categories with zero instances to keep pie chart readable
    non_zero = [(l, c) for l, c in zip(labels, counts) if c > 0]
    if not non_zero:
        non_zero = [("No Anomalies/Failures", 1)]
        
    f_labels, f_counts = zip(*non_zero)
    
    wedges, texts, autotexts = ax.pie(f_counts, labels=f_labels, autopct="%1.1f%%",
                                      startangle=140, colors=colors[:len(f_labels)],
                                      textprops=dict(color="black"))
                                      
    # Make value numbers bold and readable
    for autotext in autotexts:
        autotext.set_fontweight("bold")
        
    ax.set_title(title)
    fig.tight_layout()
    save_publication_figure(fig, output_dir, name)
