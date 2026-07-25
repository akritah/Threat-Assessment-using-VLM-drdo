import csv
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
csv_path = PROJECT_ROOT / "evaluation_results.csv"
plots_dir = PROJECT_ROOT / "evaluation" / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# 1. Parse threat levels
base_levels = {"Low": 0, "Medium": 0, "High": 0}
ft_levels = {"Low": 0, "Medium": 0, "High": 0}

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        # Base Model Threat Parsing (from raw text)
        base_out = r.get("Base Gemma Output", "").lower()
        base_level = "Low"
        for line in base_out.split("\n"):
            if "threat level" in line:
                if "high" in line:
                    base_level = "High"
                elif "medium" in line:
                    base_level = "Medium"
                break
        base_levels[base_level] += 1

        # Fine-Tuned Model Action-to-Threat Mapping
        ft_out = r.get("Fine-Tuned Gemma Output", "").lower()
        
        # Mapping rules based on the fine-tuned action vocabulary
        if any(w in ft_out for w in ["combat", "accident", "bombing", "karate", "violence", "assault"]):
            ft_level = "High"
        elif any(w in ft_out for w in ["drifting", "clandestine"]):
            ft_level = "Medium"
        else:
            ft_level = "Low"
        
        ft_levels[ft_level] += 1

# 2. Plotting the chart
categories = ["Low", "Medium", "High"]
base_counts = [base_levels[c] for c in categories]
ft_counts = [ft_levels[c] for c in categories]

x = range(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
# Use premium, polished color palette
ax.bar([i - width/2 for i in x], base_counts, width, label="Base Gemma 3 (Raw Output)", color="#90caf9")
ax.bar([i + width/2 for i in x], ft_counts, width, label="Fine-Tuned Gemma 3 (Mapped Actions)", color="#1565c0")

ax.set_ylabel("Number of Videos")
ax.set_title("Qualitative Threat Level Estimation Comparison")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
plt.tight_layout()

plot_path = plots_dir / "threat_distribution.png"
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Chart successfully regenerated at: {plot_path}")
print(f"Base Gemma distribution: {base_levels}")
print(f"Fine-Tuned Gemma distribution: {ft_levels}")
