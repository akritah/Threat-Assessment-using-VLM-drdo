import csv
from pathlib import Path

csv_path = Path("evaluation_results.csv")
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader, 1):
        print(f"\n--- Video {idx}: {row['Video ID']} ({row['Ground Truth Category']}) ---")
        print(f"Base Output:\n{row['Base Gemma Output'][:300]}...")
        print(f"FT Output:\n{row['Fine-Tuned Gemma Output']}")
