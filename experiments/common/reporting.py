import csv
import json
from pathlib import Path
from typing import Any, Dict, List

def save_json_results(data: Dict[str, Any], output_dir: Path, name: str) -> None:
    """Save raw results data to a formatted JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{name}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_csv_results(records: List[Dict[str, Any]], fieldnames: List[str], output_dir: Path, name: str) -> None:
    """Save results list progressively to a CSV spreadsheet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

def generate_latex_table(
    headers: List[str],
    rows: List[List[Any]],
    label: str,
    caption: str,
    output_dir: Path,
    name: str
) -> str:
    """Generate a clean, publication-ready LaTeX table using booktabs styling."""
    col_format = "l" + "c" * (len(headers) - 1)
    
    latex_lines = []
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"  \centering")
    latex_lines.append(f"  \\caption{{{caption}}}")
    latex_lines.append(f"  \\label{{tab:{label}}}")
    latex_lines.append(f"  \\begin{{tabular}}{{{col_format}}}")
    latex_lines.append(r"    \toprule")
    
    # Headers
    latex_lines.append("    " + " & ".join(headers) + r" \\")
    latex_lines.append(r"    \midrule")
    
    # Rows
    for row in rows:
        formatted_row = [str(x) for x in row]
        latex_lines.append("    " + " & ".join(formatted_row) + r" \\")
        
    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"  \end{tabular}")
    latex_lines.append(r"\end{table}")
    
    latex_code = "\n".join(latex_lines)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    latex_path = output_dir / f"{name}.tex"
    with latex_path.open("w", encoding="utf-8") as f:
        f.write(latex_code)
        
    return latex_code

def generate_markdown_report(
    title: str,
    sections: Dict[str, str],
    tables_tex: Dict[str, str],
    output_dir: Path,
    name: str
) -> None:
    """Compile and write a clean, readable Markdown dashboard report."""
    md_lines = []
    md_lines.append(f"# {title}")
    md_lines.append("\n*Generated dynamically by the DRDO VLM Research framework*\n")
    
    for sec_title, content in sections.items():
        md_lines.append(f"## {sec_title}")
        md_lines.append(content)
        md_lines.append("\n")
        
    if tables_tex:
        md_lines.append("## LaTeX Tables code (Copy-Paste to Thesis)")
        for key, code in tables_tex.items():
            md_lines.append(f"### Table: {key}")
            md_lines.append("```latex")
            md_lines.append(code)
            md_lines.append("```\n")

    md_content = "\n".join(md_lines)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{name}.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md_content)
