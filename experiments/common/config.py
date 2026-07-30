import argparse
import os
import yaml
from pathlib import Path
from typing import Any, Dict

# Resolve project roots
COMMON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = COMMON_DIR.parent.parent

def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Helper to safely load a YAML configuration file."""
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Failed to load config at {config_path}: {e}")
        return {}

def get_config(experiment_name: str | None = None) -> Dict[str, Any]:
    """Load default global configs, experiment-specific configs, and parse CLI argument overrides."""
    parser = argparse.ArgumentParser(description="Defense-Grade Surveillance VLM Research Framework")
    parser.add_argument("--experiment", type=str, default=experiment_name, help="Name of the experiment (e.g. E1, E2)")
    parser.add_argument("--dataset-dir", type=str, default="datasets/raw/ucf-crime-mini", help="Path to ucf-crime-mini or XD-Violence")
    parser.add_argument("--output-dir", type=str, default="evaluation/results", help="Directory where experiment outputs are saved")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--backend", type=str, default=None, help="Inference backend (ollama or hf)")
    parser.add_argument("--model", type=str, default=None, help="Ollama model name (e.g. gemma3:4b)")
    parser.add_argument("--adapter-path", type=str, default="models/adapters/activitynet_v1", help="Path to fine-tuned adapters")
    parser.add_argument("--max-videos", type=int, default=100, help="Max videos to run in this experiment segment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline model run (useful for fast iterations)")

    # Allow custom arguments without crashing
    args, unknown = parser.parse_known_args()

    # Load global default config if exists
    global_config_path = PROJECT_ROOT / "configs" / "surveillance_training_config.yaml"
    config = load_yaml_config(global_config_path)

    # Load experiment specific config if experiment_name is defined
    exp_name = args.experiment or experiment_name
    if exp_name:
        exp_map = {
            "E1": "e1_core_ablation",
            "E2": "e2_frame_sampling",
            "E3": "e3_expanded_eval",
            "E4": "e4_quantization",
            "E5": "e5_pipeline_profiling",
            "E6": "e6_hallucination",
            "E10": "e10_prompt_eng",
            "E11": "e11_failure_analysis"
        }
        folder = exp_map.get(exp_name.upper(), exp_name.lower())
        exp_dir = PROJECT_ROOT / "experiments" / folder
        exp_config_path = exp_dir / "config.yaml"
        exp_config = load_yaml_config(exp_config_path)
        config.update(exp_config)

    # Apply command-line overrides
    config["experiment"] = exp_name
    config["dataset_dir"] = args.dataset_dir
    config["output_dir"] = args.output_dir
    config["device"] = args.device
    if args.backend:
        config["backend"] = args.backend
    else:
        config["backend"] = config.get("backend", os.getenv("INFERENCE_BACKEND", "ollama"))
        
    if args.model:
        config["model"] = args.model
    else:
        config["model"] = config.get("model", os.getenv("OLLAMA_MODEL", "gemma3:4b"))

    config["adapter_path"] = args.adapter_path
    config["max_videos"] = args.max_videos
    config["seed"] = args.seed
    config["skip_baseline"] = args.skip_baseline

    # Parse any unknown key=value overrides from command line
    for item in unknown:
        if item.startswith("--"):
            parts = item[2:].split("=")
            if len(parts) == 2:
                key, val = parts
                # Try to convert to int/float if applicable
                try:
                    if "." in val:
                        val = float(val)
                    else:
                        val = int(val)
                except ValueError:
                    if val.lower() == "true":
                        val = True
                    elif val.lower() == "false":
                        val = False
                config[key.replace("-", "_")] = val

    # Setup directories
    out_dir = PROJECT_ROOT / config["output_dir"]
    if exp_name:
        out_dir = out_dir / exp_name.lower()
    config["resolved_output_dir"] = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    return config
