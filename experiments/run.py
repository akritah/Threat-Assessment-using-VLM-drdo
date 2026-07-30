import sys
import argparse
from pathlib import Path
import importlib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description="DRDO VLM Surveillance Experiments Runner CLI")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name (e.g. E1, E2, E3, E4, E5, E6, E10, E11)")
    args, unknown = parser.parse_known_args()
    
    exp_name = args.experiment.upper()
    
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
    
    if exp_name not in exp_map:
        print(f"Error: Unknown experiment name '{exp_name}'. Must be one of: {list(exp_map.keys())}")
        sys.exit(1)
        
    folder = exp_map[exp_name]
    print(f"==================================================")
    print(f"Starting VLM Benchmarking: {exp_name} ({folder})")
    print(f"==================================================")
    
    module_path = f"experiments.{folder}.run"
    try:
        # Dynamic import of specific experiment runner module
        exp_module = importlib.import_module(module_path)
        if hasattr(exp_module, "run_experiment"):
            exp_module.run_experiment()
        elif hasattr(exp_module, "main"):
            exp_module.main()
        else:
            print(f"Error: Module {module_path} has no run_experiment() or main() entrypoint.")
            sys.exit(1)
    except Exception as e:
        print(f"Failed to execute experiment {exp_name}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
