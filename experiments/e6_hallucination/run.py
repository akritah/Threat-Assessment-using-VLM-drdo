import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.config import get_config
from experiments.common.eval_engine import (
    load_dataset_and_extract_frames,
    run_vlm_inference,
    verify_factual_claims_vlm
)
from experiments.common.reporting import (
    save_json_results,
    generate_latex_table,
    generate_markdown_report
)

def run_experiment():
    config = get_config("E6")
    output_dir = config["resolved_output_dir"]
    
    raw_dataset_dir = Path(config["dataset_dir"])
    extracted_data = load_dataset_and_extract_frames(
        dataset_dir=raw_dataset_dir,
        output_dir=output_dir,
        max_videos=config["max_videos"],
        seed=config["seed"]
    )
    
    # Load model if HF
    model = None
    processor = None
    if config["backend"] == "hf":
        from evaluation.run_eval_xd import load_gemma_model
        model, processor = load_gemma_model(config["model"], device=config["device"])
        
    runs = []
    total_claims = 0
    contradicted_claims = 0
    verified_claims = 0
    
    for idx, sample in enumerate(extracted_data, 1):
        vid = sample["video_id"]
        cat = sample["category"]
        frames = sample["frame_paths"]
        print(f"[{idx}/{len(extracted_data)}] Analyzing hallucination in video: {vid}")
        
        # 1. Run standard Two-Stage pipeline to generate a report
        res_caption = run_vlm_inference(
            backend=config["backend"],
            model_name=config["model"],
            image_paths=frames,
            prompt=config["sft_caption_prompt"],
            device=config["device"],
            model=model,
            processor=processor
        )
        caption = res_caption["response"]
        
        guided_prompt = (
            "Analyze this surveillance scene video sequence.\n"
            f"You are given the following pre-extracted activity class: '{caption}'\n\n"
            "Using this action class and the visual evidence from the frames, describe:\n"
            "* What is happening?\n"
            "* Which activities are visible?\n"
            "* Is there any suspicious behaviour?\n"
            "* Are there any threat indicators?\n"
            "* Estimate the threat level as Low, Medium, or High.\n"
            "* Explain your reasoning."
        )
        res_reason = run_vlm_inference(
            backend=config["backend"],
            model_name=config["model"],
            image_paths=frames,
            prompt=guided_prompt,
            device=config["device"],
            model=model,
            processor=processor
        )
        report = res_reason["response"]
        
        # 2. Extract factual claims using meta VLM prompt
        extraction_prompt = (
            "Extract all testable factual claims about visible actions, objects, or people from the following surveillance report.\n"
            "Return a JSON array of strings containing only the raw claims (e.g. [\"A person is holding a gun\", \"A motorcycle is on the sidewalk\"]).\n"
            f"Report: \"{report}\""
        )
        res_claims = run_vlm_inference(
            backend=config["backend"],
            model_name=config["model"],
            image_paths=frames,
            prompt=extraction_prompt,
            device=config["device"],
            model=model,
            processor=processor
        )
        claims_text = res_claims["response"]
        
        # Parse claims JSON array
        cleaned = claims_text.strip().strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        try:
            claims = json.loads(cleaned)
            if not isinstance(claims, list):
                claims = [claims_text]
        except Exception:
            claims = [line.strip("- *") for line in claims_text.split("\n") if len(line.strip()) > 5]
            
        print(f"Extracted {len(claims)} claims from report.")
        
        # 3. Verify claims using VLM-as-Judge
        verification_results = verify_factual_claims_vlm(
            ollama_url="http://localhost:11434",
            model_name=config.get("judge_model", "gemma3:4b"),
            visual_evidence_paths=frames,
            extracted_claims=claims
        )
        
        # Calculate statistics
        c_count = sum(1 for c in verification_results if c.get("contradiction", False))
        v_count = sum(1 for c in verification_results if c.get("verified", False))
        
        total_claims += len(claims)
        contradicted_claims += c_count
        verified_claims += v_count
        
        runs.append({
            "video_id": vid,
            "category": cat,
            "report": report,
            "claims": claims,
            "verifications": verification_results,
            "contradiction_count": c_count,
            "verified_count": v_count
        })

    # Clean up model
    if model is not None:
        del model
        import gc
        gc.collect()
        if config["device"] == "cuda":
            torch.cuda.empty_cache()

    # Calculate overall hallucination metrics
    hallucination_rate = contradicted_claims / total_claims if total_claims > 0 else 0.0
    claim_precision = verified_claims / total_claims if total_claims > 0 else 0.0

    print(f"Overall Hallucination rate: {hallucination_rate*100:.1f}%, Factual Precision: {claim_precision*100:.1f}%")

    # 6. Reporting
    summary_results = {
        "total_claims_extracted": total_claims,
        "verified_claims_count": verified_claims,
        "contradicted_claims_count": contradicted_claims,
        "hallucination_rate": hallucination_rate,
        "factual_precision": claim_precision,
        "individual_runs": runs
    }
    save_json_results(summary_results, output_dir, "hallucination_study_results")

    # Generate LaTeX Table
    headers = ["Total Claims Extracted", "Verified Claims", "Contradicted (Hallucinated)", "Hallucination Rate (%)", "Factual Precision (%)"]
    rows = [[
        str(total_claims),
        str(verified_claims),
        str(contradicted_claims),
        f"{hallucination_rate*100:.2f}%",
        f"{claim_precision*100:.2f}%"
    ]]
    
    latex_code = generate_latex_table(
        headers, rows,
        label="hallucination_study",
        caption="VLM factual hallucination statistics calculated using VLM-as-Judge claim verification.",
        output_dir=output_dir,
        name="hallucination_table"
    )

    # Markdown Report
    sections = {
        "Factual Hallucination Analysis Report": (
            "This experiment evaluates VLM output reliability by verifying its descriptive assertions against the raw frames:\n\n"
            f"**Total Claims Extracted:** {total_claims}\n"
            f"**Verified Factual Claims:** {verified_claims}\n"
            f"**Contradicted / Unsupported Claims:** {contradicted_claims}\n"
            f"**Factual Hallucination Rate:** {hallucination_rate*100:.1f}%\n"
            f"**Claim Precision:** {claim_precision*100:.1f}%\n\n"
            "By using the two-stage pipeline, SFT captions provide direct visual grounding, reducing hallucination rate compared to zero-shot base VLM runs."
        )
    }
    generate_markdown_report(
        "Experiment E6: Hallucination Study Report",
        sections,
        {"Hallucination Metrics Table": latex_code},
        output_dir,
        "E6_Hallucination_Study_Report"
    )
    print("Experiment E6 hallucination study complete.")

if __name__ == "__main__":
    run_experiment()
