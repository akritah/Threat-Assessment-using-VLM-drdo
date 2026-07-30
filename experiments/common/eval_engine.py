import time
import os
import gc
import json
import base64
import requests
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image
import numpy as np

# Try importing psutil for CPU/RAM profiling, fallback to mock/basic if missing
try:
    import psutil
except ImportError:
    psutil = None

import torch

# Centralized VLM inference connector
def run_vlm_inference(
    backend: str,
    model_name: str,
    image_paths: List[Path],
    prompt: str,
    device: str = "cpu",
    model: Any = None,
    processor: Any = None,
    temperature: float = 0.0,
    max_tokens: int = 256
) -> Dict[str, Any]:
    """Execute VLM query, measuring latency, token usage, and hardware metrics."""
    metrics = {
        "latency_sec": 0.0,
        "token_count": 0,
        "cpu_usage_pct": 0.0,
        "peak_ram_mb": 0.0,
        "vram_allocated_mb": 0.0,
        "vram_cached_mb": 0.0
    }

    # Record hardware state before running
    ram_before = psutil.virtual_memory().used / (1024 * 1024) if psutil else 0.0
    cpu_before = psutil.cpu_percent(interval=None) if psutil else 0.0
    
    if device == "cuda" and torch.cuda.is_available():
        vram_alloc_before = torch.cuda.memory_allocated() / (1024 * 1024)
        vram_cache_before = torch.cuda.memory_reserved() / (1024 * 1024)
    else:
        vram_alloc_before = 0.0
        vram_cache_before = 0.0

    start_time = time.perf_counter()
    response = ""

    if backend == "ollama":
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        base64_images = []
        
        # Resizing and base64 encoding keyframes
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                img_resized = img.resize((448, 448), Image.Resampling.LANCZOS)
                buffered = BytesIO()
                img_resized.save(buffered, format="JPEG", quality=85)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                base64_images.append(img_str)
            except Exception as e:
                print(f"Warning: Frame process failed for {path}: {e}")

        payload = {
            "model": model_name,
            "prompt": prompt,
            "images": base64_images,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            r = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=180)
            if r.status_code == 200:
                response = r.json().get("response", "").strip()
            else:
                response = f"Error: Ollama returned code {r.status_code}"
        except Exception as e:
            response = f"Error connecting to Ollama: {e}"

    elif backend == "hf" and model is not None and processor is not None:
        content_list = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            content_list.append({"type": "image", "image": img})
        content_list.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content_list}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        if device == "cuda":
            inputs = {k: (v.to(torch.float16) if v.dtype == torch.float32 else v) for k, v in inputs.items()}
        else:
            inputs = {k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=(temperature > 0.0), temperature=temperature if temperature > 0.0 else None)
        
        input_len = inputs["input_ids"].shape[-1]
        response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

    else:
        response = "Error: Invalid backend or model configuration specified."

    metrics["latency_sec"] = time.perf_counter() - start_time
    
    # Approximate token count (4 chars = 1 token rule-of-thumb if exact tokenizer unavailable)
    metrics["token_count"] = len(response) // 4

    # Record hardware state after running
    ram_after = psutil.virtual_memory().used / (1024 * 1024) if psutil else 0.0
    cpu_after = psutil.cpu_percent(interval=None) if psutil else 0.0
    metrics["cpu_usage_pct"] = max(0.0, cpu_after - cpu_before) if psutil else 0.0
    metrics["peak_ram_mb"] = max(0.0, ram_after - ram_before) if psutil else 0.0

    if device == "cuda" and torch.cuda.is_available():
        vram_alloc_after = torch.cuda.memory_allocated() / (1024 * 1024)
        vram_cache_after = torch.cuda.memory_reserved() / (1024 * 1024)
        metrics["vram_allocated_mb"] = max(0.0, vram_alloc_after - vram_alloc_before)
        metrics["vram_cached_mb"] = max(0.0, vram_cache_after - vram_cache_before)

    return {"response": response, "metrics": metrics}

# Compute evaluation scores
def compute_classification_metrics(gt_labels: List[int], pred_labels: List[int]) -> Dict[str, Any]:
    """Calculate Accuracy, Precision, Recall, F1, and Confusion Matrix boundaries."""
    gt = np.array(gt_labels)
    pred = np.array(pred_labels)
    
    tp = int(np.sum((gt == 1) & (pred == 1)))
    fp = int(np.sum((gt == 0) & (pred == 1)))
    tn = int(np.sum((gt == 0) & (pred == 0)))
    fn = int(np.sum((gt == 1) & (pred == 0)))

    accuracy = (tp + tn) / len(gt) if len(gt) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    }

# Statistical Significance Tests
def run_mcnemar_test(gt: List[int], model_a_preds: List[int], model_b_preds: List[int]) -> Dict[str, Any]:
    """Execute McNemar test for comparing two binary classification models."""
    # Contingency Table:
    #                 Model B Correct    Model B Incorrect
    # Model A Correct       n00                n01
    # Model A Incorrect     n10                n11
    n01 = 0
    n10 = 0
    
    for real, a, b in zip(gt, model_a_preds, model_b_preds):
        a_correct = (real == a)
        b_correct = (real == b)
        if a_correct and not b_correct:
            n01 += 1
        elif not a_correct and b_correct:
            n10 += 1

    # Compute chi-squared statistic with continuity correction
    if n01 + n10 > 0:
        stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    else:
        stat = 0.0

    # Quick estimate of p-value using chi2 CDF (approximate 1 degree of freedom)
    # Using simple numerical approximation if scipy is not loaded
    try:
        from scipy import stats
        p_value = 1.0 - stats.chi2.cdf(stat, 1)
    except ImportError:
        # Standard lookup values for quick analysis
        if stat > 3.84:
            p_value = 0.05
        if stat > 6.63:
            p_value = 0.01
        else:
            p_value = 0.50

    return {
        "n01": n01,
        "n10": n10,
        "chi2_statistic": stat,
        "p_value": p_value,
        "significant_05": p_value < 0.05
    }

def run_paired_bootstrap(gt: List[int], model_a_preds: List[int], model_b_preds: List[int], num_samples: int = 1000) -> Dict[str, Any]:
    """Run bootstrap resampling to establish standard errors and statistical confidence intervals."""
    scores_diff = []
    
    gt = np.array(gt)
    a = np.array(model_a_preds)
    b = np.array(model_b_preds)
    
    for _ in range(num_samples):
        indices = np.random.choice(len(gt), size=len(gt), replace=True)
        acc_a = np.mean(gt[indices] == a[indices])
        acc_b = np.mean(gt[indices] == b[indices])
        scores_diff.append(acc_b - acc_a)

    scores_diff = np.array(scores_diff)
    mean_diff = float(np.mean(scores_diff))
    std_diff = float(np.std(scores_diff))
    ci_lower = float(np.percentile(scores_diff, 2.5))
    ci_upper = float(np.percentile(scores_diff, 97.5))

    return {
        "mean_difference": mean_diff,
        "std_difference": std_diff,
        "confidence_interval_95": (ci_lower, ci_upper)
    }

# Factual Claim Verification (VLM-as-Judge)
def verify_factual_claims_vlm(
    ollama_url: str,
    model_name: str,
    visual_evidence_paths: List[Path],
    extracted_claims: List[str]
) -> List[Dict[str, Any]]:
    """Verify list of claims against visual frames using a VLM-as-Judge setup."""
    results = []
    for claim in extracted_claims:
        prompt = (
            f"You are a strict, objective AI research judge. Verify the following factual claim:\n"
            f"Claim: \"{claim}\"\n\n"
            "Analyze the image sequence. Determine if this claim is physically present/correct based only on visual evidence.\n"
            "Return JSON with these keys:\n"
            "- verified: true/false\n"
            "- confidence: float value from 0.0 to 1.0\n"
            "- contradiction: true/false\n"
            "- evidence_description: short string describing what is visible to back up your decision."
        )
        
        # Load the middle frame as judge focus
        res = run_vlm_inference("ollama", model_name, visual_evidence_paths, prompt)
        text = res["response"]
        
        # Parse JSON output from VLM
        cleaned = text.strip().strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
            
        try:
            parsed = json.loads(cleaned)
        except Exception:
            parsed = {
                "verified": "true" in cleaned.lower(),
                "confidence": 0.5,
                "contradiction": "contradiction" in cleaned.lower(),
                "evidence_description": "Failed to parse judge JSON"
            }
            
        parsed["claim"] = claim
        results.append(parsed)
        
    return results

def load_dataset_and_extract_frames(
    dataset_dir: Path,
    output_dir: Path,
    max_videos: int,
    num_frames: int = 8,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Scan dataset directory, select random subset, and extract keyframes."""
    import random
    test_root = dataset_dir / "Test"
    if not test_root.exists():
        if (dataset_dir / "videos").exists():
            test_root = dataset_dir / "videos"
        else:
            test_root = dataset_dir
        
    if not test_root.exists():
        # Fallback to local paths
        test_root = Path(__file__).resolve().parent.parent.parent / "datasets" / "raw" / "ucf-crime-mini"
        if (test_root / "videos").exists():
            test_root = test_root / "videos"

    if not test_root.exists():
        # Create mock data if dataset doesn't exist
        mock_data = []
        for cat in ["Abuse", "Fighting", "Normal"]:
            mock_dir = output_dir / "extracted_frames" / f"Mock_{cat}_001"
            mock_dir.mkdir(parents=True, exist_ok=True)
            mock_frames = []
            for i in range(num_frames):
                frame_path = mock_dir / f"frame_{i}.jpg"
                img = Image.new("RGB", (224, 224), color=(100 + i*20, 50, 50))
                img.save(frame_path, "JPEG")
                mock_frames.append(frame_path)
            mock_data.append({
                "video_id": f"Mock_{cat}_001",
                "category": cat,
                "frame_paths": mock_frames
            })
        return mock_data[:max_videos]

    # Determine mode
    mode = "png"
    for category_dir in test_root.iterdir():
        if category_dir.is_dir():
            for entry in category_dir.iterdir():
                if entry.is_dir():
                    mode = "png"
                    break
                elif entry.is_file() and entry.name.lower().endswith(('.mp4', '.avi', '.mkv', '.webm')):
                    mode = "video"
                    break
            break

    video_groups = []
    if mode == "video":
        for category_dir in test_root.iterdir():
            if not category_dir.is_dir():
                continue
            category = category_dir.name
            count = 0
            for entry in category_dir.iterdir():
                if entry.is_file() and entry.name.lower().endswith(('.mp4', '.avi', '.mkv', '.webm')):
                    video_groups.append((entry.name.split(".")[0], category, entry))
                    count += 1
    else:
        for category_dir in test_root.iterdir():
            if not category_dir.is_dir():
                continue
            category = category_dir.name
            video_groups_dict = {}
            for entry in category_dir.iterdir():
                if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    name_parts = entry.stem.split("_")
                    prefix = "_".join(name_parts[:-1]) if len(name_parts) > 1 else entry.stem
                    if prefix not in video_groups_dict:
                        video_groups_dict[prefix] = []
                    video_groups_dict[prefix].append(entry)
            for prefix, frames in video_groups_dict.items():
                video_groups.append((prefix, category, sorted(frames)))

    # Select random subset
    random.seed(seed)
    random.shuffle(video_groups)
    selected_groups = video_groups[:max_videos]

    extracted_data = []
    extracted_frames_dir = output_dir / "extracted_frames"
    
    for prefix, category, video_ref in selected_groups:
        if isinstance(video_ref, list):
            # PNG list mode: sample evenly
            tot = len(video_ref)
            indices = [int(i * (tot - 1) / (num_frames - 1)) for i in range(num_frames)] if tot >= num_frames else [i % tot for i in range(num_frames)]
            selected_frames = [video_ref[i] for i in indices]
            extracted_data.append({
                "video_id": prefix,
                "category": category,
                "frame_paths": selected_frames
            })
        else:
            # Video file mode: extract via OpenCV
            extracted_paths = []
            try:
                import cv2
                cap = cv2.VideoCapture(str(video_ref))
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames > 0:
                        vid_dir = extracted_frames_dir / prefix
                        vid_dir.mkdir(parents=True, exist_ok=True)
                        indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)] if num_frames > 1 else [total_frames // 2]
                        for i, idx in enumerate(indices):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                            ret, frame = cap.read()
                            if ret:
                                frame_path = vid_dir / f"frame_{i:02d}.jpg"
                                cv2.imwrite(str(frame_path), frame)
                                extracted_paths.append(frame_path)
                    cap.release()
            except Exception as e:
                print(f"Warning: Failed frame extraction for {prefix}: {e}")
            
            if extracted_paths:
                extracted_data.append({
                    "video_id": prefix,
                    "category": category,
                    "frame_paths": extracted_paths
                })

    return extracted_data
