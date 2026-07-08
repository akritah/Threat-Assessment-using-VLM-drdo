import argparse
import json
import logging
from pathlib import Path
import sys
import time
import cv2
from PIL import Image
import torch
import uvicorn
from fastapi import FastAPI, BackgroundTasks

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.model_loader import load_base_model
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="DRDO Real-time VLM CCTV Threat Monitor")

# Global models dictionary to prevent double loading
MODELS = {"ft": None, "base": None, "processor": None}

def load_pipelines(adapter_path, device):
    base_model_id = "google/gemma-3-4b-it"
    use_4bit = (device == "cuda")
    device_map = "auto" if device == "cuda" else {"": "cpu"}
    
    # Load processor and base model
    logger.info("Initializing base model and processor...")
    base_model, processor = load_base_model(model_id=base_model_id, use_4bit=use_4bit, device_map=device_map)
    MODELS["processor"] = processor
    MODELS["base"] = base_model
    
    # Load Fine-Tuned adapter
    actual_adapter_path = adapter_path
    if adapter_path and not Path(adapter_path).exists():
        for p in PROJECT_ROOT.glob("**/adapter_model.safetensors"):
            actual_adapter_path = str(p.parent)
            break
            
    if actual_adapter_path and Path(actual_adapter_path).exists():
        logger.info(f"Attaching LoRA adapter from: {actual_adapter_path}")
        # Note: In standard PEFT, we can create a PeftModel around the base model.
        # Since we run sequentially, we wrap the base model.
        ft_model = PeftModel.from_pretrained(base_model, actual_adapter_path, is_trainable=False)
        MODELS["ft"] = ft_model
    else:
        logger.warning("LoRA adapter not found. Falling back to base model for Stage 1.")
        MODELS["ft"] = base_model

def run_vlm_inference(model, processor, images, prompt, device):
    content_list = []
    for img in images:
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
        output_ids = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    input_len = inputs["input_ids"].shape[-1]
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

def process_stream_simulation(video_source: str, device: str):
    logger.info(f"Starting stream processing on source: {video_source}")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {video_source}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = int(fps * 5) # Process a batch every 5 seconds
    
    frame_buffer = []
    frame_count = 0
    
    # Setup alerts output dir
    alerts_dir = PROJECT_ROOT / "outputs" / "alerts"
    alerts_dir.mkdir(parents=True, exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video stream reached.")
            break
            
        # Maintain a rolling window of 8 frames
        # We sample at 1.5-second intervals to get a good temporal spread of the last 5 seconds
        if frame_count % int(fps * 0.6) == 0:
            # Convert CV2 BGR to PIL RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            frame_buffer.append(pil_img)
            if len(frame_buffer) > 8:
                frame_buffer.pop(0)
                
        # Trigger inference every 5 seconds
        if frame_count > 0 and frame_count % frame_interval == 0 and len(frame_buffer) == 8:
            logger.info("Analyzing rolling temporal window...")
            
            # --- STAGE 1: Action Classification (Fine-Tuned) ---
            caption_prompt = "Describe the exact activity happening in this video sequence as a concise caption (e.g., A person is performing...)."
            action_caption = run_vlm_inference(MODELS["ft"], MODELS["processor"], frame_buffer, caption_prompt, device)
            logger.info(f"[STAGE 1 - Action Classification]: {action_caption}")
            
            # --- STAGE 2: Guided Threat Assessment (Base) ---
            guided_prompt = (
                "Analyze this surveillance scene video sequence.\n"
                f"You are given the following pre-extracted activity class: '{action_caption}'\n\n"
                "Using this action class and the visual evidence from the frames, describe:\n"
                "* What is happening?\n"
                "* Estimate the threat level as Low, Medium, or High.\n"
                "* Explain your reasoning."
            )
            report = run_vlm_inference(MODELS["base"], MODELS["processor"], frame_buffer, guided_prompt, device)
            logger.info(f"[STAGE 2 - Guided Threat Assessment]:\n{report}")
            
            # Parse threat level
            threat_level = "Low"
            for line in report.split("\n"):
                if "threat level" in line.lower():
                    if "high" in line.lower():
                        threat_level = "High"
                    elif "medium" in line.lower():
                        threat_level = "Medium"
                    break
                    
            if threat_level == "High":
                alert_time = time.strftime("%Y%m%d-%H%M%S")
                alert_payload = {
                    "alert_id": f"ALERT-{alert_time}",
                    "timestamp": time.time(),
                    "threat_level": "High",
                    "action_caption": action_caption,
                    "threat_assessment": report,
                    "recommended_action": "DISPATCH IMMEDIATE EMERGENCY SERVICES / POLICE WARNING"
                }
                
                # Write alert dispatch JSON
                alert_file = alerts_dir / f"alert_{alert_time}.json"
                with open(alert_file, "w") as f:
                    json.dump(alert_payload, f, indent=4)
                    
                logger.critical(
                    f"\n=========================================\n"
                    f"[ALERT] SECURITY WARNING: HIGH THREAT DETECTED!\n"
                    f"Action: {action_caption}\n"
                    f"Dispatching Emergency Services...\n"
                    f"=========================================\n"
                )
                
        frame_count += 1
        # Sleep slightly to simulate real-time processing
        time.sleep(1 / fps)
        
    cap.release()
    logger.info("Real-time simulation complete.")

@app.post("/start_stream")
def start_stream(video_path: str, background_tasks: BackgroundTasks, device: str = "cpu"):
    """Trigger background streaming threat analysis."""
    background_tasks.add_task(process_stream_simulation, video_path, device)
    return {"status": "processing", "source": video_path, "device": device}

@app.get("/alerts")
def get_alerts():
    """Retrieve all logged high-threat security alerts."""
    alerts_dir = PROJECT_ROOT / "outputs" / "alerts"
    if not alerts_dir.exists():
        return {"alerts": []}
    files_list = list(alerts_dir.glob("*.json"))
    payload = []
    for f in files_list:
        with open(f, "r") as handle:
            payload.append(json.load(handle))
    return {"alerts": sorted(payload, key=lambda x: x["timestamp"], reverse=True)}

def main():
    parser = argparse.ArgumentParser(description="DRDO Real-time VLM CCTV Threat Monitor")
    parser.add_argument("--video", default="sample.mp4", help="Path to video source")
    parser.add_argument("--adapter-path", default="adapters/activitynet_v1", help="Path to LoRA weights")
    parser.add_argument("--device", default="cpu", help="Device (cuda or cpu)")
    parser.add_argument("--port", type=int, default=8000, help="Web server port")
    args = parser.parse_args()
    
    # Auto-detect CUDA
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    
    # Load pipelines
    load_pipelines(args.adapter_path, device)
    
    # In CLI mode, run stream simulation directly
    logger.info("Running CLI CCTV Simulation...")
    video_source = args.video
    if not Path(video_source).exists():
        # Fallback to local root search
        video_source = str(PROJECT_ROOT / args.video)
        
    process_stream_simulation(video_source, device)
    
    # Start web server
    logger.info(f"Starting FastAPI monitoring dashboard on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
