import logging
import os
import sys
import time
import json
from pathlib import Path
from PIL import Image
import torch
import gradio as gr
from datetime import datetime
import cv2
import threading
import psutil
from collections import deque
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import env_loader
env_loader.load_env()

from models.model_loader import load_base_model
from peft import PeftModel

logger = logging.getLogger(__name__)

# Preload CUDA libraries if on Linux
def _preload_cuda_libs():
    import platform
    if platform.system() != "Linux":
        return
    import ctypes
    import site
    paths = list(site.getsitepackages())
    try:
        paths.append(site.getusersitepackages())
    except Exception:
        pass
    for sdir in paths:
        nv_path = Path(sdir) / "nvidia"
        if nv_path.exists():
            for sub in nv_path.iterdir():
                if sub.name == "nvjitlink":
                    lib_dir = sub / "lib"
                    if lib_dir.exists():
                        for so in lib_dir.glob("libnvJitLink.so*"):
                            try:
                                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                            except Exception:
                                pass
            for sub in nv_path.iterdir():
                if sub.name != "nvjitlink":
                    lib_dir = sub / "lib"
                    if lib_dir.exists():
                        for so in lib_dir.glob("*.so*"):
                            try:
                                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                            except Exception:
                                pass

try:
    _preload_cuda_libs()
except Exception as e:
    logger.warning("CUDA preloader was bypassed: %s", e)

# Global models cache
MODEL = None
PROCESSOR = None

def load_vlm(device="cuda"):
    global MODEL, PROCESSOR
    if MODEL is not None:
        return MODEL, PROCESSOR
        
    backend = os.getenv("INFERENCE_BACKEND", "ollama").lower()
    if backend == "ollama":
        logger.info("INFERENCE_BACKEND is set to 'ollama'. Bypassing PyTorch VLM load and using local Ollama REST API.")
        return None, None

    base_model_id = "google/gemma-3-4b-it"
    adapter_path = PROJECT_ROOT / "models" / "adapters" / "surveillance_colab"
    if not adapter_path.exists():
        adapter_path = PROJECT_ROOT / "models" / "adapters" / "surveillance_v1"
        
    use_4bit = (device == "cuda" and torch.cuda.is_available())
    device_map = "auto" if use_4bit else {"": "cpu"}
    
    logger.info("Loading VLM base model %s (use_4bit=%s, device=%s)...", base_model_id, use_4bit, device)
    MODEL, PROCESSOR = load_base_model(model_id=base_model_id, use_4bit=use_4bit, device_map=device_map)
    
    if adapter_path.exists():
        logger.info("Attaching Fine-Tuned Adapter from: %s", adapter_path)
        MODEL = PeftModel.from_pretrained(MODEL, str(adapter_path), is_trainable=False)
    else:
        logger.warning("surveillance LoRA adapter not found. Falling back to Base model only.")
        
    MODEL.eval()
    return MODEL, PROCESSOR

def run_inference(model, processor, images, prompt, device, max_new_tokens=256):
    if model is None:
        import base64
        import requests
        from io import BytesIO
        
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "gemma3:4b")
        
        base64_images = []
        for img in images:
            # Downsample to 224x224 for local CPU execution to reduce prefill load by 4x
            img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
            buffered = BytesIO()
            img_resized.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_images.append(img_str)
            
        payload = {
            "model": ollama_model,
            "prompt": prompt,
            "images": base64_images,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": 0.0
            }
        }
        
        try:
            r = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=360)
            if r.status_code == 200:
                return r.json().get("response", "").strip()
            else:
                return f"Error: Ollama API returned status code {r.status_code}: {r.text}"
        except Exception as e:
            return f"Error connecting to Ollama at {ollama_url}: {str(e)}"

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
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            use_cache=True,
            do_sample=False
        )
        
    input_len = inputs["input_ids"].shape[-1]
    response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    return response

# Live Video Stream Handler for Control Room Mode
class LiveSurveillanceStream:
    def __init__(self):
        self.cap = None
        self.running = False
        self.frame_buffer = deque(maxlen=30)
        self.thread = None
        self.vlm_thread = None
        self.latest_frame = None
        self.fps = 0.0
        self.proc_fps = 0.0
        self.vlm_busy = False
        self.last_vlm_time = 0.0
        self.threat_history = []  # List of (time_str, score)
        self.timeline_data = []    # List of [time, event, threat_level]
        self.current_threat_level = "Low"
        self.current_threat_score = 10
        self.current_confidence = "Medium"
        self.current_activity = "Normal operations"
        self.dispatch_action = "No dispatch required"
        self.evidence_frames = []
        self.status = "SYSTEM READY"
        self.frame_count = 0
        self.start_t = 0.0

    def start(self, source_type, path_or_url=None):
        if self.running:
            return
            
        self.running = True
        self.vlm_busy = False
        self.threat_history = []
        self.timeline_data = []
        self.evidence_frames = []
        self.status = "INITIALIZING STREAM..."
        
        # Determine cv2 capture source
        if source_type == "Laptop Webcam":
            src = 0
        elif source_type == "Local Video Loop" and path_or_url:
            src = path_or_url
        elif source_type == "RTSP Stream" and path_or_url:
            src = path_or_url
        else:
            src = 0
            
        self.cap = cv2.VideoCapture(src)
        
        # Camera grabber thread
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        # Async VLM analyzer worker
        self.vlm_thread = threading.Thread(target=self._vlm_worker, daemon=True)
        self.vlm_thread.start()

    def _capture_loop(self):
        self.start_t = time.time()
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                # Loop video files if they reach the end
                if isinstance(self.cap, cv2.VideoCapture) and self.cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.03)
                continue
                
            self.frame_count += 1
            self.latest_frame = frame
            self.frame_buffer.append(frame)
            
            # Calculate FPS
            elapsed = time.time() - self.start_t
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_t = time.time()
                
            time.sleep(0.01)

    def _vlm_worker(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, processor = load_vlm(device)
        self.status = "SYSTEM OPERATIONAL"
        self.last_vlm_time = time.time()
        
        # Import threat engine helpers
        from backend.threat_engine import (
            calculate_threat_metrics,
            build_event_timeline,
            generate_explainable_report,
            generate_timestamps
        )
        
        while self.running:
            now = time.time()
            # Analyze a sample of the frame buffer every 10 seconds asynchronously
            if not self.vlm_busy and (now - self.last_vlm_time >= 10.0) and len(self.frame_buffer) >= 4:
                self.vlm_busy = True
                self.status = "ANALYZING ACTIVE FEED..."
                
                try:
                    # Capture snapshot of current frame buffer
                    buf_snapshot = list(self.frame_buffer)
                    num_samples = 4
                    indices = [int(i * (len(buf_snapshot) - 1) / (num_samples - 1)) for i in range(num_samples)]
                    sampled_frames = [buf_snapshot[idx] for idx in indices]
                    
                    # Convert to PIL and downscale to 448x448
                    pil_images = []
                    for sf in sampled_frames:
                        rgb = cv2.cvtColor(sf, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb).resize((448, 448), Image.Resampling.LANCZOS)
                        pil_images.append(pil_img)
                        
                    self.evidence_frames = pil_images
                    
                    # Stage 1: SFT Caption
                    start_inf = time.time()
                    caption_prompt = "Describe the exact activity happening in this video sequence as a concise caption (e.g., A person is performing...)."
                    action_caption = run_inference(model, processor, [pil_images[len(pil_images)//2]], caption_prompt, device, max_new_tokens=40)
                    
                    # Stage 2: Reasoning
                    guided_prompt = (
                        "Analyze this surveillance scene video sequence.\n"
                        f"You are given the following pre-extracted activity class: '{action_caption}'\n\n"
                        "Using this action class and the visual evidence from the frames, describe:\n"
                        "* What is happening?\n"
                        "* Which activities are visible?\n"
                        "* Is there any suspicious behaviour?\n"
                        "* Are there any threat indicators?\n"
                        "* Estimate the threat level as Low, Medium, or High.\n"
                        "* Explain your reasoning."
                    )
                    threat_report = run_inference(model, processor, pil_images, guided_prompt, device, max_new_tokens=256)
                    
                    # Parse threat level robustly
                    from backend.threat_engine import parse_threat_level
                    threat_level = parse_threat_level(threat_report)
                            
                    # Calculate metrics
                    dummy_frames = []
                    for i, img in enumerate(pil_images, 1):
                        dummy_frames.append({
                            "frame": i,
                            "description": action_caption if i == len(pil_images) else "Active monitoring frame.",
                            "actions_occurring": [action_caption.split(" ")[-1]] if " " in action_caption else []
                        })
                        
                    metrics = calculate_threat_metrics(threat_level, dummy_frames, threat_report)
                    
                    # Recommendations mappings
                    dispatch = "No emergency dispatch required."
                    act_lower = action_caption.lower()
                    if any(k in act_lower for k in ["fire", "arson", "explosion"]):
                        dispatch = "🚒 Fire Rescue & Emergency Medical Teams Dispatched"
                    elif any(k in act_lower for k in ["shooting", "gun", "weapon"]):
                        dispatch = "🚨 Armed Police & SWAT Tactical Containment Dispatched"
                    elif any(k in act_lower for k in ["assault", "abuse", "fighting", "riot"]):
                        dispatch = "🚓 Police Patrol & Local Security Dispatched"
                    elif any(k in act_lower for k in ["burglary", "stealing", "vandalism", "shoplifting"]):
                        dispatch = "🚓 Local Police Dispatched to secure the property"
                        
                    self.current_threat_level = threat_level
                    self.current_threat_score = metrics["threat_score"]
                    self.current_confidence = metrics["model_confidence"]
                    self.current_activity = action_caption
                    self.dispatch_action = dispatch
                    
                    # Append history and timeline
                    time_str = datetime.now().strftime("%H:%M:%S")
                    self.threat_history.append((time_str, metrics["threat_score"]))
                    if len(self.threat_history) > 20:
                        self.threat_history.pop(0)
                        
                    self.timeline_data.insert(0, [time_str, action_caption, f"{threat_level} ({metrics['threat_score']})"])
                    if len(self.timeline_data) > 10:
                        self.timeline_data.pop()
                        
                    # Save explainable reports to output
                    outputs_dir = PROJECT_ROOT / "outputs"
                    timestamps = generate_timestamps(len(pil_images))
                    timeline_struct = build_event_timeline(dummy_frames, timestamps)
                    generate_explainable_report(
                        video_name="Live_Camera_Feed",
                        action_caption=action_caption,
                        threat_report=threat_report,
                        threat_level=threat_level,
                        metrics=metrics,
                        timeline=timeline_struct,
                        frame_results=dummy_frames,
                        output_dir=outputs_dir,
                        prefix=f"report_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    
                    inf_time = time.time() - start_inf
                    self.proc_fps = 4.0 / inf_time if inf_time > 0 else 0.0
                    self.status = "SYSTEM OPERATIONAL"
                    
                except Exception as e:
                    logger.error("VLM background thread error: %s", e)
                    self.status = f"VLM THREAD ERROR: {str(e)[:30]}"
                    
                self.vlm_busy = False
                self.last_vlm_time = time.time()
                
            time.sleep(0.5)

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.status = "SYSTEM READY"

    def get_latest_frame_rgb(self):
        if self.latest_frame is not None:
            return cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
        return None

stream_handler = LiveSurveillanceStream()

# UI Assessment function for Upload Video / Images
def analyze_surveillance(video_path, custom_frames, num_frames=4, progress=gr.Progress(track_tqdm=True)):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_frames = int(num_frames)
    
    progress(0.0, desc="Initializing model configuration...")
    try:
        model, processor = load_vlm(device)
    except Exception as e:
        logger.error("Failed to load VLM: %s", e, exc_info=True)
        return f"Error loading VLM: {str(e)}", "N/A", "🚨 ERROR", "System Error", None, None, None, None
 
    selected_images = []
    video_name = "Custom Keyframes"
    duration_sec = None
    
    if custom_frames:
        progress(0.2, desc=f"Loading {num_frames} custom keyframe images...")
        for f in custom_frames[:num_frames]:
            img = Image.open(f).convert("RGB")
            img = img.resize((448, 448), Image.Resampling.LANCZOS)
            selected_images.append(img)
    elif video_path:
        progress(0.2, desc=f"Extracting {num_frames} representative motion peaks from video...")
        video_name = Path(video_path).name
        try:
            from backend.frame_extractor import extract_frames
            temp_dir = PROJECT_ROOT / "outputs" / "temp_frames"
            frame_paths = extract_frames(Path(video_path), temp_dir, num_frames)
            
            # Capture duration for timeline timestamps
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps > 0:
                duration_sec = total_frames / fps
            cap.release()
            
            # Load and downsample extracted keyframes
            for fp in frame_paths:
                img = Image.open(fp).convert("RGB")
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                selected_images.append(img)
        except Exception as e:
            logger.error("Error extracting motion frames: %s", e)
            return f"Error extracting motion frames: {str(e)}", "N/A", "🚨 ERROR", "System Error", None, None, None, None
            
    if not selected_images:
        return "Please upload a video or keyframe images.", "N/A", "⚠️ NO INPUT", "No Action Required", None, None, None, None

    # Stage 1: Concise Action Caption
    progress(0.4, desc="Executing Stage 1: Action Classification (LoRA)...")
    caption_prompt = "Describe the exact activity happening in this video sequence as a concise caption (e.g., A person is performing...)."
    try:
        start_inf = time.time()
        action_caption = run_inference(model, processor, [selected_images[len(selected_images)//2]], caption_prompt, device, max_new_tokens=40)
        logger.info("Stage 1 action caption: %s", action_caption)
    except Exception as e:
        logger.error("Failed SFT Stage 1: %s", e)
        action_caption = f"Failed to run SFT Stage 1: {str(e)}"

    # Stage 2: Reasoning & Threat Report
    progress(0.6, desc="Executing Stage 2: Structured Reasoning & Threat Report...")
    guided_prompt = (
        "Analyze this surveillance scene video sequence.\n"
        f"You are given the following pre-extracted activity class: '{action_caption}'\n\n"
        "Using this action class and the visual evidence from the frames, describe:\n"
        "* What is happening?\n"
        "* Which activities are visible?\n"
        "* Is there any suspicious behaviour?\n"
        "* Are there any threat indicators?\n"
        "* Estimate the threat level as Low, Medium, or High.\n"
        "* Explain your reasoning."
    )
    
    try:
        threat_report = run_inference(model, processor, selected_images, guided_prompt, device, max_new_tokens=256)
    except Exception as e:
        logger.error("Failed Stage 2: %s", e)
        threat_report = f"Failed to run Stage 2: {str(e)}"

    # Parse threat level robustly
    from backend.threat_engine import parse_threat_level
    threat_level = parse_threat_level(threat_report)

    progress(0.8, desc="Calculating Threat Metrics and Timelines...")
    
    # Heuristics calculations
    from backend.threat_engine import (
        calculate_threat_metrics,
        generate_timestamps,
        build_event_timeline,
        export_timeline,
        generate_explainable_report
    )
    
    # Build list of frame results structure
    frame_results = []
    for i, img in enumerate(selected_images, 1):
        frame_results.append({
            "frame": i,
            "description": action_caption if i == len(selected_images) else "CCTV surveillance sequence step.",
            "actions_occurring": [action_caption.split(" ")[-1]] if " " in action_caption else [],
            "image": str(PROJECT_ROOT / "outputs" / f"frame_{i}.jpg")
        })
        # Save temp image physically to link as evidence frame
        img.save(PROJECT_ROOT / "outputs" / f"frame_{i}.jpg")

    timestamps = generate_timestamps(len(selected_images), duration_sec)
    timeline = build_event_timeline(frame_results, timestamps)
    metrics = calculate_threat_metrics(threat_level, frame_results, threat_report)
    
    outputs_dir = PROJECT_ROOT / "outputs"
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_prefix = f"report_{timestamp_str}"
    
    # Export timelines and explainable reports
    export_timeline(timeline, outputs_dir, prefix=f"timeline_{timestamp_str}")
    report_paths = generate_explainable_report(
        video_name=video_name,
        action_caption=action_caption,
        threat_report=threat_report,
        threat_level=threat_level,
        metrics=metrics,
        timeline=timeline,
        frame_results=frame_results,
        output_dir=outputs_dir,
        prefix=report_prefix
    )

    # Build timeline dataframe for UI
    timeline_rows = []
    for row in timeline:
        timeline_rows.append([row["timestamp"], row["event"], f"{threat_level} ({row['threat_score']})"])
    timeline_df = pd.DataFrame(timeline_rows, columns=["Timestamp", "Event Description", "Threat Level"])

    progress(1.0, desc="Assessment complete")
    
    # Recommendations dispatch
    dispatch = "No emergency dispatch required."
    act_lower = action_caption.lower()
    if any(k in act_lower for k in ["fire", "arson", "explosion"]):
        dispatch = "🚒 Fire Rescue & Emergency Medical Teams Dispatched"
    elif any(k in act_lower for k in ["shooting", "gun", "weapon"]):
        dispatch = "🚨 Armed Police & SWAT Tactical Containment Dispatched"
    elif any(k in act_lower for k in ["assault", "abuse", "fighting", "riot"]):
        dispatch = "🚓 Police Patrol & Local Security Dispatched"
    elif any(k in act_lower for k in ["burglary", "stealing", "vandalism", "shoplifting"]):
        dispatch = "🚓 Local Police Dispatched to secure the property"
        
    inf_time = time.time() - start_inf
    proc_fps = num_frames / inf_time if inf_time > 0 else 0.0
    
    metrics_summary = f"Threat Score: {metrics['threat_score']}/100 | Evidence Strength: {int(metrics['evidence_strength']*100)}% | Confidence: {metrics['model_confidence']} | Inf Time: {inf_time:.1f}s | FPS: {proc_fps:.2f}"

    return (
        action_caption,
        threat_report,
        threat_level,
        dispatch,
        str(report_paths["json"]),
        str(report_paths["md"]),
        timeline_df,
        metrics_summary,
        selected_images
    )

# Streaming Generator loop function
def stream_surveillance(source_type, rtsp_url):
    global stream_handler
    
    # Stop and clear if currently active
    if stream_handler.running:
        stream_handler.stop()
        time.sleep(0.5)
        
    stream_handler.start(source_type, rtsp_url)
    
    while stream_handler.running:
        frame = stream_handler.get_latest_frame_rgb()
        if frame is None:
            time.sleep(0.05)
            continue
            
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        gpu = "N/A"
        if torch.cuda.is_available():
            gpu = f"{torch.cuda.memory_allocated() / (1024*1024):.1f} MB"
            
        sys_fps = f"{stream_handler.fps:.1f} FPS"
        proc_fps = f"{stream_handler.proc_fps:.2f} FPS"
        status_text = stream_handler.status
        
        # Threat assessment outputs
        threat_level = stream_handler.current_threat_level
        threat_score = stream_handler.current_threat_score
        confidence = stream_handler.current_confidence
        activity = stream_handler.current_activity
        dispatch = stream_handler.dispatch_action
        
        # Build timeline df
        timeline_df = pd.DataFrame(stream_handler.timeline_data, columns=["Timestamp", "Activity Log", "Threat Rating"])
        
        # Build threat history df
        history_df = pd.DataFrame(stream_handler.threat_history, columns=["Time", "Threat Score"])
        if history_df.empty:
            history_df = pd.DataFrame([("00:00", 10)], columns=["Time", "Threat Score"])
            
        # Format HTML status panel
        status_html = f"<span class='status-ready'>● {status_text}</span>"
        if "ANALYZING" in status_text:
            status_html = f"<span class='status-analyzing'>● {status_text}</span>"
        elif "ERROR" in status_text:
            status_html = f"<span class='status-dispatch'>● {status_text}</span>"
            
        yield (
            frame,
            sys_fps,
            proc_fps,
            threat_level,
            f"{threat_score}%",
            confidence,
            activity,
            dispatch,
            timeline_df,
            history_df,
            status_html,
            f"{cpu}%",
            f"{ram}%",
            gpu,
            stream_handler.evidence_frames
        )
        time.sleep(0.04) # refresh at ~25 FPS

def stop_surveillance():
    global stream_handler
    stream_handler.stop()
    return (
        None, "0.0 FPS", "0.0 FPS", "Low", "10%", "Medium", "System Idle", "No dispatch required",
        pd.DataFrame(columns=["Timestamp", "Activity Log", "Threat Rating"]),
        pd.DataFrame([("00:00", 10)], columns=["Time", "Threat Score"]),
        "<span class='status-ready'>● SYSTEM READY</span>",
        "0%", "0%", "N/A", []
    )

# Control Room Dark Theme Styling
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
).set(
    body_background_fill="#0b0f19",
    block_background_fill="#111827",
    block_border_color="#1f2937",
    button_primary_background_fill="#1d4ed8",
    button_primary_background_fill_hover="#2563eb",
    input_background_fill="#1f2937"
)

css_style = """
.status-ready {
    color: #10b981 !important;
    font-weight: bold;
    font-family: monospace;
}
.status-analyzing {
    color: #f59e0b !important;
    font-weight: bold;
    font-family: monospace;
    animation: blink 1.5s infinite;
}
.status-dispatch {
    color: #ef4444 !important;
    font-weight: bold;
    font-family: monospace;
}
@keyframes blink {
    0% { opacity: 0.3; }
    50% { opacity: 1; }
    100% { opacity: 0.3; }
}
.header-box {
    background-color: #1f2937;
    padding: 15px;
    border-radius: 8px;
    border-left: 5px solid #2563eb;
    margin-bottom: 15px;
}
"""

with gr.Blocks(theme=theme, css=css_style, title="DRDO Surveillance Threat Assessment Dashboard") as demo:
    
    with gr.Row(elem_classes=["header-box"]):
        with gr.Column(scale=4):
            gr.HTML("<h2>🛡️ DRDO COGNITIVE SURVEILLANCE CONTROL CENTER</h2>")
            gr.HTML("<p>Tactical Threat Detection & Explainable Decision-Support System powered by Gemma 3 VLM</p>")
        with gr.Column(scale=1):
            status_indicator = gr.HTML("<span class='status-ready'>● SYSTEM READY</span>", label="System Status")
            
    with gr.Tabs():
        
        # =====================================================================
        # TAB 1: LIVE SURVEILLANCE MODE
        # =====================================================================
        with gr.TabItem("🔴 Live Surveillance Mode"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 🎥 Camera / Stream Feed")
                    live_image = gr.Image(label="Live Camera Feed", type="numpy", interactive=False)
                    
                    with gr.Row():
                        source_type = gr.Dropdown(
                            choices=["Laptop Webcam", "Local Video Loop", "RTSP Stream"], 
                            value="Laptop Webcam", 
                            label="Camera Input Source"
                        )
                        stream_path = gr.Textbox(
                            label="RTSP URL / Local Video Path", 
                            placeholder="e.g. data/test.mp4 or rtsp://username:password@ip"
                        )
                        
                    with gr.Row():
                        start_btn = gr.Button("▶️ Start Surveillance", variant="primary")
                        stop_btn = gr.Button("🛑 Stop Surveillance", variant="stop")
                        
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Active Threat Telemetry")
                    
                    with gr.Row():
                        live_threat_level = gr.Textbox(label="Threat Status Level", value="Low")
                        live_threat_score = gr.Textbox(label="Unified Threat Score", value="10%")
                        live_confidence = gr.Textbox(label="VLM Confidence", value="Medium")
                        
                    live_activity = gr.Textbox(label="Detected Action Caption (Stage 1)", value="System Idle")
                    live_dispatch = gr.Textbox(label="Emergency Dispatch Target Plan", value="No dispatch required")
                    
                    with gr.Row():
                        live_sys_fps = gr.Textbox(label="Camera Stream FPS", value="0.0 FPS")
                        live_proc_fps = gr.Textbox(label="VLM Processing Rate", value="0.0 FPS")
                        
                    with gr.Row():
                        cpu_metric = gr.Textbox(label="CPU Usage", value="0%")
                        ram_metric = gr.Textbox(label="RAM Usage", value="0%")
                        gpu_metric = gr.Textbox(label="GPU VRAM Allocated", value="N/A")
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📅 Chronological Activity Log")
                    live_timeline_tbl = gr.DataFrame(
                        headers=["Timestamp", "Activity Log", "Threat Rating"],
                        datatype=["str", "str", "str"],
                        value=pd.DataFrame(columns=["Timestamp", "Activity Log", "Threat Rating"])
                    )
                with gr.Column(scale=2):
                    gr.Markdown("### 📈 Continuous Threat Assessment History")
                    threat_history_chart = gr.LinePlot(
                        value=pd.DataFrame([("00:00", 10)], columns=["Time", "Threat Score"]),
                        x="Time",
                        y="Threat Score",
                        label="Threat Severity Trend"
                    )
                    
            with gr.Row():
                gr.Markdown("### 🖼️ Evidence Capture Gallery")
                live_evidence_gallery = gr.Gallery(label="VLM Sampled Keyframes", show_label=True)

        # =====================================================================
        # TAB 2: OFFLINE EVALUATION & FILE UPLOAD
        # =====================================================================
        with gr.TabItem("📁 Offline Video & File Assessment"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎥 Select Source")
                    video_input = gr.Video(label="Upload Surveillance Video Clip")
                    image_input = gr.Files(file_count="multiple", file_types=["image"], label="Or Upload Keyframe Images")
                    num_frames_input = gr.Slider(minimum=2, maximum=8, step=2, value=4, label="Number of Keyframes to Sample")
                    analyze_btn = gr.Button("🔍 Run Threat Assessment", variant="primary")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 🚨 Assessment Results Dashboard")
                    
                    with gr.Row():
                        threat_level_output = gr.Textbox(label="Assessment Threat Level")
                        dispatch_output = gr.Textbox(label="Active Emergency Response Plan")
                        
                    caption_output = gr.Textbox(label="Stage 1: Concise Action Caption", lines=2)
                    metrics_box = gr.Textbox(label="System Resource Metrics & Diagnostics")
                    report_output = gr.Markdown(label="Stage 2: Structured Reasoning & Decision-Support Report")
                    
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📅 Chronological Event Timeline")
                    timeline_tbl = gr.DataFrame(
                        headers=["Timestamp", "Event Description", "Threat Level"],
                        datatype=["str", "str", "str"]
                    )
                with gr.Column(scale=2):
                    gr.Markdown("### 🖼️ Evidence Frames")
                    evidence_gallery = gr.Gallery(label="Processed Keyframes", columns=4)
                    
            with gr.Row():
                json_file_output = gr.File(label="Download Explainable Report (JSON)")
                md_file_output = gr.File(label="Download Explainable Report (Markdown)")

    # -------------------------------------------------------------------------
    # WIRING EVENTS
    # -------------------------------------------------------------------------
    
    # 1. Start live streaming surveillance mode
    start_btn.click(
        fn=stream_surveillance,
        inputs=[source_type, stream_path],
        outputs=[
            live_image, live_sys_fps, live_proc_fps, live_threat_level, 
            live_threat_score, live_confidence, live_activity, live_dispatch,
            live_timeline_tbl, threat_history_chart, status_indicator,
            cpu_metric, ram_metric, gpu_metric, live_evidence_gallery
        ]
    )
    
    # 2. Stop live surveillance mode
    stop_btn.click(
        fn=stop_surveillance,
        outputs=[
            live_image, live_sys_fps, live_proc_fps, live_threat_level, 
            live_threat_score, live_confidence, live_activity, live_dispatch,
            live_timeline_tbl, threat_history_chart, status_indicator,
            cpu_metric, ram_metric, gpu_metric, live_evidence_gallery
        ]
    )
    
    # 3. Analyze uploaded file
    analyze_btn.click(
        fn=analyze_surveillance,
        inputs=[video_input, image_input, num_frames_input],
        outputs=[
            caption_output, report_output, threat_level_output, dispatch_output,
            json_file_output, md_file_output, timeline_tbl, metrics_box, evidence_gallery
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
