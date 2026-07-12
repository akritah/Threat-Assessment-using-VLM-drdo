import os
import sys
import time
from pathlib import Path
import gradio as gr
from PIL import Image
import torch

# Ensure local imports work
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.model_loader import load_base_model
from peft import PeftModel

# Preload CUDA libraries
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
except Exception:
    pass

# Global models cache
MODEL = None
PROCESSOR = None

def load_vlm(device="cuda"):
    global MODEL, PROCESSOR
    if MODEL is not None:
        return MODEL, PROCESSOR
        
    base_model_id = "google/gemma-3-4b-it"
    adapter_path = PROJECT_ROOT / "adapters" / "surveillance_v1"
    
    use_4bit = (device == "cuda" and torch.cuda.is_available())
    device_map = "auto" if use_4bit else {"": "cpu"}
    
    print(f"Loading VLM (use_4bit={use_4bit}, device={device})...")
    MODEL, PROCESSOR = load_base_model(model_id=base_model_id, use_4bit=use_4bit, device_map=device_map)
    
    if adapter_path.exists():
        print(f"Attaching Fine-Tuned Adapter from: {adapter_path}")
        MODEL = PeftModel.from_pretrained(MODEL, str(adapter_path), is_trainable=False)
    else:
        print("⚠️ Warning: surveillance_v1 adapter not found. Using base model only.")
        
    MODEL.eval()
    return MODEL, PROCESSOR

def run_inference(model, processor, images, prompt, device):
    content_list = []
    # Vision-Language prompt: list of images + text prompt
    for img in images[:8]:
        content_list.append({"type": "image", "image": img})
    content_list.append({"type": "text", "text": prompt})
    
    messages = [{"role": "user", "content": content_list}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Cast compute precision
    if device == "cuda":
        inputs = {k: (v.to(torch.float16) if v.dtype == torch.float32 else v) for k, v in inputs.items()}
    else:
        inputs = {k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v) for k, v in inputs.items()}
        
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    input_len = inputs["input_ids"].shape[-1]
    response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    return response

# Main analysis function for Gradio UI
def analyze_surveillance(video_path, custom_frames):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, processor = load_vlm(device)
    except Exception as e:
        return f"Error loading VLM: {str(e)}", "N/A", "🚨 ERROR", "System Error"

    # Extract frames from video or use uploaded frames
    selected_images = []
    if custom_frames:
        for f in custom_frames[:8]:
            selected_images.append(Image.open(f).convert("RGB"))
    elif video_path:
        # If user uploaded a video, we extract 8 keyframes using OpenCV
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                indices = [int(i * (total_frames - 1) / 7) for i in range(8)]
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        selected_images.append(Image.fromarray(rgb_frame))
            cap.release()
        except Exception as e:
            return f"Error extracting video frames: {str(e)}", "N/A", "🚨 ERROR", "System Error"
            
    if not selected_images:
        return "Please upload a video or keyframe images.", "N/A", "⚠️ NO INPUT", "No Action Required"

    # Stage 1: Concise Action Caption (Fine-Tuned Adapter output)
    caption_prompt = "Describe the exact activity happening in this video sequence as a concise caption (e.g., A person is performing...)."
    try:
        action_caption = run_inference(model, processor, selected_images, caption_prompt, device)
    except Exception as e:
        action_caption = f"Failed to run SFT Stage 1: {str(e)}"

    # Stage 2: Structured Threat Report (Guided Base Model Reasoning)
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
        threat_report = run_inference(model, processor, selected_images, guided_prompt, device)
    except Exception as e:
        threat_report = f"Failed to run Stage 2: {str(e)}"

    # Parse threat level
    threat_level = "Low"
    for line in threat_report.split("\n"):
        if "threat level" in line.lower():
            raw_level = line.split(":")[-1].strip().lower()
            if "high" in raw_level:
                threat_level = "High"
            elif "medium" in raw_level:
                threat_level = "Medium"
            break

    # Parse emergency service dispatch based on activity keywords
    emergency_services = "No dispatch required. Normal operations."
    action_lower = action_caption.lower()
    if any(k in action_lower for k in ["fire", "arson", "explosion"]):
        emergency_services = "🚒 Fire Rescue & Emergency Medical Teams Dispatched"
    elif any(k in action_lower for k in ["shooting", "gun", "weapon"]):
        emergency_services = "🚨 Armed Police & SWAT Tactical Containment Dispatched"
    elif any(k in action_lower for k in ["assault", "abuse", "fighting", "riot"]):
        emergency_services = "🚓 Police Patrol & Local Security Dispatched"
    elif any(k in action_lower for k in ["burglary", "stealing", "vandalism", "shoplifting"]):
        emergency_services = "🚓 Local Police Dispatched to secure the property"

    return action_caption, threat_report, threat_level, emergency_services

# Build Gradio Web Interface
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
)

with gr.Blocks(theme=theme, title="Surveillance Threat Detection Dashboard") as demo:
    gr.Markdown("# 🛡️ VLM-Based Video Threat Detection & Assessment System")
    gr.Markdown("### Developed for DRDO Surveillance Operations using Fine-Tuned Gemma 3 VLM")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎥 Input Video / Keyframes")
            video_input = gr.Video(label="Upload Surveillance Video Clip")
            image_input = gr.Files(file_count="multiple", file_types=["image"], label="Or Upload Keyframe Images (Max 8)")
            analyze_btn = gr.Button("🔍 Run Threat Assessment", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 🚨 Threat Monitoring Dashboard")
            
            with gr.Row():
                threat_level_output = gr.Textbox(label="Assessment Threat Level", placeholder="Low / Medium / High")
                dispatch_output = gr.Textbox(label="Active Emergency Response", placeholder="Waiting for analysis...")
                
            caption_output = gr.Textbox(label="Stage 1: Concise Action Caption", lines=2)
            report_output = gr.Markdown(label="Stage 2: Detailed Structured Reasoning & Threat Report")

    # Wire button action
    analyze_btn.click(
        fn=analyze_surveillance,
        inputs=[video_input, image_input],
        outputs=[caption_output, report_output, threat_level_output, dispatch_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
