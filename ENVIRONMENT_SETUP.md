# Environment Setup and Reproducibility Guide

This guide details the system requirements, environment configuration, dependency installation, and verification steps necessary to execute the Gemma 3 Vision threat detection and activity understanding pipeline.

---

## 1. System Requirements

### Operating Systems
* **Linux**: Ubuntu 22.04 LTS or Debian 12 (Recommended for training).
* **Windows**: Windows 10/11 with PowerShell 7+ or command prompt (Suitable for Ollama-based local inference).

### Hardware Requirements
* **Memory**: Minimum 16 GB System RAM.
* **Storage**: 15 GB free space (for base model checkpoint downloads).
* **GPU (Inference)**: 
  * **Ollama (default)**: CPU execution is supported. For GPU acceleration, any NVIDIA/AMD/Apple Silicon GPU supported by Ollama.
  * **Hugging Face (HF)**: NVIDIA GPU with minimum 8 GB VRAM (for 4-bit quantized base model).
* **GPU (Training)**:
  * **Required**: NVIDIA GPU with minimum 12 GB VRAM (e.g., RTX 3060/4060, Tesla T4 in Google Colab).
  * **Recommended**: NVIDIA GPU with 16 GB to 24 GB VRAM (e.g., RTX 3090, RTX 4090, Tesla A10G, A100).
* **CUDA Support**: CUDA Toolkit 11.8 or 12.1+ installed with compatible NVIDIA drivers.

### Python Version
* **Required**: Python 3.10, 3.11, or 3.12.
* **Tested on**: Python 3.12.0

---

## 2. Environment Setup

Select the instructions corresponding to your operating system.

### Option A: Windows (PowerShell / Command Prompt)

#### 1. Verify Python & Pip Installation
Ensure Python 3.10-3.12 is installed and added to your system PATH:
```powershell
python --version
pip --version
```

#### 2. Create and Activate Virtual Environment
Navigate to the repository root directory and create a virtual environment:
```powershell
# In PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# In Command Prompt:
python -m venv .venv
.\.venv\Scripts\activate.bat
```

#### 3. Configure Environment Variables
Copy the template configuration file to configure your local runtime:
```powershell
Copy-Item .env.example .env
```

---

### Option B: Linux (Ubuntu / Debian)

#### 1. Install System Dependencies
Install Python virtual environment utilities and video frame extraction tools:
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv ffmpeg libsm6 libxext6
```

#### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Configure Environment Variables
```bash
cp .env.example .env
```

---

## 3. Dependency Installation

The dependencies are divided into **Inference** (core application) and **Training** (fine-tuning and evaluation modules).

### 1. Base Inference Dependencies
To run video frame extraction and the default Ollama-based analyzer:
```bash
pip install -r requirements.txt
```

### 2. Training and Evaluation Dependencies
If you plan to run fine-tuning scripts, evaluate LoRA adapters, or use the local Hugging Face model loading backend:
```bash
# Upgrade pip first to avoid package resolution conflicts
pip install --upgrade pip setuptools

# Install PyTorch and core training dependencies
pip install -r requirements-train.txt
```

---

## 4. Hugging Face Authentication

Gemma 3 is a gated model. To download the weights:
1. Accept the model license terms on the Hugging Face model card for [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it).
2. Generate a **Read** token under your Hugging Face account settings.
3. Authenticate your local environment by running:
   ```bash
   huggingface-cli login
   ```
   *Alternatively*, you can add your token directly to your `.env` file under `HF_TOKEN=your_token_here`.

---

## 5. Directory Structure Expectations

When executing inference or training, the pipeline expects the following structure relative to the project root:

```
├── config/
│   ├── activitynet_training_config.yaml  # Fine-tuning parameters
│   └── adapter_config.yaml              # Inference backend settings
├── models/
│   └── gemma_base/                       # (Gitignored) Base model checkpoints
├── adapters/
│   └── activitynet_v1/                   # (Gitignored) Fine-tuned LoRA weights
├── training/
│   └── data/
│       ├── train.jsonl                   # Prepared training split
│       ├── eval.jsonl                    # Prepared evaluation split
│       └── frames/                       # Extracted segment midpoint frames
├── outputs/                              # Directory for reports and frame output logs
```

---

## 6. Verification and Diagnostics

Run the following commands within your activated virtual environment to verify the configuration.

### 1. Check GPU Visibility
Confirm PyTorch detects your GPU and verifies the CUDA version:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### 2. Validate Preprocessing Frame Extraction
Extract frames from a sample video to verify OpenCV and folder structures:
```bash
# Windows
python -m training.preprocess_activitynet --annotations training/data/mock_activitynet.json --videos-dir . --output-dir training/data --split-ratio 0.5

# Linux
python3 -m training.preprocess_activitynet --annotations training/data/mock_activitynet.json --videos-dir . --output-dir training/data --split-ratio 0.5
```
Verify that `training/data/train.jsonl` has been generated and contains relative paths with forward slashes (`/`).

### 3. Validate Base Model Load
Run a sanity test to ensure PyTorch can load the Gemma 3 tokenizer and model structure into memory:
```bash
python -c "from transformers import AutoProcessor; processor = AutoProcessor.from_pretrained('google/gemma-3-4b-it', trust_remote_code=True); print('Gemma 3 Processor Loaded Successfully')"
```

---

## 7. Troubleshooting

### 1. bitsandbytes loading errors on Linux/Colab
* **Symptom**: `bitsandbytes library load error: libnvJitLink.so.13: cannot open shared object file`
* **Cause**: Recent `bitsandbytes` versions lookup system CUDA 13 components which are not default in the linker path.
* **Solution**: Our script contains an automatic runtime preloader in `train.py`. If you run other standalone tools, run:
  ```bash
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
  ```

### 2. FileNotFoundError during training dataset load
* **Symptom**: `FileNotFoundError: No such file or directory: 'training/data/frames/sample.jpg'`
* **Cause**: Path mismatch between where annotations are saved and execution root.
* **Solution**: Always run commands from the repository root directory. The data preprocessor saves paths relative to the project root, and the dataloader resolves them relative to the execution root directory.

### 3. Out-Of-Memory (OOM) on GPU
* **Symptom**: `torch.OutOfMemoryError: CUDA out of memory.`
* **Solution**:
  1. Reduce `per_device_train_batch_size` to `1` in `config/activitynet_training_config.yaml`.
  2. Increase `gradient_accumulation_steps` to `8` or `16` to maintain effective batch size.
  3. Ensure `gradient_checkpointing: true` is enabled in your YAML config.
  4. Ensure `load_in_4bit: true` is active.
