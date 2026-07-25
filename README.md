# 🛡️ VLM-Based Video Threat Detection & Assessment System

A production-quality, research-grade surveillance intelligence prototype developed for **DRDO Surveillance Operations** using a fine-tuned **Gemma 3 4B Vision-Language Model** (VLM). 

The system processes multi-frame surveillance video sequences, identifies physical security threats (e.g. *Burglary, Assault, Abuse, Fighting, Arrest*), generates structured threat assessment reports, and automates active emergency service dispatch mappings.

---

## 1. System Architecture

The core of the system is the **Two-Stage Hybrid Inference Pipeline**, designed to bypass SFT style-shift ("instruction-following collapse"):
1. **Stage 1 (LoRA Adapter):** A PEFT adapter classifies video keyframe sequences, outputting a concise activity label (e.g., *"A person is performing burglary"*).
2. **Stage 2 (Base VLM):** The base Gemma 3 model receives the keyframes and the Stage 1 label as a factual anchor, producing the final formatted report (Scene Summary, Threat Level, and Action Recommendations).

---

## 2. Directory Structure

The repository is organized following clean, maintainable, and reproducible standards:

```text
├── app.py                         # Root launcher stub (performs diagnostics & starts dashboard)
├── main.py                        # Root API launcher stub
├── backend/                       # VLM API factory loader, model wrappers, and prompts
├── frontend/                      # User Interface dashboard package
│   └── app.py                     # Gradio Web Server
├── configs/                       # Centralized YAML configurations (adapters, training, SFT)
│   ├── adapter_config.yaml        # Active adapter and inference backend configuration
│   └── *_training_config.yaml     # SFT QLoRA hyperparameter configs
├── datasets/                      # Isolated datasets folder
│   └── raw/                       # Raw video datasets (ignored by Git)
│       ├── ucf-crime-mini/        # UCF-Crime surveillance clips
│       ├── XD-Violence/           # XD-Violence surveillance clips
│       └── sample_videos/         # Demo video assets
├── models/                        # Centralized model weight directory
│   └── adapters/                  # PEFT LoRA checkpoints (ignored by Git)
│       ├── activitynet_v1/        # General ActivityNet adapter
│       └── surveillance_colab/    # Fine-tuned surveillance adapter
├── evaluation/                    # Quantitative benchmarking package
│   ├── outputs/                   # Predictions CSVs and selected video indices
│   ├── reports/                   # Quantitative markdown report outputs
│   ├── plots/                     # Threat level distribution charts
│   └── run_eval_xd.py             # Evaluation benchmark runner script
├── scripts/                       # Shell/Python diagnostics and utility scripts
│   ├── verify_environment.py      # System capability verification
│   └── generate_charts.py         # Visual chart generation utility
├── tests/                         # Automated unit test suite (loaders, OpenCV, configs)
├── logs/                          # System execution log dumps (run.log)
├── outputs/                       # UI Runtime exports (Markdown, JSON, logs)
├── Dockerfile                     # Headless dashboard container layout
├── docker-compose.yml             # GPU-allocated docker orchestrator with host-Ollama link
├── setup.sh / setup_project.bat   # Linux / Windows One-Click environment installers
└── run.sh / run_project.bat       # Linux / Windows One-Click application launchers
```

---

## 3. System Requirements

* **Operating System:** Windows 10/11 or Ubuntu Linux (>= 20.04)
* **Git:** Installed and configured.
* **Docker & Docker Compose:** Installed (for containerized deployment).
* **Ollama:** Installed and running locally on port `11434` (for local GPU/CPU model execution).
* **Python:** Version `>= 3.9` (if deploying bare-metal).
* **GPU (Optional):** NVIDIA GPU with CUDA drivers (T4 / RTX 30-series or higher recommended for fast Hugging Face backend).

---

## 4. Quick Start (One-Command Bare-Metal)

We provide fully automated setup and launch scripts that verify dependencies, copy configurations, and pull the required VLM model from Ollama:

### On Linux / macOS
```bash
# 1. Clone the repository
git clone <repo-url>
cd project_root

# 2. Run the environment setup (auto-creates venv, copies env, pulls gemma3:4b from Ollama)
./setup.sh

# 3. Launch the dashboard
./run.sh
```

### On Windows
Double-click `setup_project.bat` to install dependencies and run environment diagnostics, then run `run_project.bat` to launch the Gradio server.

*Once launched, open **[http://localhost:7860](http://localhost:7860)** in your browser.*

---

## 5. One-Command Docker Deployment

To run the application inside a fully isolated Docker container while communicating with the host OS's Ollama service:

1. Create your local environment configuration:
   ```bash
   cp .env.example .env
   ```
2. Start the container:
   ```bash
   docker compose up --build -d
   ```
3. *Access the dashboard at **[http://localhost:7860](http://localhost:7860)**.*

### Container Networking Note
Inside the Docker container, the environment variable `OLLAMA_URL` is set to `http://host.docker.internal:11434` in `docker-compose.yml`. This routes traffic out of the container bridge network back to your host machine's Ollama instance. The `extra_hosts` mapping is defined in the compose file to enable this gateway automatically.

---

## 6. Configuration & Env Variables

All operational configurations are loaded dynamically from the `.env` file at startup. Key variables include:

| Environment Variable | Default Value | Description |
| :--- | :--- | :--- |
| `INFERENCE_BACKEND` | `ollama` | Inference backend: `ollama` (recommended) or `hf` (Hugging Face) |
| `OLLAMA_URL` | `http://localhost:11434` | API connection url for Ollama service |
| `OLLAMA_MODEL` | `gemma3:4b` | Ollama model name to pull and run |
| `ACTIVE_ADAPTER` | `None` | Active PEFT adapter folder under `models/adapters/` |
| `HOST` | `0.0.0.0` | Port interfaces bound by Gradio |
| `PORT` | `7860` | Web server port bound by Gradio |
| `HF_TOKEN` | `None` | Hugging Face write token (required if using the `hf` backend) |

---

## 7. Quantitative Evaluation Benchmark

To run benchmarks evaluating the base model against the fine-tuned adapter on your raw datasets:

```bash
# Run on the raw XD-Violence dataset (default)
.venv/bin/python evaluation/run_eval_xd.py --dataset-dir datasets/raw/XD-Violence

# Run on the raw UCF-Crime dataset
.venv/bin/python evaluation/run_eval_xd.py --dataset-dir datasets/raw/ucf-crime-mini
```

All evaluation metrics, CSV logs, and qualitative comparative markdown reports are saved under the `evaluation/outputs/` and `evaluation/reports/` directories.

---

## 8. Troubleshooting

### 1. Connection Refused to Ollama
* **Error:** `DIAGNOSTIC ERROR: Cannot connect to Ollama service.`
* **Solution (Local):** Verify that the Ollama tray application is running. Check `http://localhost:11434` in your browser.
* **Solution (Docker):** Ensure you are using `http://host.docker.internal:11434` as the `OLLAMA_URL` in your `.env` file so the container can connect to the host.

### 2. Missing LoRA Adapter Warning
* **Warning:** `surveillance LoRA adapter not found. Falling back to Base model only.`
* **Solution:** Verify that your fine-tuned checkpoint folder is located at `models/adapters/surveillance_colab/` and contains `adapter_config.json` and `adapter_model.safetensors`.

### 3. Headless Server OpenCV Crash
* **Error:** `ImportError: libGL.so.1: cannot open shared object file.`
* **Solution:** The project's requirements file utilizes `opencv-python-headless` by default to bypass X11 dependency requirements. If you installed dependencies manually, uninstall `opencv-python` and install `opencv-python-headless`.
