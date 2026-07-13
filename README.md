# 🛡️ VLM-Based Video Threat Detection & Assessment System

A production-ready research prototype developed for **DRDO Surveillance Operations** using a fine-tuned **Gemma 3 4B Vision-Language Model** (VLM). The system processes raw surveillance feeds, identifies physical anomalies, generates structured threat reports, and automates emergency service dispatch protocols.

---

## 1. Project Directory Structure

We organize the project into distinct packages for backend services, frontend dashboard, model management, evaluation, and documentation:

```text
├── backend/                  # Core VLM inference APIs, analyzers, and alert pipelines
│   ├── analyzer_factory.py   # Factory to instantiate Ollama or HF backends
│   ├── frame_extractor.py    # OpenCV keyframe sampler and video processor
│   ├── gemma_analyzer.py     # Ollama API adapter implementation
│   ├── hf_gemma_analyzer.py  # HuggingFace & LoRA adapter implementation
│   ├── prompts.py            # Surveillance system prompt templates
│   ├── summarizer.py         # Frame-by-frame and chronological timeline synthesizer
│   ├── video_qa.py           # Interactive CLI Video Q&A agent
│   └── realtime_alert_pipeline.py  # CCTV streaming simulation API
├── frontend/                 # Gradio dashboard user interface
│   └── app.py                # Dashboard application
├── training/                 # LoRA SFT training and dataset classes
│   ├── train.py              # QLoRA fine-tuning runner
│   ├── dataset.py            # Lazy image-loading PyTorch dataset & collator
│   └── config.py             # Dataclass configuration loader
├── evaluation/               # Metrics and comparison suites
│   └── run_eval_xd.py        # Base vs Fine-Tuned benchmarking tool
├── configs/                  # YAML training, adapter, and evaluation configs
├── docs/                     # Technical specifications and research guides
├── tests/                    # System unit tests (configuration, frames, models)
├── outputs/                  # Exported CSV results, Markdown reports, and charts
├── logs/                     # Session run logs
├── Dockerfile                # Gradio dashboard Docker image configuration
├── docker-compose.yml        # Multi-container orchestrator with GPU pass-through
├── app.py                    # Root launcher redirecting to frontend/app.py
└── main.py                   # Root launcher redirecting to backend/main.py
```

---

## 2. Technical Documentation Index

Detailed specifications, research experiments, and guides are split into dedicated sub-documents:

*   **[Architecture Specification](docs/Architecture.md):** Detailed breakdown of the Two-Stage Hybrid Inference Pipeline and data flow.
*   **[Fine-Tuning & SFT Guide](docs/Training.md):** Information on SFT dataset compilation, QLoRA adapter hyperparameters, and GPU FP16 compute optimization.
*   **[Evaluation Framework](docs/Evaluation.md):** Detailed definitions of precision/recall metrics, confusion matrices, and visualization plots.
*   **[Deployment & Operations Guide](docs/Deployment.md):** Step-by-step instructions for local, Docker containerized (with GPU pass-through), and Kaggle setups.
*   **[Threat Assessment Taxonomy](docs/ThreatAssessment.md):** Threat classification matrix, dispatch rules, and structured report format specifications.

---

## 3. Quick Start Guide

### Setup Virtual Environment
Run the setup utility to install packages and perform hardware diagnostics:

```bash
# On Windows Command Prompt:
setup_project.bat

# On PowerShell:
.\setup_env.ps1
```

### Launch Gradio Dashboard
```bash
python app.py
```
*Accessible locally at `http://127.0.0.1:7860`.*

### Run Unit Tests
```bash
python -m unittest discover -s tests
```

---

## 4. Research Highlights

*   **Two-Stage Hybrid Inference:** Bypasses "instruction collapse" by dividing detection into two distinct steps: fine-tuned action classification (Stage 1) and base-model guided threat reasoning (Stage 2).
*   **FP16 Tensor Core Optimization:** Reduces epoch times from 12 hours (software emulation) to **8 minutes** (hardware execution) by targeting T4 Tensor Cores (`fp16: true` and `bnb_4bit_compute_dtype: float16`).
*   **Zero-Dependency Fallbacks:** Includes auto-generation of Pillow dummy frames and clean log file pipelines, preventing run crashes due to directory drift or missing dataset files.
