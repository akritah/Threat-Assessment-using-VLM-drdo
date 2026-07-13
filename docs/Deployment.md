# Deployment and Production Guide

This document describes how to deploy the Gradio Threat Monitoring Dashboard locally, inside Docker containers, or on cloud notebook environments (Colab/Kaggle).

---

## 1. Local Deployment

Ensure your environment is set up by executing `setup_project.bat` (Windows) or `setup_env.ps1` (PowerShell). 

To launch the web interface locally, run:

```bash
python app.py
```

By default, the Gradio dashboard will launch on:
*   **URL:** `http://127.0.0.1:7860`
*   **Public Share Link:** Gradio will print a temporary `.gradio.live` link in stdout, allowing you to access the dashboard remotely (requires active internet).

---

## 2. Docker Containerized Deployment

To ensure reproducibility across systems and isolate dependencies, we provide docker configs.

### Prerequisites
*   Install **Docker** and **Docker Compose**.
*   Install **NVIDIA Container Toolkit** to allow Docker to access GPU hardware.

### One-Command Startup
To build and run the containerized dashboard:

```bash
# Start the container
docker compose up --build -d
```

*   The dashboard will be available at `http://localhost:7860`.
*   The trained adapters and model weights will be read from the local repository directory since we mount the workspace folder inside the container.

To stop the containers:
```bash
docker compose down
```

---

## 3. Cloud Deployment (Kaggle / Colab)

If you are running SFT training or evaluation in Kaggle/Colab notebooks:
1.  Enable **GPU T4** (or T4 x2) in the notebook settings.
2.  Enable **Internet** access (required to download model weights from HuggingFace).
3.  Execute the consolidated shell cell to clone, train, evaluate, and zip deliverables.
4.  Remove the `.git` directory (`rm -rf project/.git`) before exiting the cell so Kaggle doesn't exclude files from the output tab.
