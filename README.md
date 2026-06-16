# Fine-Tuning Gemma 3 Vision for Offline Video Activity Understanding

This repository contains the implementation of a parameter-efficient fine-tuning pipeline and an interactive Video Q&A agent for the **Gemma 3 4B Vision** model. The project adapts the base Vision-Language Model (VLM) using **QLoRA** on the **ActivityNet Captions** dataset to understand activity segments in video streams, with a down-stream agent capable of identifying security threats and suggesting emergency action protocols.

---

## 1. System Architecture and Pipelines

The pipeline operates in two distinct phases: **Dataset Preprocessing & Fine-Tuning** and **Offline Inference & Video Q&A**.

### Fine-Tuning Pipeline
```
Raw Video Log (.mp4) 
   │
   ▼ (OpenCV Segment-Midpoint Sampling)
Representative Frame Extraction (.jpg)
   │
   ▼ (Dataset Path & Label Formatting)
train.jsonl / eval.jsonl (Cross-platform path serialization)
   │
   ▼ (Gemma3VLMDataCollator: lazy image loading + token-level masking)
TRL SFTTrainer (PEFT/QLoRA) ──► Saved LoRA Adapter (adapters/activitynet_v1)
```

### Inference and Q&A Pipeline
```
Input Video (e.g. sample.mp4)
   │
   ▼ (OpenCV Uniform Slicing)
Frame Extraction (e.g. 6 key frames)
   │
   ▼ (Hugging Face Backend + PEFT Model Loader)
Gemma 3 Base (Frozen) + LoRA Adapter (adapters/activitynet_v1)
   │
   ▼ (Prompt Synthesis: System Prompt + Video Log Context)
Interactive Q&A Agent CLI (video_qa.py)
   │
   ▼ (Safety & Emergency Parser)
Action Recommendation (e.g. Suggest Police/Ambulance/Fire Services)
```

---

## 2. Research Details & Parameter Counts

To achieve domain adaptation on consumer/free cloud hardware (like Tesla T4 GPUs) within a tight VRAM envelope, we frozen the base VLM parameters and updated a small fraction of the model using parameter-efficient fine-tuning:

* **Base Model**: `google/gemma-3-4b-it` (frozen 4.3B parameter Vision-Language Model).
* **Trainable Parameters**: **32,788,480** parameters (applied across target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
* **Trainable Percentage**: **0.7567%** of the **4,332,867,952** total model parameters.

### Quantization and LoRA Configuration
* **Quantization**: 4-bit NormalFloat (NF4) with Double Quantization enabled.
* **Compute Datatype**: `bfloat16` (loaded dynamically on GPU).
* **LoRA Hyperparameters**: Rank $r = 16$, scaling factor $\alpha = 32$, and dropout rate = $0.05$.
* **Prompt Masking**: Prompt tokens (including user instructions and image embeddings) are masked with a label ID of `-100` during collation. This ensures cross-entropy loss is calculated exclusively on the assistant's descriptions.


## 3. Experimental Setup & Metrics

### Training Parameters
* **Epochs**: 3
* **Learning Rate**: $2.0 \times 10^{-4}$ with a warmup ratio of $0.03$.
* **Batch Size**: 1 per device (Gradient accumulation steps = 4).
* **Gradient Checkpointing**: Enabled.
* **Max Sequence Length**: 2048 tokens.

### SFT Evaluation Results
Over the course of 3 training epochs on a Tesla T4 GPU, we observed the following validation metrics:

| Epoch | Evaluation Loss | Mean Token Accuracy |
|:---:|:---:|:---:|
| 1 | 11.580 | 22.22% |
| 2 | 5.156 | 55.56% |
| 3 | 4.196 | 55.56% |

*The significant decrease in loss and improvement in token accuracy show that the model successfully learned the vocabulary and descriptive structure of the target dataset.*

---

## 4. Execution Guide

Detailed setup steps can be found in [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md).

### Preprocess the Dataset
Extract segment midpoint frames and build train/validation splits:
```bash
python -m training.preprocess_activitynet --annotations friedrichor/ActivityNet_Captions --videos-dir <path_to_videos> --output-dir training/data
```

### Run Fine-Tuning
```bash
python -m training.train --config config/activitynet_training_config.yaml
```
*To resume training from a checkpoint, append `--resume` to the command.*

### Run Comparative Evaluation
Compare base VLM descriptions against your fine-tuned adapter:
```bash
python -m training.evaluate --config config/activitynet_training_config.yaml
```

### Run Video Q&A Agent
Run the interactive CLI agent to query video summaries and trigger safety protocols in emergency cases:
```bash
python video_qa.py --report outputs/report.json --backend hf --adapter activitynet_v1
```
