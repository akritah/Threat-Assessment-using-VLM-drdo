# VLM Fine-Tuning Guide (SFT & QLoRA)

This document outlines the dataset preparation and QLoRA fine-tuning process for adapting Gemma 3 4B IT to surveillance threat detection tasks.

---

## 1. SFT Dataset Compilation

### Data Compiler Module
*   **Path:** `scripts/prepare_surveillance_sft_dataset.py`
*   **Role:** Reads raw directories of either flat pre-extracted frames (UCF-Crime format) or raw video clips (XD-Violence format), groups them by prefix, samples representative frames, and outputs SFT datasets in JSONL format.

### SFT Task Mappings
For each target category (e.g. `Fighting`, `Riot`, `Explosion`), the compiler synthesizes 4 distinct training tasks/messages to ensure robust multi-task learning:
1.  **Action Captioning (Stage 1 Target):**
    *   *Prompt:* `"Describe the exact activity happening in this video sequence as a concise caption (e.g., A person is performing...)."`
    *   *Response:* A concise semantic label description of the threat.
2.  **Threat Report (Stage 2 Target):**
    *   *Prompt:* Detailed query asking for observed behaviors and threat level.
    *   *Response:* Structured markdown report containing threat level (Low/Medium/High) and reasoning.
3.  **Suspicious Indicators:**
    *   *Prompt:* Queries about weapon presence or hostilities.
    *   *Response:* True/False flag and brief explanation.
4.  **Emergency Service Dispatch:**
    *   *Prompt:* Query asking which emergency responder to dispatch.
    *   *Response:* Action protocol (e.g. Police/SWAT/Fire/Ambulance).

---

## 2. Training Configurations

The primary config is stored in `configs/colab_surveillance_training_config.yaml`.

### Quantization & Precision (FP16 Optimization)
To run within the memory limits of a standard **NVIDIA T4 GPU** without emulated software overhead (which slows down training by 20x), the pipeline uses native **FP16** precision:
*   `load_in_4bit: true` (Uses NF4 double quantization to fit the model in <6GB of VRAM).
*   `bnb_4bit_compute_dtype: float16` (Ensures compute calculations run on the GPU's hardware Tensor Cores).
*   `fp16: true`
*   `bf16: false` (T4 has no hardware execution units for bf16; using bf16 causes emulated CPU fallback).

### LoRA Configurations
*   `lora_r: 8`
*   `lora_alpha: 16`
*   `lora_dropout: 0.05`
*   `lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]` (Targeting attention matrices is memory-efficient and prevents over-parameterization).

### Training Hyperparameters
*   `num_train_epochs: 2` (or 3 depending on dataset size).
*   `per_device_train_batch_size: 1`
*   `gradient_accumulation_steps: 8` (Simulates an effective batch size of 8).
*   `learning_rate: 1.0e-4`
*   `max_seq_length: 1024` (Reduces sequence pad overhead for faster execution).

---

## 3. How to Run Training

Verify your dataset is mounted and execute SFT compilation, then run training:

```bash
# 1. Compile dataset (example: 250 videos -> 1,000 samples)
python scripts/prepare_surveillance_sft_dataset.py \
  --dataset-dir /kaggle/input/datasets/odins0n/ucf-crime-dataset \
  --output-dir training/data \
  --max-videos 250

# 2. Run QLoRA Trainer
HF_TOKEN="your_token" python training/train.py \
  --config configs/colab_surveillance_training_config.yaml
```

The adapter checkpoints and log files will be saved in `adapters/surveillance_colab/`.
