# Fine-Tuning Migration Guide

This document describes the minimal changes made to support optional QLoRA fine-tuning
while preserving the existing Ollama workflow.

## What stayed the same

| Component | Status |
|-----------|--------|
| `GemmaAnalyzer` (Ollama API client) | Unchanged public API |
| `frame_extractor.py` | No changes |
| `summarizer.py` | No changes |
| Default CLI behavior | Still uses Ollama with `--model gemma4` |
| `requirements.txt` | Unchanged (no ML dependencies added) |

## Files modified (minimal)

### `gemma_analyzer.py`

**Why:** Share prompt templates between inference and training so fine-tuned models learn the same JSON schema.

**Change:** Imports `FRAME_ANALYSIS_PROMPT` and `build_summary_prompt` from new `prompts.py`. Behavior is identical.

### `main.py`

**Why:** Allow optional HF + adapter backend without breaking the default path.

**Changes:**
- `GemmaAnalyzer(...)` replaced with `create_analyzer(...)` (defaults to Ollama)
- Added optional `--backend` and `--adapter` flags (both default to `None`)

**Existing command still works:**

```powershell
python main.py --video sample.mp4 --model gemma3:4b --frames 12
```

## New files (additive only)

```
prompts.py                    # Shared prompt templates
analyzer_factory.py           # Backend selection (Ollama vs HF)
hf_gemma_analyzer.py          # Optional HuggingFace inference backend
models/
  model_loader.py             # load_base_model, load_finetuned_model, load_adapter_if_available
  gemma_base/                 # Optional local base model cache (never overwritten)
adapters/
  threat_assessment/          # LoRA adapter output directory
  surveillance/
  activity_understanding/
training/
  train.py                    # QLoRA training entry point
  dataset.py                  # JSONL dataset loading
  config.py                   # TrainingConfig dataclass
  evaluate.py                 # Base vs fine-tuned comparison
config/
  adapter_config.yaml         # Adapter/backend configuration
  training_config.yaml        # QLoRA hyperparameters
scripts/
  merge_adapter.py            # Optional adapter merge utility
  compare_models.py           # Video frame comparison report
requirements-train.txt        # Training/HF dependencies (separate from main app)
```

## Enabling fine-tuned inference

### Option A: Environment variables

```powershell
$env:INFERENCE_BACKEND = "hf"
$env:ACTIVE_ADAPTER = "threat_assessment"
python main.py --video sample.mp4 --frames 6
```

### Option B: CLI flags

```powershell
python main.py --video sample.mp4 --backend hf --adapter threat_assessment
```

### Option C: Config file

Edit `config/adapter_config.yaml`:

```yaml
inference_backend: hf
active_adapter: threat_assessment
```

### Revert to base model instantly

```powershell
# Remove adapter — use base model only
python main.py --video sample.mp4 --backend hf

# Or revert to original Ollama workflow (default)
python main.py --video sample.mp4 --model gemma3:4b
```

## Training workflow

1. Install training dependencies:

```powershell
pip install -r requirements-train.txt
```

2. Prepare training data (JSONL format):

```powershell
python -m training.train --create-sample-data
```

3. Edit `config/training_config.yaml` and place your data in `training/data/`.

4. Run QLoRA training:

```powershell
python -m training.train --config config/training_config.yaml
```

5. Adapter weights are saved to `adapters/<name>/` — base model is never modified. Training checkpoints go to `adapters/<name>/checkpoints/`.

6. Resume from checkpoint:

```powershell
python -m training.train --resume
```

## Evaluation

Compare base vs fine-tuned on a held-out dataset:

```powershell
python -m training.evaluate --adapter adapters/threat_assessment
```

Compare on video frames:

```powershell
python scripts/compare_models.py --video sample.mp4 --adapter threat_assessment
```

## Optional: merge adapter for export

```powershell
python scripts/merge_adapter.py --adapter adapters/threat_assessment --output outputs/merged_model
```

Merged output goes to `outputs/merged_model/` — never touches `models/gemma_base/`.

## Switching between adapters

```powershell
python main.py --video sample.mp4 --backend hf --adapter surveillance
python main.py --video sample.mp4 --backend hf --adapter activity_understanding
```

## Ollama + fine-tuned model (no HF backend)

Train with QLoRA, merge the adapter, export to Ollama format externally, then:

```powershell
python main.py --video sample.mp4 --model your-finetuned-ollama-tag
```

No code changes required for this path.
