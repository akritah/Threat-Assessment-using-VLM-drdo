# Offline Video Activity Understanding

Small offline Python app for extracting video frames with OpenCV and analyzing them through a local Ollama multimodal model.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start Ollama in another terminal:

```powershell
ollama serve
```

Check installed models:

```powershell
ollama list
```

## Run

Put a local video in the project folder, or pass its full path.

```powershell
python main.py --video sample.mp4 --model gemma3:4b --frames 12
```

The app writes extracted frames and reports to `outputs/`.

## Notes

- Processing stays local.
- Videos, generated frames, reports, virtual environments, and `.env` files are ignored by Git.
- Use an Ollama model that supports images.
