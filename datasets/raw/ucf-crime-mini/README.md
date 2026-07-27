---
pretty_name: UCF-Crime Mini Teacher Dataset
task_categories:
- video-captioning
- video-classification
- text-generation
language:
- en
license: other
size_categories:
- 1K<n<10K
---

UCF-Crime Mini Teacher Dataset

What this is
- A small local subset of UCF-Crime prepared for CCTV/anomaly-captioning experiments.
- Includes 42 videos across 6 categories.
- Includes dense teacher captions generated with openai-codex gpt-5.2.
- Includes distilled teacher captions used for student-model training experiments.

Contents
- videos/
  - 42 mp4 files
  - categories: Abuse, Arrest, Assault, Burglary, Fighting, normal
- annotations/teacher_ucf.jsonl
  - raw dense teacher captions
- annotations/teacher_ucf_distilled.jsonl
  - concise distilled captions for training
- annotations/manifest.json
  - counts and basic metadata

Summary
- videos: 42
- categories: 6 (7 videos each)
- total video size: about 0.894 GB
- raw teacher rows: 459
- distilled teacher rows: 459
- teacher model: openai-codex gpt-5.2

JSONL fields
Common fields in teacher JSONL files include:
- video
- video_path
- chunk_idx
- start_sec
- end_sec
- frame_times
- caption
- teacher_model
- chunk_duration
- frames_per_chunk

Notes
- This is a research/export package assembled from a local UCF-Crime mini subset in the user's workspace.
- UCF-Crime full is a separate larger source dataset.
- The distilled captions are downstream training artifacts derived from the raw teacher captions.
