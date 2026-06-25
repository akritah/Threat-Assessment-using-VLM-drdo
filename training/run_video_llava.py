"""Inference script for Video-LLaVA direct video processing baseline comparison."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import av
import numpy as np
import torch

logger = logging.getLogger(__name__)


def read_video_pyav(video_path: Path, num_frames: int = 8) -> np.ndarray:
    """Decode and sample video frames uniformly using PyAV."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    container = av.open(str(video_path))
    video_stream = container.streams.video[0]
    total_frames = video_stream.frames

    if total_frames <= 0:
        # Fallback to estimating via duration and fps
        fps = video_stream.average_rate
        duration = video_stream.duration * video_stream.time_base
        total_frames = int(duration * fps) if duration and fps else 100

    # Calculate uniform indices
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    max_idx = indices[-1]

    frames = []
    container.seek(0)
    
    frame_count = 0
    for frame in container.decode(video=0):
        if frame_count > max_idx:
            break
        if frame_count in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
        frame_count += 1

    # Safe fallback if PyAV decoding returned fewer frames than expected
    if not frames:
        container.seek(0)
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= num_frames:
                break

    # If it is still empty (e.g. corrupt stream), raise error
    if not frames:
        raise RuntimeError(f"Failed to decode any frames from video: {video_path}")

    # Pad if decoded frames is less than num_frames
    while len(frames) < num_frames:
        frames.append(frames[-1])

    # If we have more than num_frames, slice it
    if len(frames) > num_frames:
        frames = frames[:num_frames]

    return np.stack(frames)


def run_inference(
    video_path: Path,
    prompt: str,
    model_id: str = "LanguageBind/Video-LLaVA-7B-HF",
    num_frames: int = 8,
) -> str:
    from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor

    # CPU Fallback check
    is_cuda = torch.cuda.is_available()
    device = "cuda" if is_cuda else "cpu"
    
    logger.info("Initializing Video-LLaVA (model: %s, device: %s)...", model_id, device)
    
    load_kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    
    if is_cuda:
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = {"": "cpu"}

    # Pass HF Token if configured
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        load_kwargs["token"] = hf_token

    processor = VideoLlavaProcessor.from_pretrained(model_id, token=hf_token)
    model = VideoLlavaForConditionalGeneration.from_pretrained(model_id, **load_kwargs)

    logger.info("Decoding video and extracting frames...")
    video_data = read_video_pyav(video_path, num_frames=num_frames)

    # Format instruction in LLaVA dialogue structure
    formatted_prompt = f"USER: <video>\n{prompt} ASSISTANT:"
    
    logger.info("Preparing model inputs...")
    inputs = processor(text=formatted_prompt, videos=video_data, return_tensors="pt")
    
    # Map inputs to execution device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if is_cuda:
        inputs = {k: (v.to(torch.float16) if v.dtype == torch.float32 else v) for k, v in inputs.items()}

    logger.info("Generating model response...")
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[-1]
    response = processor.batch_decode(
        generate_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return response


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    # Ensure project root is in path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        
    import env_loader
    env_loader.load_env()

    parser = argparse.ArgumentParser(description="Video-LLaVA Direct Video Inference Baseline")
    parser.add_argument("--video", required=True, help="Path to local video file")
    parser.add_argument(
        "--prompt",
        default="Describe what is happening in this video clip.",
        help="Prompt instruction for the model",
    )
    parser.add_argument(
        "--model",
        default="LanguageBind/Video-LLaVA-7B-HF",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=8,
        help="Number of video frames to sample",
    )
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    try:
        response = run_inference(
            video_path=video_path,
            prompt=args.prompt,
            model_id=args.model,
            num_frames=args.frames,
        )
        print("\n" + "=" * 60)
        print(f"Video-LLaVA Output:\n{response}")
        print("=" * 60 + "\n")
        return 0
    except Exception as exc:
        logger.error("Video-LLaVA inference failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
