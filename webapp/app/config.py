"""
Central configuration for the SkyReels V2 Web Application.
All tuneable parameters in one place.
"""
import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AppConfig:
    # ── Model settings ────────────────────────────────────────────────────────
    default_model_id: str = "Skywork/SkyReels-V2-T2V-14B-540P"
    offload: bool = True           # CPU offload to save VRAM (set False if you have >24 GB)
    use_teacache: bool = True      # ~3x speed-up with minor quality trade-off
    teacache_thresh: float = 0.2

    # ── Generation defaults ───────────────────────────────────────────────────
    default_num_frames: int = 97
    default_fps: int = 24
    default_guidance_scale: float = 6.0
    default_shift: float = 8.0
    default_inference_steps: int = 30
    default_resolution: Literal["540P", "720P"] = "540P"

    # ── Prompt enhancer ───────────────────────────────────────────────────────
    prompt_enhancer_model: str = "Qwen/Qwen2.5-7B-Instruct"  # lighter than 32B default

    # ── Storage ───────────────────────────────────────────────────────────────
    output_dir: str = os.path.join(os.path.dirname(__file__), "..", "outputs")
    max_saved_videos: int = 50     # oldest videos auto-deleted beyond this count

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    gradio_port: int = 7860

    # ── Negative prompt ───────────────────────────────────────────────────────
    negative_prompt: str = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, "
        "works, paintings, images, static, overall gray, worst quality, low quality, "
        "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
        "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
        "still picture, messy background, three legs, many people in the background, "
        "walking backwards"
    )


# Singleton config instance
config = AppConfig()
