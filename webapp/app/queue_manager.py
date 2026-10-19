"""
Job queue and lifecycle management.

Design:
- Jobs are stored in a thread-safe in-memory dict (swap for Redis/SQLite in production)
- A single background thread processes jobs one at a time (GPU is single-tenant)
- Status updates are pushed via asyncio Event so the API can long-poll cheaply
"""
import gc
import os
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import torch


class JobStatus(str, Enum):
    QUEUED = "queued"
    ENHANCING = "enhancing"          # LLM prompt enhancement in progress
    GENERATING = "generating"        # Diffusion inference running
    DONE = "done"
    FAILED = "failed"


@dataclass
class GenerationJob:
    job_id: str
    prompt: str
    model_id: str
    resolution: str
    num_frames: int
    fps: int
    guidance_scale: float
    shift: float
    inference_steps: int
    use_prompt_enhancer: bool
    seed: int
    image_path: Optional[str] = None       # for image-to-video jobs
    use_teacache: bool = True
    teacache_thresh: float = 0.2

    # Runtime state (not from user)
    status: JobStatus = JobStatus.QUEUED
    enhanced_prompt: Optional[str] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: int = 0                      # 0–100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "prompt": self.prompt,
            "enhanced_prompt": self.enhanced_prompt,
            "model_id": self.model_id,
            "resolution": self.resolution,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "guidance_scale": self.guidance_scale,
            "shift": self.shift,
            "inference_steps": self.inference_steps,
            "seed": self.seed,
            "status": self.status.value,
            "output_path": self.output_path,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": self.progress,
            "duration_sec": (
                round(self.finished_at - self.started_at, 1)
                if self.started_at and self.finished_at
                else None
            ),
        }


class JobQueue:
    """Thread-safe job manager with a single worker thread for GPU generation."""

    def __init__(self):
        self._jobs: Dict[str, GenerationJob] = {}
        self._queue: list[str] = []           # ordered list of job_ids
        self._lock = threading.Lock()
        self._event = threading.Event()       # signals new job arrived
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # ── Public API ─────────────────────────────────────────────────────────

    def submit(self, job: GenerationJob) -> str:
        with self._lock:
            self._jobs[job.job_id] = job
            self._queue.append(job.job_id)
        self._event.set()
        return job.job_id

    def get(self, job_id: str) -> Optional[GenerationJob]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[Dict]:
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]

    def queue_length(self) -> int:
        with self._lock:
            return sum(
                1 for jid in self._queue
                if self._jobs[jid].status == JobStatus.QUEUED
            )

    # ── Worker ─────────────────────────────────────────────────────────────

    def _worker_loop(self):
        while True:
            self._event.wait()
            self._event.clear()
            while True:
                job = self._pop_next_job()
                if job is None:
                    break
                self._run_job(job)

    def _pop_next_job(self) -> Optional[GenerationJob]:
        with self._lock:
            for jid in self._queue:
                if self._jobs[jid].status == JobStatus.QUEUED:
                    return self._jobs[jid]
        return None

    def _run_job(self, job: GenerationJob):
        import random
        import imageio
        from diffusers.utils import load_image

        # Lazy imports to avoid loading SkyReels at server startup
        import sys
        root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
        if root not in sys.path:
            sys.path.insert(0, root)

        from skyreels_v2_infer.modules import download_model
        from skyreels_v2_infer.pipelines import (
            Image2VideoPipeline,
            PromptEnhancer,
            Text2VideoPipeline,
            resizecrop,
        )
        from .config import config

        job.started_at = time.time()

        try:
            # ── 1. Resolve resolution ──────────────────────────────────────
            if job.resolution == "540P":
                height, width = 544, 960
            else:
                height, width = 720, 1280

            # ── 2. Download / locate model ────────────────────────────────
            job.status = JobStatus.ENHANCING
            model_path = download_model(job.model_id)

            # ── 3. Prompt enhancement (optional) ──────────────────────────
            prompt_input = job.prompt
            if job.use_prompt_enhancer and job.image_path is None:
                enhancer = PromptEnhancer(model_name=config.prompt_enhancer_model)
                prompt_input = enhancer(prompt_input)
                job.enhanced_prompt = prompt_input
                del enhancer
                gc.collect()
                torch.cuda.empty_cache()

            job.status = JobStatus.GENERATING
            job.progress = 5

            # ── 4. Build pipeline ─────────────────────────────────────────
            image = None
            if job.image_path:
                image = load_image(job.image_path).convert("RGB")
                image_width, image_height = image.size
                if image_height > image_width:
                    height, width = width, height
                image = resizecrop(image, height, width)
                pipe = Image2VideoPipeline(
                    model_path=model_path,
                    dit_path=model_path,
                    offload=config.offload,
                )
            else:
                pipe = Text2VideoPipeline(
                    model_path=model_path,
                    dit_path=model_path,
                    offload=config.offload,
                )

            if job.use_teacache:
                pipe.transformer.initialize_teacache(
                    enable_teacache=True,
                    num_steps=job.inference_steps,
                    teacache_thresh=job.teacache_thresh,
                    use_ret_steps=False,
                    ckpt_dir=model_path,
                )

            job.progress = 20

            # ── 5. Run inference ──────────────────────────────────────────
            kwargs = {
                "prompt": prompt_input,
                "negative_prompt": config.negative_prompt,
                "num_frames": job.num_frames,
                "num_inference_steps": job.inference_steps,
                "guidance_scale": job.guidance_scale,
                "shift": job.shift,
                "generator": torch.Generator(device="cuda").manual_seed(job.seed),
                "height": height,
                "width": width,
            }
            if image is not None:
                kwargs["image"] = image

            with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
                video_frames = pipe(**kwargs)[0]

            job.progress = 90

            # ── 6. Save output ────────────────────────────────────────────
            os.makedirs(config.output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{job.job_id}_{timestamp}.mp4"
            out_path = os.path.join(config.output_dir, filename)
            imageio.mimwrite(
                out_path, video_frames, fps=job.fps, quality=8,
                output_params=["-loglevel", "error"]
            )
            job.output_path = out_path
            job.status = JobStatus.DONE
            job.progress = 100

            # Cleanup old videos
            _cleanup_old_videos(config.output_dir, config.max_saved_videos)

        except Exception:
            job.status = JobStatus.FAILED
            job.error_message = traceback.format_exc()

        finally:
            job.finished_at = time.time()
            # Free GPU memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _cleanup_old_videos(output_dir: str, max_count: int):
    """Delete oldest mp4 files when count exceeds max_count."""
    files = sorted(
        [f for f in os.listdir(output_dir) if f.endswith(".mp4")],
        key=lambda f: os.path.getmtime(os.path.join(output_dir, f)),
    )
    while len(files) > max_count:
        os.remove(os.path.join(output_dir, files.pop(0)))


# Global singleton
job_queue = JobQueue()
