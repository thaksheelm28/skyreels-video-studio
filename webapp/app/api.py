"""
FastAPI REST backend for SkyReels V2 Video Generation.

Endpoints:
  POST /generate          – submit a new T2V or I2V job
  GET  /jobs/{job_id}     – poll job status + result URL
  GET  /jobs              – list all jobs
  GET  /videos/{filename} – stream / download a generated video
  GET  /health            – health check + GPU info
"""
import os
import random
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import config
from .queue_manager import GenerationJob, JobStatus, job_queue

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SkyReels V2 – AI Video Generation API",
    description=(
        "Production-grade REST API wrapping SkyReels-V2 diffusion video models. "
        "Supports Text-to-Video and Image-to-Video with async job queuing."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated videos as static files
os.makedirs(config.output_dir, exist_ok=True)
app.mount("/videos", StaticFiles(directory=config.output_dir), name="videos")

# Temp upload dir
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Returns server health, GPU info, and current queue depth."""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
            "vram_used_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
        }
    return {
        "status": "ok",
        "queue_depth": job_queue.queue_length(),
        "gpu": gpu_info,
    }


@app.post("/generate", tags=["Generation"])
async def submit_generation(
    prompt: str = Form(..., description="Short or detailed video description"),
    model_id: str = Form(config.default_model_id),
    resolution: str = Form(config.default_resolution),
    num_frames: int = Form(config.default_num_frames),
    fps: int = Form(config.default_fps),
    guidance_scale: float = Form(config.default_guidance_scale),
    shift: float = Form(config.default_shift),
    inference_steps: int = Form(config.default_inference_steps),
    use_prompt_enhancer: bool = Form(False),
    seed: Optional[int] = Form(None),
    image: Optional[UploadFile] = File(None),
):
    """
    Submit a video generation job.

    - **prompt**: The text description of the video you want to generate.
    - **model_id**: Which SkyReels model to use (T2V or I2V).
    - **resolution**: 540P (faster) or 720P (higher quality).
    - **use_prompt_enhancer**: Automatically expand a short prompt using an LLM.
    - **image**: (Optional) Upload a reference image for Image-to-Video generation.

    Returns a `job_id` you can poll via `GET /jobs/{job_id}`.
    """
    # Validate model / mode consistency
    if image is not None and "T2V" in model_id:
        raise HTTPException(
            400,
            "Image-to-Video mode requires an I2V model. "
            "Use e.g. Skywork/SkyReels-V2-I2V-1.3B-540P"
        )
    if image is None and "I2V" in model_id:
        raise HTTPException(
            400,
            "Text-to-Video mode requires a T2V model. "
            "Use e.g. Skywork/SkyReels-V2-T2V-14B-540P"
        )
    if resolution not in ("540P", "720P"):
        raise HTTPException(400, "resolution must be '540P' or '720P'")

    # Save uploaded image if present
    image_path = None
    if image:
        ext = Path(image.filename).suffix or ".png"
        image_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
        with open(image_path, "wb") as f:
            f.write(await image.read())

    actual_seed = seed if seed is not None else random.randint(0, 4_294_967_294)

    job = GenerationJob(
        job_id=str(uuid.uuid4()),
        prompt=prompt,
        model_id=model_id,
        resolution=resolution,
        num_frames=num_frames,
        fps=fps,
        guidance_scale=guidance_scale,
        shift=shift,
        inference_steps=inference_steps,
        use_prompt_enhancer=use_prompt_enhancer,
        seed=actual_seed,
        image_path=image_path,
        use_teacache=config.use_teacache,
        teacache_thresh=config.teacache_thresh,
    )
    job_queue.submit(job)

    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "queue_position": job_queue.queue_length(),
        "message": "Job queued successfully. Poll /jobs/{job_id} for updates.",
    }


@app.get("/jobs/{job_id}", tags=["Generation"])
def get_job_status(job_id: str):
    """Poll the status of a specific generation job."""
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    data = job.to_dict()

    # Add a public video URL if generation is done
    if job.status == JobStatus.DONE and job.output_path:
        filename = os.path.basename(job.output_path)
        data["video_url"] = f"/videos/{filename}"

    return JSONResponse(data)


@app.get("/jobs", tags=["Generation"])
def list_all_jobs():
    """List all jobs (queued, running, done, failed)."""
    jobs = job_queue.list_jobs()
    # Attach video URLs
    for j in jobs:
        if j["status"] == "done" and j.get("output_path"):
            j["video_url"] = f"/videos/{os.path.basename(j['output_path'])}"
    return {"total": len(jobs), "jobs": jobs}


@app.delete("/jobs/{job_id}", tags=["Generation"])
def delete_job_video(job_id: str):
    """Delete the output video file for a completed job (frees disk space)."""
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    if job.output_path and os.path.exists(job.output_path):
        os.remove(job.output_path)
        job.output_path = None
        return {"message": "Video deleted successfully."}
    return {"message": "No video file to delete."}
