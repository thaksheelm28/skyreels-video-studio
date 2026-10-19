"""
Gradio Web UI for SkyReels V2 — AI Video Generation Studio

Tabs:
  1. Text → Video
  2. Image → Video
  3. Job History / Gallery
  4. System Info
"""
import os
import time
from pathlib import Path

import gradio as gr
import requests

# Point to the FastAPI backend
API_BASE = "http://127.0.0.1:8000"

# ── Model options ──────────────────────────────────────────────────────────────
T2V_MODELS = [
    "Skywork/SkyReels-V2-T2V-14B-540P",
    "Skywork/SkyReels-V2-T2V-14B-720P",
]
I2V_MODELS = [
    "Skywork/SkyReels-V2-I2V-1.3B-540P",
    "Skywork/SkyReels-V2-I2V-14B-540P",
    "Skywork/SkyReels-V2-I2V-14B-720P",
]

# Example prompts for inspiration
EXAMPLE_PROMPTS = [
    "A majestic eagle soaring over snow-capped mountains at golden hour",
    "Time-lapse of a city street from dawn to dusk, bustling with life",
    "Underwater coral reef with colorful fish darting between formations",
    "A cozy café on a rainy evening, warm lights reflecting on wet cobblestones",
    "Astronaut floating in zero gravity inside a spacecraft, Earth visible through porthole",
]


# ── API helpers ────────────────────────────────────────────────────────────────

def _poll_until_done(job_id: str, status_box):
    """Poll job status every 3s, yield progress updates."""
    url = f"{API_BASE}/jobs/{job_id}"
    while True:
        try:
            r = requests.get(url, timeout=10)
            data = r.json()
        except Exception as e:
            return None, f"❌ Network error: {e}"

        status = data.get("status", "unknown")
        progress = data.get("progress", 0)
        enhanced = data.get("enhanced_prompt", "")

        info = f"**Status:** `{status}` | **Progress:** {progress}%"
        if enhanced:
            info += f"\n\n**Enhanced prompt:** _{enhanced}_"
        if status == "failed":
            return None, f"❌ Job failed:\n```\n{data.get('error_message', 'Unknown error')}\n```"
        if status == "done":
            video_url = data.get("video_url")
            if video_url:
                video_path = _download_video(f"{API_BASE}{video_url}", job_id)
                dur = data.get("duration_sec")
                info += f"\n\n✅ **Done in {dur}s** | Seed: `{data.get('seed')}`"
                return video_path, info
        time.sleep(3)


def _download_video(url: str, job_id: str) -> str:
    """Download the generated video to a local temp path for Gradio to display."""
    tmp_dir = Path(__file__).parent.parent / "tmp_preview"
    tmp_dir.mkdir(exist_ok=True)
    local_path = str(tmp_dir / f"{job_id}.mp4")
    r = requests.get(url, stream=True, timeout=120)
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return local_path


def _get_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        d = r.json()
        gpu = d.get("gpu", {})
        q = d.get("queue_depth", 0)
        if gpu:
            return (
                f"✅ **API Online** | Queue: {q} job(s)\n\n"
                f"🎮 **GPU:** {gpu['gpu_name']} | "
                f"VRAM: {gpu['vram_used_gb']}/{gpu['vram_total_gb']} GB"
            )
        return f"✅ **API Online** | Queue: {q} job(s) | ⚠️ No GPU detected"
    except Exception:
        return "❌ **API Offline** — make sure the FastAPI server is running"


# ── Generation functions ───────────────────────────────────────────────────────

def generate_t2v(
    prompt, model_id, resolution, num_frames, fps,
    guidance_scale, shift, inference_steps,
    use_enhancer, seed,
):
    if not prompt.strip():
        return None, "⚠️ Please enter a prompt."

    form = {
        "prompt": prompt,
        "model_id": model_id,
        "resolution": resolution,
        "num_frames": int(num_frames),
        "fps": int(fps),
        "guidance_scale": float(guidance_scale),
        "shift": float(shift),
        "inference_steps": int(inference_steps),
        "use_prompt_enhancer": str(use_enhancer).lower(),
    }
    if seed and int(seed) >= 0:
        form["seed"] = int(seed)

    try:
        r = requests.post(f"{API_BASE}/generate", data=form, timeout=30)
        r.raise_for_status()
        job_id = r.json()["job_id"]
    except Exception as e:
        return None, f"❌ Failed to submit job: {e}"

    return _poll_until_done(job_id, None)


def generate_i2v(
    prompt, image, model_id, resolution, num_frames, fps,
    guidance_scale, shift, inference_steps, seed,
):
    if not prompt.strip():
        return None, "⚠️ Please enter a prompt."
    if image is None:
        return None, "⚠️ Please upload a reference image."

    import tempfile
    from PIL import Image as PILImage
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        PILImage.fromarray(image).save(tmp.name)
        img_path = tmp.name

    form = {
        "prompt": prompt,
        "model_id": model_id,
        "resolution": resolution,
        "num_frames": int(num_frames),
        "fps": int(fps),
        "guidance_scale": float(guidance_scale),
        "shift": float(shift),
        "inference_steps": int(inference_steps),
        "use_prompt_enhancer": "false",
    }
    if seed and int(seed) >= 0:
        form["seed"] = int(seed)

    try:
        with open(img_path, "rb") as img_file:
            r = requests.post(
                f"{API_BASE}/generate",
                data=form,
                files={"image": ("image.png", img_file, "image/png")},
                timeout=30,
            )
        r.raise_for_status()
        job_id = r.json()["job_id"]
    except Exception as e:
        return None, f"❌ Failed to submit job: {e}"

    return _poll_until_done(job_id, None)


def load_history():
    try:
        r = requests.get(f"{API_BASE}/jobs", timeout=5)
        jobs = r.json().get("jobs", [])
        rows = []
        for j in sorted(jobs, key=lambda x: x["created_at"], reverse=True):
            rows.append([
                j["job_id"][:8] + "…",
                j["status"],
                j["prompt"][:60] + ("…" if len(j["prompt"]) > 60 else ""),
                j["resolution"],
                j.get("duration_sec", "—"),
                j["seed"],
            ])
        return rows
    except Exception:
        return []


# ── Build UI ───────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="SkyReels V2 — AI Video Studio",
        theme=gr.themes.Soft(primary_hue="violet"),
    ) as demo:

        gr.Markdown(
            """
            # 🎬 SkyReels V2 — AI Video Generation Studio
            *Powered by SkyReels-V2 diffusion models · FastAPI backend · Async job queue*
            """
        )

        # ── Health bar ────────────────────────────────────────────────────
        with gr.Row():
            health_md = gr.Markdown(_get_health())
            refresh_btn = gr.Button("🔄 Refresh Status", size="sm", scale=0)
        refresh_btn.click(fn=_get_health, outputs=health_md)

        # ── Tabs ──────────────────────────────────────────────────────────
        with gr.Tabs():

            # ── Tab 1: Text → Video ───────────────────────────────────────
            with gr.Tab("📝 Text → Video"):
                with gr.Row():
                    with gr.Column(scale=2):
                        t2v_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the video you want to create…",
                            lines=4,
                        )
                        gr.Examples(
                            examples=EXAMPLE_PROMPTS,
                            inputs=t2v_prompt,
                            label="✨ Example prompts",
                        )
                        t2v_enhancer = gr.Checkbox(
                            label="🪄 Auto-enhance prompt (uses LLM)", value=False
                        )

                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            t2v_model = gr.Dropdown(T2V_MODELS, value=T2V_MODELS[0], label="Model")
                            t2v_res = gr.Radio(["540P", "720P"], value="540P", label="Resolution")
                            with gr.Row():
                                t2v_frames = gr.Slider(17, 193, value=97, step=8, label="Frames")
                                t2v_fps = gr.Slider(8, 30, value=24, step=1, label="FPS")
                            with gr.Row():
                                t2v_guidance = gr.Slider(1.0, 15.0, value=6.0, step=0.5, label="Guidance Scale")
                                t2v_shift = gr.Slider(1.0, 20.0, value=8.0, step=0.5, label="Shift")
                            t2v_steps = gr.Slider(10, 50, value=30, step=5, label="Inference Steps")
                            t2v_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)

                        t2v_btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")

                    with gr.Column(scale=3):
                        t2v_video = gr.Video(label="Generated Video", height=480)
                        t2v_status = gr.Markdown("_Waiting for generation…_")

                t2v_btn.click(
                    fn=generate_t2v,
                    inputs=[
                        t2v_prompt, t2v_model, t2v_res, t2v_frames, t2v_fps,
                        t2v_guidance, t2v_shift, t2v_steps, t2v_enhancer, t2v_seed,
                    ],
                    outputs=[t2v_video, t2v_status],
                )

            # ── Tab 2: Image → Video ──────────────────────────────────────
            with gr.Tab("🖼️ Image → Video"):
                with gr.Row():
                    with gr.Column(scale=2):
                        i2v_prompt = gr.Textbox(
                            label="Motion Prompt",
                            placeholder="Describe the motion to apply to the image…",
                            lines=3,
                        )
                        i2v_image = gr.Image(label="Reference Image", type="numpy", height=250)

                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            i2v_model = gr.Dropdown(I2V_MODELS, value=I2V_MODELS[0], label="Model")
                            i2v_res = gr.Radio(["540P", "720P"], value="540P", label="Resolution")
                            with gr.Row():
                                i2v_frames = gr.Slider(17, 193, value=97, step=8, label="Frames")
                                i2v_fps = gr.Slider(8, 30, value=24, step=1, label="FPS")
                            with gr.Row():
                                i2v_guidance = gr.Slider(1.0, 15.0, value=6.0, step=0.5, label="Guidance Scale")
                                i2v_shift = gr.Slider(1.0, 20.0, value=8.0, step=0.5, label="Shift")
                            i2v_steps = gr.Slider(10, 50, value=30, step=5, label="Inference Steps")
                            i2v_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)

                        i2v_btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")

                    with gr.Column(scale=3):
                        i2v_video = gr.Video(label="Generated Video", height=480)
                        i2v_status = gr.Markdown("_Waiting for generation…_")

                i2v_btn.click(
                    fn=generate_i2v,
                    inputs=[
                        i2v_prompt, i2v_image, i2v_model, i2v_res, i2v_frames, i2v_fps,
                        i2v_guidance, i2v_shift, i2v_steps, i2v_seed,
                    ],
                    outputs=[i2v_video, i2v_status],
                )

            # ── Tab 3: Job History ────────────────────────────────────────
            with gr.Tab("📋 Job History"):
                history_btn = gr.Button("🔄 Refresh History")
                history_table = gr.Dataframe(
                    headers=["Job ID", "Status", "Prompt", "Resolution", "Duration (s)", "Seed"],
                    datatype=["str", "str", "str", "str", "number", "number"],
                    interactive=False,
                    wrap=True,
                )
                history_btn.click(fn=load_history, outputs=history_table)
                demo.load(fn=load_history, outputs=history_table)

            # ── Tab 4: System Info ────────────────────────────────────────
            with gr.Tab("🖥️ System Info"):
                gr.Markdown("""
                ## Architecture

                ```
                ┌─────────────────────────────────────────────────────┐
                │                   Gradio Web UI                     │
                │   Tab: T2V  │  Tab: I2V  │  History  │  System      │
                └──────────────────────┬──────────────────────────────┘
                                       │ HTTP REST
                ┌──────────────────────▼──────────────────────────────┐
                │               FastAPI Backend (:8000)               │
                │   POST /generate  GET /jobs/{id}  GET /videos/*     │
                └──────────────────────┬──────────────────────────────┘
                                       │
                ┌──────────────────────▼──────────────────────────────┐
                │            Job Queue (background thread)            │
                │  QUEUED → ENHANCING → GENERATING → DONE / FAILED    │
                └──────────────────────┬──────────────────────────────┘
                                       │
                ┌──────────────────────▼──────────────────────────────┐
                │           SkyReels V2 Inference Engine              │
                │  PromptEnhancer │ Text2VideoPipeline │ I2VPipeline  │
                │           TeaCache · GPU Offload · CUDA             │
                └─────────────────────────────────────────────────────┘
                ```

                ## Key Features
                - ⚡ **TeaCache** — ~3× faster inference with minimal quality loss
                - 🪄 **Prompt Enhancer** — LLM expands short prompts into rich cinematic descriptions
                - 🖼️ **Image-to-Video** — Animate any still image with natural motion
                - 📦 **Async Queue** — Submit multiple jobs, GPU runs them one at a time
                - 🗃️ **Auto-cleanup** — Old videos auto-deleted when storage limit reached
                """)

    return demo


if __name__ == "__main__":
    from .config import config
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=config.gradio_port,
        share=False,
        show_api=True,
    )
