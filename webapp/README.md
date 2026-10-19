# 🎬 SkyReels V2 — AI Video Generation Studio

> **A production-grade, full-stack AI video generation web application** built on top of
> [SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2) diffusion models with a
> FastAPI REST backend, async job queue, LLM-powered prompt enhancement, and an
> interactive Gradio UI.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📝 **Text → Video** | Generate videos from natural language descriptions |
| 🖼️ **Image → Video** | Animate any still image with natural motion |
| 🪄 **Prompt Enhancer** | LLM (Qwen2.5) automatically expands short prompts into rich cinematic descriptions |
| ⚡ **TeaCache** | ~3× faster inference with minimal quality loss |
| 📦 **Async Job Queue** | Submit jobs; GPU processes one at a time in background |
| 🔌 **REST API** | Full FastAPI backend with Swagger docs at `/docs` |
| 🗃️ **Auto-cleanup** | Old videos deleted automatically when storage limit reached |
| 📊 **Job History** | Browse all past generations with status, seed, and duration |

---

## 🏗️ Architecture

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

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
# Base SkyReels dependencies
pip install -r requirements.txt

# Webapp layer
pip install -r webapp/requirements.txt
```

### 2. Launch the app

```bash
# From the project root:
python -m webapp.main
```

This starts:
- **FastAPI** at `http://localhost:8000` (REST API + Swagger docs)
- **Gradio UI** at `http://localhost:7860`

### 3. Open the UI

Navigate to `http://localhost:7860` in your browser.

---

## 📡 REST API

The FastAPI backend is fully documented at `http://localhost:8000/docs`.

### Submit a Text-to-Video job

```bash
curl -X POST http://localhost:8000/generate \
  -F "prompt=A majestic eagle soaring over mountains at golden hour" \
  -F "resolution=540P" \
  -F "num_frames=97" \
  -F "use_prompt_enhancer=true"
```

Response:
```json
{
  "job_id": "abc123...",
  "status": "queued",
  "queue_position": 1,
  "message": "Job queued successfully. Poll /jobs/{job_id} for updates."
}
```

### Poll job status

```bash
curl http://localhost:8000/jobs/abc123...
```

Response (when done):
```json
{
  "job_id": "abc123...",
  "status": "done",
  "enhanced_prompt": "Close-up shot of a bald eagle with outstretched wings...",
  "video_url": "/videos/abc123_20240101_120000.mp4",
  "duration_sec": 142.3,
  "seed": 3847291
}
```

### List all jobs

```bash
curl http://localhost:8000/jobs
```

### Health check

```bash
curl http://localhost:8000/health
```

---

## ⚙️ Configuration

All settings are in [`webapp/app/config.py`](webapp/app/config.py):

```python
@dataclass
class AppConfig:
    default_model_id: str = "Skywork/SkyReels-V2-T2V-14B-540P"
    offload: bool = True           # CPU offload — set False if you have >24 GB VRAM
    use_teacache: bool = True      # ~3x speedup
    teacache_thresh: float = 0.2   # 0.1 = 2x speedup, 0.2 = 3x speedup
    default_num_frames: int = 97
    default_fps: int = 24
    default_inference_steps: int = 30
    prompt_enhancer_model: str = "Qwen/Qwen2.5-7B-Instruct"
    max_saved_videos: int = 50     # auto-delete oldest beyond this
```

---

## 📁 Project Structure

```
SkyReels-V2-main/
├── skyreels_v2_infer/          # Core model inference (original)
│   ├── pipelines/
│   │   ├── text2video_pipeline.py
│   │   ├── image2video_pipeline.py
│   │   └── prompt_enhancer.py
│   ├── modules/
│   └── scheduler/
├── webapp/                     # 🆕 Production web application
│   ├── app/
│   │   ├── config.py           # Centralised configuration
│   │   ├── queue_manager.py    # Async job queue + GPU worker
│   │   ├── api.py              # FastAPI REST backend
│   │   └── ui.py               # Gradio web UI
│   ├── main.py                 # Launch script
│   ├── requirements.txt
│   └── outputs/                # Generated videos saved here
├── generate_video.py           # Original CLI script
└── requirements.txt
```

---

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | 12 GB VRAM (with offload) | 24+ GB VRAM |
| RAM | 32 GB | 64 GB |
| Storage | 50 GB (model weights) | 100 GB |
| CUDA | 12.1+ | 12.4+ |
| Python | 3.10+ | 3.11+ |

---

## 🛠️ Tech Stack

- **Model:** SkyReels-V2 (Wan2.1-based video diffusion transformer, 1.3B–14B params)
- **Backend:** FastAPI + Uvicorn
- **Frontend:** Gradio 4.x
- **Inference:** PyTorch + CUDA, bfloat16, TeaCache acceleration
- **Prompt Enhancement:** Qwen2.5-Instruct (via HuggingFace Transformers)
- **Video I/O:** imageio-ffmpeg

---

## 📈 Performance

| Config | Frames | Resolution | Steps | Time (A100) |
|---|---|---|---|---|
| TeaCache ON, Offload OFF | 97 | 540P | 30 | ~2.5 min |
| TeaCache ON, Offload ON | 97 | 540P | 30 | ~4 min |
| TeaCache OFF, Offload OFF | 97 | 540P | 30 | ~7 min |

---

## 📄 License

Built on [SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2) — see original [LICENSE](LICENSE.txt).
