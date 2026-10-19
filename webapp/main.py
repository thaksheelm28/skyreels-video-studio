"""
Entry point: starts FastAPI (uvicorn) + Gradio simultaneously.

Usage:
    python -m webapp.main
    # or
    python webapp/main.py
"""
import subprocess
import sys
import time
import threading


def run_api():
    """Launch FastAPI backend with uvicorn."""
    import uvicorn
    from webapp.app.api import app
    from webapp.app.config import config
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        log_level="info",
    )


def run_ui():
    """Launch Gradio frontend."""
    # Wait a moment for the API to start first
    time.sleep(3)
    from webapp.app.ui import build_ui
    from webapp.app.config import config
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=config.gradio_port,
        share=False,
        show_api=True,
        quiet=False,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("  🎬 SkyReels V2 — AI Video Generation Studio")
    print("=" * 60)
    print(f"  API  → http://localhost:8000")
    print(f"  UI   → http://localhost:7860")
    print(f"  Docs → http://localhost:8000/docs")
    print("=" * 60)

    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    run_ui()  # Blocks on main thread
