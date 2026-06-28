#!/usr/bin/env python3
"""
Qwen3-TTS GGUF POC — FastAPI wrapper around qwentts.cpp binary.
Serves a web UI and proxies TTS requests to the C++ server.
Supports streaming (PCM passthrough) and non-streaming (WAV/MP3) modes.
"""
import io
import os
import subprocess
import logging
import signal
import sys
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

# Binary selection: use GPU build if GGML_BACKEND starts with CUDA, else CPU build
USE_GPU = os.environ.get("GGML_BACKEND", "cpu").upper().startswith("CUDA")
BUILD_DIR = "build-gpu" if USE_GPU else "build"
TTS_BINARY = BASE_DIR / BUILD_DIR / "tts-server"
TALKER_MODEL = BASE_DIR / "models" / "qwen-talker-0.6b-customvoice-Q4_K_M.gguf"
CODEC_MODEL = BASE_DIR / "models" / "qwen-tokenizer-12hz-Q4_K_M.gguf"
STATIC_DIR = BASE_DIR / "static"

CPP_SERVER_HOST = "127.0.0.1"
CPP_SERVER_PORT = 8870
CPP_SERVER_URL = f"http://{CPP_SERVER_HOST}:{CPP_SERVER_PORT}"

SPEAKERS = ["aiden", "dylan", "eric", "ryan", "serena", "sohee", "uncle_fu", "ono_anna", "vivian"]
LANGUAGES = ["auto", "spanish", "english", "chinese", "french", "german", "italian", "japanese", "korean", "portuguese", "russian"]

BACKEND_LABEL = "GPU (CUDA)" if USE_GPU else "CPU"

app = FastAPI(title="Qwen3-TTS GGUF POC")

# --------------------------------------------------------------------------- #
#  C++ server lifecycle
# --------------------------------------------------------------------------- #
_cpp_proc: Optional[subprocess.Popen] = None


def start_cpp_server():
    """Start the qwentts.cpp server in background."""
    global _cpp_proc
    if _cpp_proc and _cpp_proc.poll() is None:
        logger.info("C++ server already running")
        return

    if not TTS_BINARY.exists():
        raise RuntimeError(f"Binary not found: {TTS_BINARY}. Run buildcpu-noblas.sh or buildgpu.sh first.")
    if not TALKER_MODEL.exists():
        raise RuntimeError(f"Talker model not found: {TALKER_MODEL}")
    if not CODEC_MODEL.exists():
        raise RuntimeError(f"Codec model not found: {CODEC_MODEL}")

    logger.info(f"Starting C++ server ({BACKEND_LABEL}) on port {CPP_SERVER_PORT}...")
    env = os.environ.copy()
    if USE_GPU:
        # Ensure CUDA libraries are findable
        env["LD_LIBRARY_PATH"] = "/opt/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    # Redirect C++ server output to log file (not PIPE — pipe buffer fills
    # and blocks the server during model loading on GPU)
    log_file = open(BASE_DIR / "cpp_server.log", "w")
    _cpp_proc = subprocess.Popen(
        [
            str(TTS_BINARY),
            "--model", str(TALKER_MODEL),
            "--codec", str(CODEC_MODEL),
            "--host", CPP_SERVER_HOST,
            "--port", str(CPP_SERVER_PORT),
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )

    # Wait for health check (GPU mode takes ~40s to load models)
    max_wait = 60 if USE_GPU else 30
    for i in range(max_wait):
        time.sleep(1)
        try:
            r = httpx.get(f"{CPP_SERVER_URL}/health", timeout=2)
            if r.status_code == 200:
                logger.info(f"C++ server ready ({BACKEND_LABEL}, took {i+1}s)")
                return
        except Exception:
            continue
    raise RuntimeError(f"C++ server failed to start within {max_wait}s")


def stop_cpp_server():
    global _cpp_proc
    if _cpp_proc and _cpp_proc.poll() is None:
        _cpp_proc.terminate()
        _cpp_proc.wait(timeout=5)
        logger.info("C++ server stopped")
    _cpp_proc = None


@app.on_event("startup")
async def app_startup():
    start_cpp_server()


@app.on_event("shutdown")
async def app_shutdown():
    stop_cpp_server()


# --------------------------------------------------------------------------- #
#  API
# --------------------------------------------------------------------------- #
class TTSRequest(BaseModel):
    input: str
    voice: str = "serena"
    language: str = "auto"
    speed: float = 1.0
    format: str = "mp3"  # mp3 or wav
    stream: bool = False  # streaming mode (PCM passthrough)


@app.get("/api/health")
async def health():
    return {"status": "ok", "cpp_server": CPP_SERVER_URL, "backend": BACKEND_LABEL}


@app.get("/api/speakers")
async def speakers():
    return {"speakers": SPEAKERS, "languages": LANGUAGES, "backend": BACKEND_LABEL}


@app.post("/api/tts")
async def tts(req: TTSRequest):
    if req.voice not in SPEAKERS:
        raise HTTPException(400, f"Invalid voice. Available: {SPEAKERS}")
    if req.language not in LANGUAGES:
        raise HTTPException(400, f"Invalid language. Available: {LANGUAGES}")

    # --- Streaming mode: PCM passthrough from C++ server ---
    if req.stream:
        payload = {
            "input": req.input,
            "voice": req.voice,
            "language": req.language,
            "response_format": "pcm",
        }
        if req.speed != 1.0:
            payload["speed"] = req.speed

        async def pcm_stream():
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", f"{CPP_SERVER_URL}/v1/audio/speech", json=payload) as r:
                    r.raise_for_status()
                    async for chunk in r.aiter_bytes():
                        yield chunk

        return StreamingResponse(
            pcm_stream(),
            media_type="audio/pcm",
            headers={
                "X-Format": "pcm",
                "X-Stream": "true",
                "X-Sample-Rate": "24000",
                "X-Channels": "1",
                "X-Sample-Format": "s16le",
            },
        )

    # --- Non-streaming mode: WAV from C++ server, optionally convert to MP3 ---
    payload = {
        "input": req.input,
        "voice": req.voice,
        "language": req.language,
        "response_format": "wav",
    }
    if req.speed != 1.0:
        payload["speed"] = req.speed

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{CPP_SERVER_URL}/v1/audio/speech", json=payload)
            r.raise_for_status()
            wav_bytes = r.content
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"C++ server error: {e.response.text[:200]}")
    except Exception as e:
        raise HTTPException(502, f"C++ server unreachable: {str(e)}")

    if req.format == "mp3":
        proc = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", "-f", "mp3", "-ab", "192k", "-y", "pipe:1"],
            input=wav_bytes, capture_output=True,
        )
        if proc.returncode != 0:
            raise HTTPException(500, f"ffmpeg failed: {proc.stderr.decode()[:300]}")
        return StreamingResponse(
            io.BytesIO(proc.stdout),
            media_type="audio/mpeg",
            headers={"X-Format": "mp3"},
        )

    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={"X-Format": "wav"},
    )


# --------------------------------------------------------------------------- #
#  Static files (web UI)
# --------------------------------------------------------------------------- #
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8871))
    logger.info(f"Backend: {BACKEND_LABEL} | Binary: {BUILD_DIR}/tts-server")
    uvicorn.run(app, host="0.0.0.0", port=port)