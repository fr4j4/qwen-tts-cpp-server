# qwen-tts-cpp-server

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Platform: Linux](https://img.shields.io/badge/Platform-Linux-yellow.svg)](https://www.linux.org/)
[![Backend: GGML](https://img.shields.io/badge/Backend-GGML_C++-orange.svg)](https://github.com/ggerganov/ggml)
[![Models: GGUF](https://img.shields.io/badge/Models-GGUF_Q4_K_M-green.svg)](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
[![Languages: 11](https://img.shields.io/badge/Languages-11-purple.svg)](#)
[![GPU: CUDA](https://img.shields.io/badge/GPU-CUDA_Vulkan_CPU-brightgreen.svg)](#)

Local TTS server powered by **Qwen3-TTS** (Alibaba / Qwen team) with native **C++/GGML** inference — no Python or PyTorch required for the core engine. Runs on CPU or NVIDIA GPU (CUDA) using quantized GGUF models. Supports 11 languages, 9 voices, ES+EN code-switching, and PCM streaming.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Requirements](#requirements)
- [Building](#building)
  - [CPU](#cpu-no-gpu-required)
  - [GPU (NVIDIA CUDA)](#gpu-nvidia-cuda)
  - [Vulkan](#vulkan-amd--intel--nvidia)
- [Downloading Models](#downloading-models)
- [Running the Server](#running-the-server)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Standalone C++ Server](#standalone-c-server)
- [Streaming Notes](#streaming-notes)
- [Voice Cloning](#voice-cloning)
- [Project Structure](#project-structure)
- [Tested Models](#tested-models)
- [Credits](#credits)
- [License](#license)

## Features

- **Native C++ inference** via GGML — no PyTorch, no 3GB model downloads
- **CPU and GPU** support (CUDA, Vulkan, Metal)
- **Quantized GGUF models** (Q4_K_M to F32) — smallest is 884 MB total
- **11 languages** with automatic code-switching (ES+EN in same sentence)
- **9 named voices** (Serena, Vivian, Aiden, Dylan, Eric, Ryan, Sohee, Ono Anna, Uncle Fu)
- **Voice cloning** from reference audio clip (CustomVoice models)
- **PCM streaming** via HTTP chunked transfer
- **REST API** + bilingual web UI (English/Spanish)
- **Apache 2.0** licensed

## Quick Start

```bash
# 1. Clone with submodules
git clone --recurse-submodules https://github.com/fr4j4/qwen-tts-cpp-server.git
cd qwen-tts-cpp-server

# 2. Download models (default: 0.6B CustomVoice Q4_K_M, 884 MB total)
./scripts/download-models.sh

# 3. Build (CPU)
./scripts/build-cpu.sh

# 4. Create Python venv and install dependencies
python3 -m venv .venv
.venv/bin/pip install fastapi uvicorn httpx pydantic

# 5. Start the server
./start-cpu.sh start

# 6. Open the web UI
#    http://localhost:8871
```

## Requirements

### Build dependencies

| Dependency | Required? | Install |
|---|---|---|
| **cmake** >= 3.14 | Yes | `apt install cmake` / `pacman -S cmake` |
| **g++** (C++17) | Yes | `apt install g++` / `pacman -S gcc` |
| **git** | Yes | `apt install git` / `pacman -S git` |
| **CUDA toolkit** (nvcc) | GPU only | `apt install nvidia-cuda-toolkit` / `pacman -S cuda` |
| **OpenBLAS** | Optional (CPU accel) | `apt install libopenblas-dev` / `pacman -S openblas` |
| **Vulkan SDK** | Vulkan only | See [vulkan.lunarg.com](https://vulkan.lunarg.com/) |

### Runtime dependencies

| Dependency | Required? | Install |
|---|---|---|
| **Python 3.10+** | Yes (wrapper only) | System package |
| **ffmpeg** | Yes (MP3 output) | `apt install ffmpeg` / `pacman -S ffmpeg` |
| **NVIDIA driver** | GPU only | System package |
| **pip packages** | Yes (wrapper) | `fastapi`, `uvicorn`, `httpx`, `pydantic` |

> **Note:** The C++ server (`tts-server`) runs standalone without Python. The Python wrapper adds the web UI, REST API, and MP3 conversion. See [Standalone C++](#standalone-c-server) below.

> **Windows / WSL2:** This project has only been tested on Linux. It may work on Windows via WSL2 or native compilation with MSVC/MinGW, but this has not been verified. Feel free to try it and open an issue with your results — we'd love to hear from you.

## Building

### CPU (no GPU required)

```bash
./scripts/build-cpu.sh
```

Output: `build/tts-server`

### GPU (NVIDIA CUDA)

```bash
# Auto-detects your GPU architecture
./scripts/build-gpu.sh

# Or specify architecture manually (e.g. 86 for RTX 30/40 series)
./scripts/build-gpu.sh 86

# Multiple architectures (e.g. 75 for RTX 20xx, 86 for RTX 30xx)
./scripts/build-gpu.sh "75;86"

# All supported architectures (for distributing binaries, ~15+ min)
./scripts/build-gpu.sh all
```

Output: `build-gpu/tts-server`

> **Build time:** ~5 min for one architecture, ~15 min for all default architectures. The script auto-detects your GPU's compute capability to minimize compile time.

<details>
<summary><b>GPU architecture reference</b> (click to expand)</summary>

| Code | Architecture name | GPU series | Examples |
|------|-------------------|------------|----------|
| `75` | Turing | RTX 20xx / GTX 16xx | RTX 2070, RTX 2080 Ti, GTX 1660 |
| `80` | Ampere | A100 (data center) | A100, A30 |
| `86` | Ampere | RTX 30xx / A40 / A6000 | RTX 3060, RTX 3090, RTX 3080 |
| `89` | Ada Lovelace | RTX 40xx / L40 | RTX 4060, RTX 4090, RTX 4080 |
| `120a` | Blackwell | RTX 50xx | RTX 5070, RTX 5080, RTX 5090 |
| `121a` | Blackwell | B100 / B200 (data center) | B100, B200 |

**How to find your code:**

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Output e.g. "8.6" → remove the dot → use "86"
```

Or just run `./scripts/build-gpu.sh` without arguments — it auto-detects for you.

</details>

### Vulkan (AMD / Intel / NVIDIA)

```bash
./scripts/build-vulkan.sh
```

Output: `build-vulkan/tts-server`

## Downloading Models

Models are NOT included in the repo. Download them separately:

```bash
# Default: 0.6B CustomVoice Q4_K_M (884 MB, recommended for most users)
./scripts/download-models.sh

# Specific model + quantization
./scripts/download-models.sh 0.6b-customvoice Q8_0

# Or download individual models:
./scripts/download-0.6b-base.sh           # 629 MB — no voice cloning
./scripts/download-0.6b-customvoice.sh    # 605 MB — with voice cloning (recommended)
./scripts/download-1.7b-base.sh           # 1.2 GB — higher quality, no cloning
./scripts/download-1.7b-customvoice.sh    # 1.2 GB — higher quality + cloning
./scripts/download-1.7b-voicedesign.sh    # 1.2 GB — voice from text description
./scripts/download-tokenizer.sh           # Required for all models
```

### Available models

| Model | Size (Q4_K_M) | Voice cloning | Named speakers | Use case |
|---|---|---|---|---|
| 0.6b-base | 629 MB | No | Yes | Smallest, CPU-friendly |
| 0.6b-customvoice | 605 MB | Yes | Yes (9) | **Recommended** — best balance |
| 1.7b-base | 1.2 GB | No | Yes | Higher quality |
| 1.7b-customvoice | 1.2 GB | Yes | Yes (9) | Higher quality + cloning |
| 1.7b-voicedesign | 1.2 GB | From text | No | Voice from attributes |

### Quantization variants

| Variant | Talker 0.6B | Tokenizer | Quality | Use case |
|---|---|---|---|---|
| **Q4_K_M** | 629 MB | 255 MB | Good | Lowest VRAM, CPU |
| **Q8_0** | 993 MB | 291 MB | Very good | **Recommended default** |
| BF16 | 1.8 GB | 359 MB | Excellent | Max precision |
| F32 | 3.7 GB | 647 MB | Perfect | Debug / reference only |

> The tokenizer (codec) GGUF is shared across all talker models. You only need one tokenizer regardless of which talker you use.

## Running the Server

### CPU mode

```bash
./start-cpu.sh start       # Start (default port 8871)
./start-cpu.sh 9000 start  # Custom port
./start-cpu.sh stop        # Stop
./start-cpu.sh restart     # Restart
./start-cpu.sh status      # Check status
./start-cpu.sh logs        # View logs
```

### GPU mode

```bash
./start-gpu.sh start       # Start with GPU 0 (CUDA0)
GGML_BACKEND=CUDA1 ./start-gpu.sh start  # Use second GPU
./start-gpu.sh stop        # Stop
./start-gpu.sh status      # Check status
```

### Ports

| Port | Service |
|---|---|
| **8871** | Web UI + REST API (Python wrapper) |
| **8870** | C++ TTS server (internal, managed by wrapper) |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GGML_BACKEND` | `CPU` | Backend for inference: `CPU`, `CUDA0`, `CUDA1`, `VULKAN0` |
| `CUDA_VISIBLE_DEVICES` | (all) | Restrict which NVIDIA GPUs are visible (by UUID or index) |
| `TTS_HOST` | `127.0.0.1` | C++ server bind address |
| `TTS_PORT` | `8870` | C++ server port (internal) |
| `WRAPPER_PORT` | `8871` | Python wrapper / web UI port |

**Examples:**

```bash
# Use second GPU
GGML_BACKEND=CUDA1 ./start-gpu.sh start

# Restrict to specific GPU by UUID (useful for multi-GPU systems)
CUDA_VISIBLE_DEVICES=GPU-3d1bda3d-12da-cd16-0404-98de27b7d8d1 ./start-gpu.sh start

# Bind to all interfaces (LAN access)
TTS_HOST=0.0.0.0 ./start-cpu.sh start

# Custom wrapper port
WRAPPER_PORT=9000 ./start-cpu.sh start
```

## API Reference

### `POST /api/tts`

Synthesize speech from text.

```json
{
  "input": "Hello world, este es un test de code-switching.",
  "voice": "serena",
  "language": "auto",
  "format": "mp3",
  "stream": false
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` | string | (required) | Text to synthesize |
| `voice` | string | `serena` | Voice name (see below) |
| `language` | string | `auto` | Language for pronunciation |
| `format` | string | `mp3` | Output format: `mp3` or `wav` (non-streaming only) |
| `stream` | boolean | `false` | PCM streaming mode |
| `speed` | float | `1.0` | Speech speed multiplier |

**Voices:** `serena`, `vivian`, `ono_anna`, `aiden`, `dylan`, `eric`, `ryan`, `sohee`, `uncle_fu`

**Languages:** `auto` (code-switching), `spanish`, `english`, `chinese`, `french`, `german`, `italian`, `japanese`, `korean`, `portuguese`, `russian`

**Response:**
- Non-streaming: audio file (`audio/mpeg` for MP3, `audio/wav` for WAV)
- Streaming: chunked `audio/pcm` (raw 24kHz 16-bit signed LE)

### `GET /api/health`

```json
{"status": "ok", "cpp_server": "http://127.0.0.1:8870", "backend": "GPU (CUDA)"}
```

### `GET /api/speakers`

```json
{"speakers": ["aiden", "dylan", ...], "languages": ["auto", "spanish", ...], "backend": "CPU"}
```

## Standalone C++ Server

The C++ binary can run without the Python wrapper:

```bash
./build/tts-server \
  --model models/qwen-talker-0.6b-customvoice-Q4_K_M.gguf \
  --codec models/qwen-tokenizer-12hz-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8870

# GPU mode
GGML_BACKEND=CUDA0 ./build-gpu/tts-server \
  --model models/qwen-talker-0.6b-customvoice-Q4_K_M.gguf \
  --codec models/qwen-tokenizer-12hz-Q4_K_M.gguf
```

### C++ API (OpenAI-compatible)

```bash
curl http://localhost:8870/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "serena", "language": "auto", "response_format": "wav"}' \
  --output speech.wav
```

## Streaming Notes

The C++ server supports HTTP chunked transfer for PCM audio. However, the full audio generation completes before chunks are sent over the wire. This means:

- Chunks arrive together at the end of synthesis (not progressively during generation)
- Still beneficial: no temporary files, lower memory usage, client can start playback as bytes arrive
- For true progressive streaming, the HTTP library (`cpp-httplib`) would need explicit flush support after each codec chunk

## Voice Cloning

Voice cloning is inherited from the upstream [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp) project and the CustomVoice GGUF models. The C++ server exposes endpoints for cloning a voice from a reference audio clip.

> **Note:** This feature has **not been tested** by the maintainer of this repo. It should work as-is since it's part of the upstream C++ server, but no verification has been done. If you test it, please open an issue with your results.

For usage examples, see the upstream project's `examples/` directory (`clone.sh`, `clone.cmd`).

## Project Structure

```
qwen-tts-cpp-server/
├── src/                    # C++ source (tts-server, pipeline, codec)
├── ggml/                   # GGML submodule (inference backend)
├── vendor/                 # cpp-httplib, yyjson
├── tools/                  # Build tools (version embedding)
├── tests/                  # Test files
├── examples/               # CLI examples (base, clone, customvoice, server)
├── docs/                   # Architecture documentation
├── scripts/                # Build + model download scripts
│   ├── build-cpu.sh        # Build for CPU
│   ├── build-gpu.sh        # Build for CUDA GPU
│   ├── build-vulkan.sh     # Build for Vulkan GPU
│   ├── download-models.sh  # Download talker + tokenizer
│   ├── download-0.6b-base.sh
│   ├── download-0.6b-customvoice.sh
│   ├── download-1.7b-base.sh
│   ├── download-1.7b-customvoice.sh
│   ├── download-1.7b-voicedesign.sh
│   └── download-tokenizer.sh
├── static/                 # Web UI (bilingual ES/EN)
├── server.py               # Python FastAPI wrapper
├── start-cpu.sh            # Start script (CPU mode)
├── start-gpu.sh            # Start script (GPU mode)
├── CMakeLists.txt          # Build configuration
├── convert.py              # Model conversion script (PyTorch -> GGUF)
├── quantize.sh             # Model quantization script
├── LICENSE                 # Apache 2.0
└── README.md               # This file
```

## Tested Models

This project has been tested with the **0.6B CustomVoice** model (Q4_K_M and Q8_0 quantizations) on both CPU and NVIDIA GPU (CUDA). The 1.7B variants and other modes (base, voicedesign) should work without changes — the C++ server loads any valid GGUF pair — but are not yet verified by the maintainer. If you test them, please open an issue with your results.

## Credits

- **Qwen3-TTS** — Alibaba / Qwen team (Apache 2.0)
- **qwentts.cpp** — [ServeurpersoCom/qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp) (MIT)
- **GGML** — [ServeurpersoCom/ggml](https://github.com/ServeurpersoCom/ggml) (MIT)
- **GGUF models** — [Serveurperso/Qwen3-TTS-GGUF](https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF) (Apache 2.0)
- **Web UI + Python wrapper** — Francisco Gonzalez (fr4j4)

## License

Apache 2.0 — see [LICENSE](LICENSE)

The upstream C++ code (qwentts.cpp, GGML) is MIT licensed. The Qwen3-TTS model is Apache 2.0. This project is Apache 2.0.