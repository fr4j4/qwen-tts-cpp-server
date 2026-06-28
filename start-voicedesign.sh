#!/bin/bash
# Start Qwen3-TTS server with VoiceDesign model (1.7B).
# VoiceDesign allows synthesizing voices from text descriptions via the "instructions" parameter.
#
# Usage: ./start-voicedesign.sh [port] {start|stop|restart|status|logs}
#
# Environment variables:
#   GGML_BACKEND  CUDA device to use (default: CPU)
#                  Set to CUDA0 or CUDA1 for GPU mode
#   CUDA_LIB_PATH Path to CUDA libraries (default: auto-detect)
set -e
cd "$(dirname "$0")"

PORT=${1:-8871}
ACTION=${2:-start}
PIDFILE="server.pid"
LOGFILE="server.log"
BACKEND=${GGML_BACKEND:-cpu}
USE_GPU=false
[[ "$BACKEND" =~ ^CUDA ]] && USE_GPU=true

# VoiceDesign model paths
TALKER_MODEL="models/qwen-talker-1.7b-voicedesign-Q4_K_M.gguf"
CODEC_MODEL="models/qwen-tokenizer-12hz-Q4_K_M.gguf"

# Auto-detect CUDA library path
if [ -z "$CUDA_LIB_PATH" ]; then
  for path in /opt/cuda/lib64 /usr/local/cuda/lib64 /usr/lib/cuda/lib64; do
    if [ -d "$path" ]; then
      CUDA_LIB_PATH="$path"
      break
    fi
  done
fi

case "$ACTION" in
  start)
    # Kill any existing instance
    if [ -f "$PIDFILE" ]; then
      OLD_PID=$(cat "$PIDFILE")
      if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping existing server (PID $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
      fi
    fi
    lsof -ti :8870 2>/dev/null | xargs kill 2>/dev/null || true
    lsof -ti :$PORT 2>/dev/null | xargs kill 2>/dev/null || true
    sleep 1

    # Check models exist
    if [ ! -f "$TALKER_MODEL" ]; then
      echo "ERROR: VoiceDesign talker model not found: $TALKER_MODEL"
      echo "Run: ./scripts/download-voicedesign.sh"
      exit 1
    fi
    if [ ! -f "$CODEC_MODEL" ]; then
      echo "ERROR: Codec model not found: $CODEC_MODEL"
      echo "Run: ./scripts/download-voicedesign.sh"
      exit 1
    fi

    # Check binary
    if $USE_GPU; then
      BUILD_DIR="build-gpu"
      if [ ! -f "$BUILD_DIR/tts-server" ]; then
        echo "ERROR: GPU binary not found at $BUILD_DIR/tts-server"
        echo "Run: scripts/build-gpu.sh"
        exit 1
      fi
      if ! command -v nvidia-smi &>/dev/null; then
        echo "ERROR: nvidia-smi not found. NVIDIA driver required for GPU mode."
        exit 1
      fi
    else
      BUILD_DIR="build"
      if [ ! -f "$BUILD_DIR/tts-server" ]; then
        echo "ERROR: CPU binary not found at $BUILD_DIR/tts-server"
        echo "Run: scripts/build-cpu.sh"
        exit 1
      fi
    fi

    BACKEND_LABEL="GPU ($BACKEND)" if $USE_GPU || BACKEND_LABEL="CPU"

    echo "Starting Qwen3-TTS VoiceDesign ($BACKEND_LABEL) on port $PORT..."
    echo "  Talker: $TALKER_MODEL"
    echo "  Codec:  $CODEC_MODEL"
    echo "  VoiceDesign: enabled (use instructions parameter in API/UI)"

    export GGML_BACKEND="$BACKEND"
    export QWEN_TTS_TALKER_MODEL="$TALKER_MODEL"
    export QWEN_TTS_CODEC_MODEL="$CODEC_MODEL"
    export QWEN_TTS_VOICE_DESIGN="true"
    if [ -n "$CUDA_LIB_PATH" ] && $USE_GPU; then
      export LD_LIBRARY_PATH="$CUDA_LIB_PATH:${LD_LIBRARY_PATH:-}"
    fi
    PORT=$PORT nohup .venv/bin/python server.py > "$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
    sleep 3
    if kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
      echo "Server started (PID $(cat $PIDFILE))"
      echo "Web UI: http://localhost:$PORT"
      echo "Backend: $BACKEND_LABEL | Model: VoiceDesign 1.7B"
    else
      echo "ERROR: Server failed to start. Check $LOGFILE"
      tail -10 "$LOGFILE"
      exit 1
    fi
    ;;

  stop)
    if [ -f "$PIDFILE" ]; then
      PID=$(cat "$PIDFILE")
      kill "$PID" 2>/dev/null || true
      sleep 2
      rm -f "$PIDFILE"
      echo "Server stopped"
    else
      lsof -ti :8870 2>/dev/null | xargs kill 2>/dev/null || true
      lsof -ti :$PORT 2>/dev/null | xargs kill 2>/dev/null || true
      echo "Server stopped (by port)"
    fi
    ;;

  restart)
    $0 "$PORT" stop
    sleep 1
    $0 "$PORT" start
    ;;

  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
      echo "Running (PID $(cat $PIDFILE), port $PORT, backend: $BACKEND, model: VoiceDesign 1.7B)"
    else
      echo "Not running"
    fi
    ;;

  logs)
    tail -f "$LOGFILE"
    ;;

  *)
    echo "Usage: $0 [port] {start|stop|restart|status|logs}"
    echo "Default port: 8871, default action: start"
    echo "Model: VoiceDesign 1.7B (requires instructions parameter)"
    echo ""
    echo "Environment:"
    echo "  GGML_BACKEND  CUDA device (default: cpu, use CUDA0/CUDA1 for GPU)"
    echo ""
    echo "First time setup:"
    echo "  1. ./scripts/download-voicedesign.sh"
    echo "  2. scripts/build-cpu.sh  (or scripts/build-gpu.sh for GPU)"
    echo "  3. python3 -m venv .venv && .venv/bin/pip install fastapi uvicorn httpx pydantic"
    exit 1
    ;;
esac