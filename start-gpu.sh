#!/bin/bash
# Start Qwen3-TTS server in GPU (CUDA) mode.
# Usage: ./start-gpu.sh [port] {start|stop|restart|status|logs}
#
# Environment variables:
#   GGML_BACKEND  CUDA device to use (default: CUDA0)
#                  Use CUDA1 for second GPU, etc.
#   CUDA_LIB_PATH Path to CUDA libraries (default: auto-detect)
set -e
cd "$(dirname "$0")"

PORT=${1:-8871}
ACTION=${2:-start}
PIDFILE="server.pid"
LOGFILE="server.log"
BACKEND=${GGML_BACKEND:-CUDA0}

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

    if [ ! -f "build-gpu/tts-server" ]; then
      echo "ERROR: GPU binary not found at build-gpu/tts-server"
      echo "Run scripts/build-gpu.sh first."
      exit 1
    fi

    if ! command -v nvidia-smi &>/dev/null; then
      echo "ERROR: nvidia-smi not found. NVIDIA driver required for GPU mode."
      exit 1
    fi

    echo "Starting Qwen3-TTS (GPU mode, $BACKEND) on port $PORT..."
    export GGML_BACKEND="$BACKEND"
    if [ -n "$CUDA_LIB_PATH" ]; then
      export LD_LIBRARY_PATH="$CUDA_LIB_PATH:${LD_LIBRARY_PATH:-}"
    fi
    PORT=$PORT nohup .venv/bin/python server.py > "$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
    sleep 3
    if kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
      echo "Server started (PID $(cat $PIDFILE))"
      echo "Web UI: http://localhost:$PORT"
      echo "Backend: GPU ($BACKEND)"
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
      echo "Running (PID $(cat $PIDFILE), port $PORT, backend: GPU ($BACKEND))"
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
    echo "Backend: GPU (build-gpu/tts-server, CUDA)"
    echo ""
    echo "Environment:"
    echo "  GGML_BACKEND  CUDA device (default: CUDA0, use CUDA1 for 2nd GPU)"
    exit 1
    ;;
esac