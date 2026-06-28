#!/bin/bash
# Start Qwen3-TTS server in CPU mode.
# Usage: ./start-cpu.sh [port] {start|stop|restart|status|logs}
set -e
cd "$(dirname "$0")"

PORT=${1:-8871}
ACTION=${2:-start}
PIDFILE="server.pid"
LOGFILE="server.log"

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

    if [ ! -f "build/tts-server" ]; then
      echo "ERROR: CPU binary not found at build/tts-server"
      echo "Run scripts/build-cpu.sh first."
      exit 1
    fi

    echo "Starting Qwen3-TTS (CPU mode) on port $PORT..."
    GGML_BACKEND=cpu PORT=$PORT nohup .venv/bin/python server.py > "$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
    sleep 3
    if kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
      echo "Server started (PID $(cat $PIDFILE))"
      echo "Web UI: http://localhost:$PORT"
      echo "Backend: CPU"
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
      echo "Running (PID $(cat $PIDFILE), port $PORT, backend: CPU)"
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
    echo "Backend: CPU (build/tts-server)"
    exit 1
    ;;
esac