#!/bin/bash
# Docker build helper for qwen-tts-cpp-server.
# Supports split base image for faster GPU rebuilds.
#
# Usage:
#   ./docker/build.sh base       # Build base image (CUDA + deps + venv) — run once
#   ./docker/build.sh gpu        # Build GPU image from cached base + run
#   ./docker/build.sh cpu        # Build CPU image (standalone, no base needed) + run
#   ./docker/build.sh clean      # Remove app images + container (keeps base)
#   ./docker/build.sh clean-all  # Remove everything including base image
#
# GPU requires nvidia-container-toolkit configured:
#   sudo nvidia-ctk runtime configure --runtime docker
#   sudo systemctl restart docker

set -e

IMAGE_NAME="qwen-tts"
BASE_NAME="qwen-tts-base"
CONTAINER_NAME="qwen-tts"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cmd="${1:-gpu}"

case "$cmd" in
  base)
    echo "=== Building base image (CUDA + deps + venv) ==="
    echo "This is done once. Cached for all subsequent builds."
    docker build -f "$SCRIPT_DIR/Dockerfile.base" -t "$BASE_NAME:cuda12.6" "$SCRIPT_DIR"
    echo "=== Base image ready: $BASE_NAME:cuda12.6 ==="
    echo "Now run: ./docker/build.sh gpu  (or ./docker/build.sh cpu)"
    ;;

  cpu)
    echo "=== Building CPU Docker image (standalone) ==="
    docker build -f "$SCRIPT_DIR/Dockerfile.cpu" -t "$IMAGE_NAME:cpu" "$SCRIPT_DIR"

    echo "=== Starting CPU container ==="
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    docker run -d --name "$CONTAINER_NAME" -p 8871:8871 "$IMAGE_NAME:cpu"

    echo "=== Waiting for server to start (60s) ==="
    sleep 60

    echo "=== Health check ==="
    curl -s http://localhost:8871/api/health | python3 -m json.tool

    echo "=== TTS test (MP3) ==="
    curl -s -X POST http://localhost:8871/api/tts \
      -H "Content-Type: application/json" \
      -d '{"input": "Hola mundo, this is a code-switching test.", "voice": "serena", "language": "auto", "format": "mp3"}' \
      --output /tmp/qwen-tts-test.mp3
    file /tmp/qwen-tts-test.mp3
    ls -lh /tmp/qwen-tts-test.mp3

    echo "=== Speakers check ==="
    curl -s http://localhost:8871/api/speakers | python3 -m json.tool

    echo "=== Done. Container logs: ==="
    docker logs "$CONTAINER_NAME" --tail 20
    echo "=== To stop: docker rm -f $CONTAINER_NAME ==="
    ;;

  gpu)
    echo "=== Building GPU Docker image (from cached base) ==="
    docker build -f "$SCRIPT_DIR/Dockerfile.gpu" \
      --build-arg BASE_IMAGE="$BASE_NAME:cuda12.6" \
      -t "$IMAGE_NAME:gpu" "$SCRIPT_DIR"

    echo "=== Starting GPU container ==="
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    docker run -d --name "$CONTAINER_NAME" \
      --gpus all \
      -p 8871:8871 \
      -e GGML_BACKEND=CUDA0 \
      "$IMAGE_NAME:gpu"

    echo "=== Waiting for server to start (120s — GPU model loading is slower) ==="
    sleep 120

    echo "=== Health check ==="
    curl -s http://localhost:8871/api/health | python3 -m json.tool

    echo "=== TTS test (MP3) ==="
    curl -s -X POST http://localhost:8871/api/tts \
      -H "Content-Type: application/json" \
      -d '{"input": "Hola mundo, this is a code-switching test.", "voice": "serena", "language": "auto", "format": "mp3"}' \
      --output /tmp/qwen-tts-test.mp3
    file /tmp/qwen-tts-test.mp3
    ls -lh /tmp/qwen-tts-test.mp3

    echo "=== Speakers check ==="
    curl -s http://localhost:8871/api/speakers | python3 -m json.tool

    echo "=== Container logs (last 30 lines) ==="
    docker logs "$CONTAINER_NAME" --tail 30
    echo "=== To stop: docker rm -f $CONTAINER_NAME ==="
    ;;

  clean)
    echo "=== Cleaning up ==="
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    docker rmi "$IMAGE_NAME:cpu" "$IMAGE_NAME:gpu" 2>/dev/null || true
    rm -f /tmp/qwen-tts-test.mp3
    echo "Done."
    ;;

  clean-all)
    echo "=== Deep clean (includes base image) ==="
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    docker rmi "$IMAGE_NAME:cpu" "$IMAGE_NAME:gpu" "$BASE_NAME:cuda12.6" 2>/dev/null || true
    rm -f /tmp/qwen-tts-test.mp3
    echo "Done."
    ;;

  *)
    echo "Usage: $0 {base|cpu|gpu|clean|clean-all}"
    echo ""
    echo "  base       Build base image with CUDA + deps (run once)"
    echo "  cpu        Build and run CPU mode (standalone, no base needed)"
    echo "  gpu        Build and run GPU mode (uses cached base)"
    echo "  clean      Remove app images + container (keeps base)"
    echo "  clean-all  Remove everything including base image"
    exit 1
    ;;
esac