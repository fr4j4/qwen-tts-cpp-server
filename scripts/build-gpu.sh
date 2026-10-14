#!/bin/bash
# Build tts-server with CUDA (NVIDIA GPU).
# Automatically detects your GPU architecture(s) to minimize compile time.
#
# Usage:
#   ./scripts/build-gpu.sh              # auto-detect GPU architectures
#   ./scripts/build-gpu.sh 86           # force sm_86 only (RTX 30/40 series)
#   ./scripts/build-gpu.sh "75;86"      # build for sm_75 and sm_86
#   ./scripts/build-gpu.sh all          # build for all supported architectures (slower, ~15+ min)
set -e
cd "$(dirname "$0")/.."

echo "Building tts-server (CUDA GPU mode)..."

# Check dependencies
if ! command -v cmake &>/dev/null; then
  echo "ERROR: cmake not found. Install it first."
  exit 1
fi

# Find nvcc
NVCC=""
for path in /opt/cuda/bin/nvcc /usr/local/cuda/bin/nvcc $(command -v nvcc 2>/dev/null); do
  if [ -x "$path" ]; then
    NVCC="$path"
    break
  fi
done
if [ -z "$NVCC" ]; then
  echo "ERROR: CUDA toolkit (nvcc) not found."
  echo "Install CUDA toolkit first:"
  echo "  Ubuntu:  sudo apt install nvidia-cuda-toolkit"
  echo "  Arch:    sudo pacman -S cuda"
  echo "  Or download from: https://developer.nvidia.com/cuda-toolkit"
  exit 1
fi

echo "Using nvcc: $NVCC"
CUDA_DIR=$(dirname "$(dirname "$NVCC")")

# Detect GPU architectures if not provided
if [ "$1" = "all" ]; then
  ARCHS="75;80;86;89;120a;121a"
  echo "Building for ALL supported architectures: sm_${ARCHS//;/, sm_}"
elif [ -z "$1" ]; then
  echo "Detecting GPU architectures..."
  if command -v nvidia-smi &>/dev/null; then
    ARCHS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | \
      sort -u | \
      while read cc; do
        # Convert compute capability (e.g. "8.6") to sm_arch (e.g. "86")
        arch=$(echo "$cc" | tr -d '.')
        echo "$arch"
      done | paste -sd ';')
    if [ -n "$ARCHS" ]; then
      echo "Detected GPU architectures: sm_${ARCHS//;/, sm_}"
    fi
  fi
  if [ -z "$ARCHS" ]; then
    echo "Could not auto-detect. Using default: 86 (RTX 30/40 series)"
    ARCHS="86"
  fi
else
  ARCHS="$1"
  echo "Using specified architectures: sm_${ARCHS//;/, sm_}"
fi

# ggml is vendored in-tree (updated to v0.24.0); no submodule init needed.

# Host compiler for nvcc. Each CUDA release caps the supported gcc major
# (12.4-12.7: <= 13, 12.8-12.9: <= 14, 13.x: <= 15) ; a newer default g++
# (e.g. gcc 15 on Ubuntu 25.10) makes nvcc abort with "unsupported GNU
# version". Order: $CUDAHOSTCXX, then g++ if supported, then the newest
# supported g++-N on PATH, else let nvcc use its own configured default.
NVCC_VER=$("$NVCC" --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')
read -r NVCC_MAJ NVCC_MIN <<< "${NVCC_VER:-0 0}"
if [ "$NVCC_MAJ" -ge 13 ]; then MAX_GCC=15
elif [ "$NVCC_MAJ" -eq 12 ] && [ "$NVCC_MIN" -ge 8 ]; then MAX_GCC=14
elif [ "$NVCC_MAJ" -eq 12 ] && [ "$NVCC_MIN" -ge 4 ]; then MAX_GCC=13
else MAX_GCC=12
fi
HOST_CXX="${CUDAHOSTCXX:-}"
if [ -z "$HOST_CXX" ]; then
  GXX_MAJ=$(g++ -dumpversion 2>/dev/null | cut -d. -f1)
  if [ -n "$GXX_MAJ" ] && [ "$GXX_MAJ" -le "$MAX_GCC" ]; then
    HOST_CXX=$(command -v g++)
  else
    for v in $(seq "$MAX_GCC" -1 9); do
      if command -v "g++-$v" &>/dev/null; then HOST_CXX=$(command -v "g++-$v"); break; fi
    done
  fi
fi
HOST_FLAG=()
if [ -n "$HOST_CXX" ]; then
  echo "nvcc $NVCC_MAJ.$NVCC_MIN (gcc <= $MAX_GCC): host compiler $HOST_CXX"
  HOST_FLAG=(-DCMAKE_CUDA_HOST_COMPILER="$HOST_CXX")
else
  echo "WARN: no g++ <= $MAX_GCC found; letting nvcc pick its default host compiler"
fi

rm -rf build-gpu
mkdir build-gpu
cd build-gpu

# GGML_CUDA_GRAPHS=ON: automatic CUDA graph capture in ggml. Measured
# -5% RTF on the TTS pipeline with zero code changes (see perf/talker
# graphs branch history). Bit-exact vs the graphs=OFF build.
cmake .. \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" \
  "${HOST_FLAG[@]}"

cmake --build . --config Release -j "$(nproc)"

echo ""
echo "Build complete! Binary: build-gpu/tts-server"
echo "Start the server with: ./start-gpu.sh start"