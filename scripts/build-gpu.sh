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
fi

# Check for ggml submodule
if [ ! -f ggml/CMakeLists.txt ]; then
  echo "Initializing ggml submodule..."
  git submodule update --init --recursive
fi

rm -rf build-gpu
mkdir build-gpu
cd build-gpu

cmake .. \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" \
  -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++ 2>/dev/null || command -v c++)"

cmake --build . --config Release -j "$(nproc)"

echo ""
echo "Build complete! Binary: build-gpu/tts-server"
echo "Start the server with: ./start-gpu.sh start"