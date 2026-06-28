#!/bin/bash
# Build tts-server for CPU (no GPU required).
# Uses OpenBLAS for matrix acceleration if available.
set -e
cd "$(dirname "$0")/.."

echo "Building tts-server (CPU mode)..."

# Check dependencies
if ! command -v cmake &>/dev/null; then
  echo "ERROR: cmake not found. Install it first:"
  echo "  Ubuntu/Debian: sudo apt install cmake"
  echo "  Arch:          sudo pacman -S cmake"
  echo "  macOS:         brew install cmake"
  exit 1
fi
if ! command -v g++ &>/dev/null && ! command -v c++ &>/dev/null; then
  echo "ERROR: C++ compiler not found. Install g++ or clang."
  exit 1
fi

# Check for ggml submodule
if [ ! -f ggml/CMakeLists.txt ]; then
  echo "Initializing ggml submodule..."
  git submodule update --init --recursive
fi

rm -rf build
mkdir build
cd build

# Try with BLAS first, fall back without
if pkg-config --exists openblas 2>/dev/null; then
  echo "OpenBLAS detected, enabling BLAS acceleration..."
  cmake .. -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
else
  echo "No BLAS found, building without (still works, just slower)..."
  cmake .. -DGGML_BLAS=OFF
fi

cmake --build . --config Release -j "$(nproc)"

echo ""
echo "Build complete! Binary: build/tts-server"
echo "Start the server with: ./start-cpu.sh start"