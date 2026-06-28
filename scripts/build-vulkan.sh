#!/bin/bash
# Build tts-server with Vulkan (cross-vendor GPU: AMD, Intel, NVIDIA).
# Does not require CUDA toolkit — uses Vulkan SDK instead.
set -e
cd "$(dirname "$0")/.."

echo "Building tts-server (Vulkan GPU mode)..."

if ! command -v cmake &>/dev/null; then
  echo "ERROR: cmake not found. Install it first."
  exit 1
fi

if ! pkg-config --exists vulkan 2>/dev/null; then
  echo "ERROR: Vulkan SDK not found."
  echo "Install Vulkan SDK first:"
  echo "  Ubuntu/Debian: sudo apt install libvulkan-dev vulkan-tools"
  echo "  Arch:          sudo pacman -S vulkan-headers vulkan-tools"
  echo "  Or download from: https://vulkan.lunarg.com/"
  exit 1
fi

# Check for ggml submodule
if [ ! -f ggml/CMakeLists.txt ]; then
  echo "Initializing ggml submodule..."
  git submodule update --init --recursive
fi

rm -rf build-vulkan
mkdir build-vulkan
cd build-vulkan

cmake .. -DGGML_VULKAN=ON
cmake --build . --config Release -j "$(nproc)"

echo ""
echo "Build complete! Binary: build-vulkan/tts-server"
echo "Start with: GGML_BACKEND=Vulkan0 ./start-cpu.sh start"