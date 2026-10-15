#!/usr/bin/env bash
# scripts/package.sh — portable Linux package of tts-server (no Python).
#
# Builds with dynamically loaded ggml backends (GGML_BACKEND_DL), every CPU
# variant (the best one for the running CPU is picked at start) and a
# relocatable layout ($ORIGIN rpath, static libstdc++), then packs
#   dist/qwen-tts-<version>-linux-x64-<variant>.tar.gz
# with tts-server (web UI embedded at http://127.0.0.1:8080/), the CLI
# tools, the ggml libraries and backends, the model download scripts and
# a README.
#
# Usage:
#   scripts/package.sh cpu              # CPU backends only
#   scripts/package.sh vulkan           # CPU + Vulkan (needs Vulkan SDK / libvulkan-dev + glslc)
#   scripts/package.sh cuda             # CPU + CUDA, ships cudart + cuBLAS next to the binary
#   BUILD_DIR=build-x scripts/package.sh cpu   # reuse / choose the build dir
#   SKIP_BUILD=1 BUILD_DIR=build-dl scripts/package.sh cuda   # package an existing build
#
# Extra cmake flags can be passed in CMAKE_ARGS (e.g. CUDA architectures:
#   CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86" scripts/package.sh cuda).
# JOBS caps the parallel compile jobs (default: all cores) ; the CUDA and
# all-CPU-variant builds are heavy, use e.g. JOBS=4 on a desktop in use.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD

VARIANT=${1:-cpu}
case "$VARIANT" in
  cpu)    BACKEND_FLAGS=() ;;
  vulkan) BACKEND_FLAGS=(-DGGML_VULKAN=ON) ;;
  cuda)   BACKEND_FLAGS=(-DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON) ;;
  *) echo "variant must be cpu, vulkan or cuda" >&2; exit 2 ;;
esac

BUILD_DIR=${BUILD_DIR:-build-pkg-$VARIANT}
VERSION=${VERSION:-$(git describe --tags --always --dirty 2>/dev/null || echo dev)}
NAME="qwen-tts-$VERSION-linux-x64-$VARIANT"
OUT=$ROOT/dist
STAGE=$OUT/$NAME

if [ -z "${SKIP_BUILD:-}" ]; then
  # nvcc rejects too-new host compilers ; reuse the choice of build-gpu.sh.
  HOST_FLAG=()
  if [ "$VARIANT" = cuda ] && [ -z "${CUDAHOSTCXX:-}" ] && command -v nvcc >/dev/null; then
    read -r NVCC_MAJ NVCC_MIN <<< "$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')"
    if [ "$NVCC_MAJ" -ge 13 ]; then MAX_GCC=15
    elif [ "$NVCC_MAJ" -eq 12 ] && [ "$NVCC_MIN" -ge 8 ]; then MAX_GCC=14
    elif [ "$NVCC_MAJ" -eq 12 ] && [ "$NVCC_MIN" -ge 4 ]; then MAX_GCC=13
    else MAX_GCC=12; fi
    GXX_MAJ=$(g++ -dumpversion | cut -d. -f1)
    if [ "$GXX_MAJ" -gt "$MAX_GCC" ]; then
      for v in $(seq "$MAX_GCC" -1 9); do
        if command -v "g++-$v" >/dev/null; then HOST_FLAG=(-DCMAKE_CUDA_HOST_COMPILER="$(command -v g++-$v)"); break; fi
      done
    fi
  fi
  # shellcheck disable=SC2086
  cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DQWEN_PORTABLE=ON \
    -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON \
    -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF \
    -DGGML_BLAS=OFF \
    "${BACKEND_FLAGS[@]}" "${HOST_FLAG[@]}" ${CMAKE_ARGS:-}
  cmake --build "$BUILD_DIR" --config Release -j "${JOBS:-$(nproc)}"
fi

B=$BUILD_DIR
[ -x "$B/tts-server" ] || { echo "ERROR: $B/tts-server not found" >&2; exit 1; }

rm -rf "$STAGE"
mkdir -p "$STAGE/scripts"

# Binaries (the web UI is embedded in tts-server).
for exe in tts-server qwen-tts qwen-codec quantize; do
  [ -x "$B/$exe" ] && cp "$B/$exe" "$STAGE/"
done

# ggml core libraries under the SONAME the binaries ask for, and the
# backends ggml_backend_load_all() picks up from the binary's folder.
cp -L "$B/libggml.so.0" "$B/libggml-base.so.0" "$STAGE/"
cp "$B"/libggml-cpu-*.so "$STAGE/"
[ "$VARIANT" = vulkan ] && cp "$B/libggml-vulkan.so" "$STAGE/"
if [ "$VARIANT" = cuda ]; then
  cp "$B/libggml-cuda.so" "$STAGE/"
  # CUDA runtime + cuBLAS next to the backend ($ORIGIN rpath) ; the driver
  # (libcuda.so.1) always comes from the system.
  for lib in libcudart.so.12 libcublas.so.12 libcublasLt.so.12; do
    path=$(ldd "$B/libggml-cuda.so" | awk -v l="$lib" '$1 == l { print $3 }')
    if [ -n "$path" ] && [ -f "$path" ]; then
      cp -L "$path" "$STAGE/"
    else
      echo "WARN: $lib not found via ldd ; the package will need the CUDA runtime installed" >&2
    fi
  done
fi

cp scripts/download-*.sh "$STAGE/scripts/"
cp LICENSE "$STAGE/"
cat > "$STAGE/README.txt" <<EOF
qwen-tts $VERSION ($VARIANT, Linux x64)

1. Download a model and the audio codec (into ./models):
     scripts/download-0.6b-customvoice.sh      # talker, voices: serena, vivian, ...
     scripts/download-tokenizer.sh             # audio codec, needed by every model
   (other talkers: scripts/download-*.sh)

2. Start the server:
     ./tts-server --model models/qwen-talker-0.6b-customvoice-Q4_K_M.gguf \\
                  --codec models/qwen-tokenizer-12hz-Q4_K_M.gguf --port 8080

3. Open http://127.0.0.1:8080/ (web UI) or call the OpenAI-style API:
     curl -s http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \\
       -d '{"input":"Hola, buenos días.","voice":"serena","response_format":"wav"}' > hola.wav

Backends: the CPU backend for this processor is chosen at start
(libggml-cpu-*.so).$( [ "$VARIANT" = vulkan ] && echo " Vulkan (libggml-vulkan.so) is used when a Vulkan driver is present (NVIDIA, AMD, Intel).")$( [ "$VARIANT" = cuda ] && echo " CUDA (libggml-cuda.so) needs an NVIDIA driver; cudart and cuBLAS are included.")
Force a backend with GGML_BACKEND=CPU (or CUDA0 / Vulkan0). ./tts-server --help lists all options.
Requires glibc >= the one of the build machine (see the release notes).
EOF

# Smoke check: the binary starts and resolves its libraries from the package.
if ! LD_LIBRARY_PATH= "$STAGE/tts-server" --help >/dev/null 2>&1; then
  echo "ERROR: $STAGE/tts-server does not start from the package folder" >&2
  exit 1
fi

mkdir -p "$OUT"
tar -C "$OUT" -czf "$OUT/$NAME.tar.gz" "$NAME"
( cd "$OUT" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256" )
echo "Package: $OUT/$NAME.tar.gz ($(du -h "$OUT/$NAME.tar.gz" | cut -f1))"
