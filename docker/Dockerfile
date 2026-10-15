# Multi-stage image of tts-server (web UI embedded, no Python).
#
#   docker build -f docker/Dockerfile --build-arg VARIANT=cpu  -t qwen-tts:cpu  .
#   docker build -f docker/Dockerfile --build-arg VARIANT=cuda -t qwen-tts:cuda .
#
#   docker run -d -p 8080:8080 -v "$PWD/models:/models" qwen-tts:cpu
#   docker run -d --gpus all -p 8080:8080 -v "$PWD/models:/models" qwen-tts:cuda
#
# /models must hold the talker and the codec GGUF (scripts/download-*.sh);
# pick other files with -e TALKER=... -e CODEC=... (paths inside /models).
# The build stage runs scripts/package.sh, the same as the release
# packages ; the final stage only carries that package.
ARG VARIANT=cpu
ARG BUILD_IMAGE_cpu=ubuntu:22.04
ARG BUILD_IMAGE_cuda=nvidia/cuda:12.4.1-devel-ubuntu22.04
# Runtime : cudart and cuBLAS ship inside the package, so plain Ubuntu is
# enough for both variants (the NVIDIA container toolkit mounts the driver).
ARG RUN_IMAGE=ubuntu:22.04

FROM ${BUILD_IMAGE_cpu} AS build-cpu
FROM ${BUILD_IMAGE_cuda} AS build-cuda

FROM build-${VARIANT} AS build
ARG VARIANT
ARG CUDA_ARCHS="75-virtual;80-virtual;86-real;89-real"
# Parallel compile jobs: the all-CPU-variant (and CUDA) build is heavy ;
# raise it on a build server with --build-arg JOBS=$(nproc).
ARG JOBS=4
RUN apt-get update && apt-get install -y --no-install-recommends cmake make g++ git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN git config --global --add safe.directory /src \
    && VERSION="$(git describe --tags --always 2>/dev/null || echo docker)" \
       CMAKE_ARGS="$([ "$VARIANT" = cuda ] && echo "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHS")" \
       JOBS="$JOBS" \
       scripts/package.sh "$VARIANT" \
    && mkdir /out && cp -r dist/qwen-tts-*-linux-x64-"$VARIANT"/. /out/

FROM ${RUN_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home /opt/qwen-tts tts
COPY --from=build /out /opt/qwen-tts
ENV TALKER=qwen-talker-0.6b-customvoice-Q4_K_M.gguf \
    CODEC=qwen-tokenizer-12hz-Q4_K_M.gguf \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
VOLUME /models
EXPOSE 8080
USER tts
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1
ENTRYPOINT ["/bin/sh", "-c", "exec /opt/qwen-tts/tts-server --model \"/models/$TALKER\" --codec \"/models/$CODEC\" --host 0.0.0.0 --port 8080 \"$@\"", "--"]
