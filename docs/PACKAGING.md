# Empaquetado y distribución

`tts-server` se distribuye como una carpeta portable (Linux `.tar.gz`, Windows
`.zip`) o como imagen Docker. No necesita Python: la web UI va embebida en el
binario y se sirve en `http://<host>:<puerto>/`.

## Paquetes

| paquete | backends | para quién | tamaño aprox. |
|---|---|---|---|
| `qwen-tts-<v>-linux-x64-vulkan.tar.gz` | CPU + Vulkan | descarga por defecto: GPU NVIDIA, AMD o Intel con sólo el driver; CPU si no hay Vulkan | ~23 MB sin comprimir sin Vulkan; el backend Vulkan se mide en el primer release |
| `qwen-tts-<v>-linux-x64-cuda.tar.gz` | CPU + CUDA (incluye cudart y cuBLAS) | NVIDIA con driver para CUDA 12 (≥ 525) | 422 MB comprimido (742 MB sin comprimir) |
| `qwen-tts-<v>-windows-x64-vulkan.zip` | CPU + Vulkan | igual que Linux | similar a Linux |
| `qwen-tts-<v>-windows-x64-cuda.zip` | CPU + CUDA (incluye cudart y cuBLAS) | igual que Linux | similar a Linux |

Cada paquete trae `tts-server`, las herramientas `qwen-tts`, `qwen-codec` y
`quantize`, las librerías de ggml, los backends, los scripts de descarga de
modelos y un `README.txt`. Los modelos no van dentro (~0,9 GB el 0.6B): se
descargan con `scripts/download-0.6b-customvoice.sh` y
`scripts/download-tokenizer.sh`.

### Cómo funciona la portabilidad

- **Backends dinámicos** (`GGML_BACKEND_DL=ON`): cada backend es una librería
  aparte que `ggml_backend_load_all()` busca junto al binario. Si una no carga
  (sin driver de Vulkan, sin GPU NVIDIA), se usa la siguiente.
- **Todas las variantes de CPU** (`GGML_CPU_ALL_VARIANTS=ON`,
  `GGML_NATIVE=OFF`): 15 builds del backend CPU, de SSE4.2 a AVX-512/Zen4;
  al arrancar se elige la mejor para el procesador (en el Ryzen 3700X de
  pruebas: `haswell`).
- **Reubicable** (`QWEN_PORTABLE=ON`, Linux): rpath `$ORIGIN` y libstdc++ /
  libgcc estáticas, así que el binario sólo pide la glibc del sistema. El
  backend CUDA, enlazado por nvcc, sigue usando la libstdc++ del sistema.
  Los paquetes de Linux se compilan en Ubuntu 22.04 (glibc 2.35).
- **Sin OpenBLAS**: no cambia el audio (md5 idéntico en CPU con y sin BLAS) y
  es una dependencia menos.

## Cómo generarlos

```bash
scripts/package.sh cpu        # Linux, sólo CPU
scripts/package.sh vulkan     # Linux, CPU + Vulkan (Vulkan SDK o libvulkan-dev + glslc)
scripts/package.sh cuda       # Linux, CPU + CUDA (toolkit CUDA)
pwsh scripts/package.ps1 -Variant vulkan   # Windows (MSVC 2022)
pwsh scripts/package.ps1 -Variant cuda
```

Salida en `dist/` con su `.sha256`. Los scripts verifican que el binario
arranca desde la carpeta del paquete antes de comprimir.

## GitHub Actions

- **`ci.yml`** (cada PR y push a `development`): builds portables de CPU en
  Linux (Ubuntu 22.04) y Windows (MSVC 2022), test del ABI C y arranque del
  binario empaquetado. No descarga modelos.
- **`release.yml`** (tag `v*` o a mano desde la pestaña Actions): los cuatro
  paquetes (Vulkan y CUDA × Linux y Windows). Con un tag `v*` publica además
  un GitHub Release con los paquetes y sus `.sha256`; a mano, deja los
  paquetes como artefactos del workflow.

Para publicar una versión: `git tag v0.1.0 && git push origin v0.1.0`.

## Docker

```bash
docker build -f docker/Dockerfile --build-arg VARIANT=cpu  -t qwen-tts:cpu  .
docker build -f docker/Dockerfile --build-arg VARIANT=cuda -t qwen-tts:cuda .

docker run -d -p 8080:8080 -v "$PWD/models:/models" qwen-tts:cpu
docker run -d --gpus all -p 8080:8080 -v "$PWD/models:/models" qwen-tts:cuda
```

Multi-etapa: la etapa de build corre `scripts/package.sh` y la imagen final
(Ubuntu 22.04) sólo lleva el paquete, sin compilador ni Python, con usuario
sin privilegios y healthcheck en `/health`. Otros modelos:
`-e TALKER=<archivo> -e CODEC=<archivo>` (dentro de `/models`). Los
`docker/Dockerfile.{base,cpu,gpu}` anteriores (Python + server.py) quedan
como legado.

## Web UI y API

- `GET /`: la web UI (`static/index.html`, embebida al compilar).
- `GET /v1/info`: modelo, backend (GPU/CPU), versión, voces, idiomas y si es
  VoiceDesign.
- `POST /v1/audio/speech`, `GET /v1/voices`, `GET /v1/models`, `GET /health`:
  sin cambios.
- `server.py` sigue funcionando (reenvía `/v1/info` y `/v1/audio/speech`),
  pero ya no hace falta.

## Verificación

Ver la sección "Resultados de la verificación" al final de este archivo.

## Resultados de la verificación (2026-10-15, Kali, RTX 3060 + Ryzen 7 3700X)

| prueba | resultado |
|---|---|
| Paquete `linux-x64-cuda` (422 MB) descomprimido en otra carpeta | carga todas sus libs desde ahí (`/proc/<pid>/maps`); del sistema sólo `libcuda.so` (driver) y `libstdc++` (la pide el backend CUDA) |
| Variante de CPU elegida al arrancar | `libggml-cpu-haswell.so` (Zen 2) |
| `tests/regression.sh` GPU con el paquete | **34/34**, md5 idénticos a `build-gpu` en corto, medio, largo y extralargo |
| `tests/regression.sh --backend cpu` con el paquete (forzando CPU) | **34/34**, md5 idénticos a `build-cpu` (OpenBLAS): corto `ae21234d`, medio `9e59e8cb`, largo `ee31b22b`, extralargo `2da10d3a` |
| Web UI y `/v1/info` desde el paquete | `GET /` 200 (28,8 KB), `/v1/info` con voces e idiomas del modelo |
| Imagen Docker `cpu` (161 MB) | arranca, UI y `/v1/info` ok, healthcheck `healthy`, usuario `tts`; corto en CPU con md5 `ae21234d` (idéntico a la fase 0) |
| CPU sin OpenBLAS | md5 idéntico al build con OpenBLAS (`9e59e8cb` en medio) |

Encontrado y corregido durante la verificación: el backend CPU usa OpenMP y
la imagen `ubuntu` no trae `libgomp.so.1` (el binario no arrancaba); ahora el
paquete lo incluye. En Windows, `package.ps1` incluye por lo mismo
`vcomp140.dll` y el runtime de MSVC.

No verificado en esta máquina:

- **Vulkan**: no hay headers de Vulkan ni `glslc` (instalarlos pide sudo); se
  compila por primera vez en `release.yml`.
- **Windows**: el primer build real es el de `ci.yml` / `release.yml`.
- **Imagen Docker `cuda`**: no la construí localmente (descarga ~5 GB y
  compila CUDA); usa el mismo paquete verificado arriba.
