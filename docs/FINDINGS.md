# Engineering Findings — Performance Campaign (Sep 2026)

Registro técnico de la campaña de optimización y validación del motor
(rama `development`, commits `12a5d42` → `369dbb7`). Todo lo aquí
documentado fue verificado empíricamente en RTX 3060 (sm_86, driver
NVIDIA 595.91, Arch Linux, nvcc 12.4) con los modelos Q4_K_M oficiales.

## 1. Resultados de rendimiento (validados)

### GPU, RTF por modelo (menor = mejor; 0.2 = 5x tiempo real)

| Modelo | Config | RTF | Talker ms/frame | Estado |
|---|---|---|---|---|
| 1.7B VoiceDesign | baseline (pin viejo, legacy) | 0.205 | 5.20 | punto de partida |
| 1.7B VoiceDesign | v0.24 + CP_EXEC + TXEXEC + graphs | **0.153** | 4.58 | -25% vs baseline |
| 1.7B Base (clonación) | legacy | 0.214 | 5.02 | prompt ICL largo |
| 1.7B Base (clonación) | + execs | **0.196** | 5.17 | bit-exacto vs legacy |
| 0.6B CustomVoice | legacy | 0.184 | 5.2 | estable |
| 0.6B CustomVoice | + TXEXEC (CP_EXEC gated) | ~0.16 | 3.25-3.60 | el decode más rápido |
| 1.7B (cualquiera) | CPU fallback | ~1.7-1.8 | 47 | 16 hilos, OpenBLAS |

### Memoria

- KV cache del talker en F16 (solo path GPU+flash-attention):
  **896 MB → 448 MB** (28 layers, 8 kv heads, head_dim 128, max 4096).
  CPU y el predictor quedan F32 (referencia bit-exacta).
- VRAM total 1.7B cargado: ~2.4 GB → ~1.9 GB.

### Desglose por etapa (1.7B, por frame de 83 ms de audio)

| Etapa | Baseline | Final | Comentario |
|---|---|---|---|
| CodePredictor (16 sub-pasos AR) | 9.75 ms | 8.45-8.84 | techo: 16 pasos inherentemente secuenciales |
| TalkerDecode | 5.20 | 4.58 | ver §4: CUDA graphs automáticos NO lo capturan |
| CodecDecode | ~1.0 | ~1.0 | ok |
| HostCompose | ~8 | ~8 | despreciable |

## 2. La campaña de bugs (causa raíz + fix de cada uno)

### Bug 1 — Pin viejo de ggml: keying de CUDA-graph reuse roto

- **Síntoma A (corrupción)**: con grafos persistentes alloc-once, el
  output divergía determinísticamente (2048 frames, sin EOS).
- **Síntoma B (crash)**: segfault intermitente dentro de `libcuda.so`
  bajo uso repetido.
- **Causa raíz**: el pin (snapshot pre-histórico sin git) keyeaba la
  reutilización de capturas CUDA-graph por el **puntero del primer nodo
  del grafo** + uid. Seguro bajo el patrón rebuild-alloc-compute de
  llama.cpp (grafo nuevo cada llamada, mismo puntero = mismo grafo);
  inseguro con grafos persistentes que coexisten con otros grafos
  reconstruidos (capturas stale/cruzadas).
- **Fix**: actualizar el pin a **ggml v0.24.0** (vendido in-tree, el
  mecanismo de captura fue reescrito upstream). Compila sin cambios de
  API en `src/`.
- **Lección**: no confiar del keying por punteros; al actualizar el pin,
  re-validar bit-exactitud desde cero (los kernels cambian: ver §5).

### Bug 2 — Pool compartido movía buffers de grafos persistentes

- **Síntoma**: con v0.24 y los grafos del predictor allocados sobre el
  scheduler compartido del pipeline, un prefill de texto largo crecía el
  pool → segfault determinístico en libcuda al tocar los buffers
  stale. Reproducible al 100%: corto OK → largo → crash. Con 7 GB
  libres de VRAM (no era OOM).
- **Causa raíz**: `ggml_backend_sched_reset` + `alloc_graph` de otra
  grafía puede reubicar/encoger los buffers que los grafos persistentes
  ya habían recibido. El sched no "sabe" que esos punteros siguen vivos.
- **Fix**: **scheduler aislado por grafo persistente** (alloc exactamente
  una vez en `*_exec_init`, jamás reset). El pool del talker puede crecer
  sin tocar esos buffers. Ver `code-predictor-exec.h` y `talker-exec.h`.
- **Lección**: todo grafo de vida larga en ggml debe tener su propio
  sched + buffer; compartir pool con grafos que se re-alocan es una
  carrera contra el allocator.

### Bug 3 — CP_EXEC en 0.6B CustomVoice: logits NaN (ABIERTO, gated)

- **Síntoma**: con `KALI_QWEN_CP_EXEC=1` y el modelo 0.6B, a partir de
  la **2ª síntesis del proceso** los logits del predictor llegan **NaN**
  (verificado con instrumentación directa: `sum=nan`, `nonzero=2048/2048`
  pero valores `nan`). Como el muestreo es host-side con las mismas
  uniformes de Philox, todos los códigos colapsan al bucket 2047, nunca
  hay EOS → "audio infinito" (2048 frames). 100% reproducible: CLI
  multi-línea `Hola.\nHola.\n`, 2ª línea rota.
- **Lo que NO es**:
  - No es TXEXEC (absuelto: solo TXEXEC multi-línea 3/3 idénticas).
  - No es GGML_CUDA_GRAPHS (reproducido con graphs OFF).
  - No es OOM (7 GB libres).
  - No es el server (reproducido en el CLI single-process).
- **Diferencia geométrica única del 0.6B**: `mtp_proj identity` (sin
  capa de proyección; el grafo arranca directo del leaf `x_in` de
  tamaño talker_hidden). El 1.7B usa `mtp_proj linear` (2048→1024) y
  pasó soaks de 40+ runs. Sospecha no verificada: con identity, el
  leaf de entrada queda más expuesto al reuse/pool y algo lo pisa (o
  el evento de captura del allocator interactúa con la primera op del
  grafo distinto a como lo hace tras un mul_mat). Requiere debug con
  captura por-nodo (compute-sanitizer o dumps intermedios por op).
- **Mitigación vigente**: gate por geometría en `pipeline-tts.cpp` —
  CP_EXEC solo procede si `code_predictor.mtp_proj_w != NULL`. En 0.6B
  cae a legacy rebuild (RTF 0.184, 100% estable) con log claro. Es
  imposible disparar el bug sin forzar el código.
- **Lección**: la validación por geometría es obligatoria para
  optimizaciones a nivel grafo: el 1.7B y el Base (ambos mtp linear)
  pasaban todo; solo el tercer modelo destapó el bug.

### Bug 4 — v0.24 aborta duro ante OOM (deuda conocida)

- **Síntoma**: `cudaMalloc failed: out of memory` durante un prefill
  grande → `GGML_ASSERT` → `abort()` (SIGABRT) → el server **muere**
  en vez de responder un error HTTP. El pin viejo degradaba mejor.
- **Contexto**: v0.24 reserva picos mayores para prefills largos
  (~1.34 GB pedidos con texto de ~20 s de audio).
- **Deuda**: pre-chequear VRAM libre antes de prefills grandes y
  responder HTTP 503; o acotar el workspace del prefill. Mientras
  tanto: si kali-companion convive con otras cargas GPU, dejar margen
  o supervisar el proceso (el provider ya hace respawn al detectar
  muerte del binario — confirmar que ese path funciona).

## 3. Convenciones y decisiones de diseño vigentes

### Grafos persistentes (patrón exec)

- Un grafo de vida larga = `ggml_context` propio + grafo custom propio
  + `backend_sched_new()` PROPIO + `alloc_graph` una sola vez.
- Por replay: solo uploads de leaves que cambian (`tensor_set`) +
  `sched_graph_compute` + `tensor_get` de salidas. **Nunca** reset/alloc
  en el camino caliente.
- Constantes del grafo (posiciones, máscaras) se suben una vez en init.
- Si el problema admite índice dinámico, usar `ggml_set_rows` con leaf
  i32 (patrón del talker: escritura del KV en fila `n_past` variable
  sin rebuild; kernel CUDA disponible para F16/F32 dst).
- Fallback obligatorio: si init falla, el pipeline cae al rebuild
  legacy. Los execs son soft-failure por diseño.

### Ventana fija del talker exec (W=2048)

- El grafo lee un extent FIJO `[hd, W, n_kv]` del KV cache; las filas
  más allá de `n_past` contienen basura pero quedan fuera por la fila
  de máscara causal (leaf F16, una fila por step desde tabla estática
  [W,W] precalculada).
- Guard: `n_past >= W` → fallback legacy (cubre prompt ~250 + ~1800
  frames generados).

### Flags de entorno

| Flag | Efecto | Default |
|---|---|---|
| `KALI_QWEN_CP_EXEC=0` | desactiva el predictor pre-construido (opt-out; unset/1/otro = ON). Solo aplica en geometrías mtp linear (1.7B); el 0.6B queda gated a legacy por geometría | **ON** |
| `KALI_QWEN_TXEXEC=0` | desactiva el talker decode pre-construido (opt-out; unset/1/otro = ON; GPU + flash attention, W=2048 con legacy fallback) | **ON** |
| `GGML_CUDA_GRAPHS=ON` (cmake) | captura automática de CUDA graphs | OFF upstream; ON en `scripts/build-gpu.sh` |

Los execs quedaron **ON por defecto** (opt-out) tras validar la campaña: si fallan en init, caen solos al legacy rebuild (soft failure); el 0.6B (mtp identity) queda siempre en legacy por el gate de geometría (ver §2 Bug 3).

### Ojo con el CLI vs server

- El CLI (`qwen-tts`) valida idiomas más estricto que el server
  (`--lang es` rechazado en CustomVoice; el server lo acepta/auto-detecta).
- El CLI multi-línea (`--stream-by-line -o -`) es EL herramienta para
  reproducir bugs de estado-entre-síntesis en un solo proceso.

## 5. Numerica entre versiones de ggml (esperado, no regressión)

- Los kernels de v0.24 no son bit-idénticos a los del pin viejo (orden
  de reducción distinto). Consecuencias observadas: EOS frame 99 vs 100,
  duraciones que varían ±0.1 s, CPU golden md5 distinto.
- **No** es regresión: la calidad audible fue validada por oído (voz
  Serena es-CL, A/B pin viejo vs v0.24: indistinguible).
- Implicancia: los golden md5 son POR VERSIÓN de ggml. Al actualizar el
  pin de nuevo, regenerar golden y re-validar (ver §6).

## 6. Procedimiento de validación (regresión manual)

1. Recompilar CPU y CUDA.
2. Server legacy en 8899 (GPU) y 8897 (CPU), texto medio fijo seed 42.
3. Verificar: determinismo 3x md5, EOS presente, duración razonable,
   `fa=on kv_talker=f16` en GPU / `fa=off ... f32` en CPU.
4. Con execs ON: secuencia hostil corto→largo→medio (sin NaN, sin 2048
   frames, sin segfault) + md5 por (texto,seed) estables.
5. 0.6B: verificar que el gate loguea "unsupported geometry" y el audio
   sale bien con legacy.
6. A/B audible final de un humano.

## 7. Pendientes (orden de valor)

1. **Root cause del NaN en 0.6B** (§2 Bug 3) — desbloquearía CP_EXEC
   universal (RTF ~0.15 en 0.6B también).
2. **Captura CUDA-graph del talker**: los graphs automáticos no agarran
   el grafo de decode (5.2 → 4.58 vino solo del pre-build; el launch
   overhead sigue). Investigar qué nodo bloquea (¿set_rows? ¿FA?).
   Potencial: talker ~1.5-2 ms/f → RTF global ~0.12-0.13.
3. **OOM → HTTP 503** (§2 Bug 4).
4. Soak producción → monitorizar execs en producción (defaults ya volteados a ON; el soak ES producción).
5. Warmup en el provider Python (esconde ~2.3 s de CUDA init) y
   streaming PCM (TTFA real ~35 ms ya existe; kali-core pide wav
   one-shot hoy).
6. `tests/regression.sh` que automatice §6.
7. Tag/release con binaries (sm_86) para las 2 PCs.
