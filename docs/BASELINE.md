# Línea base de latencia y CPU (Fase 0 — medir)

Medición del motor **tal como está** (sin cambios de lógica), para tener
contra qué comparar las fases siguientes del roadmap. Fecha: 2026-10-14.

## Cómo reproducir

```bash
# Línea base (levanta su propio server en un puerto dedicado)
tests/bench_baseline.py --bin build-gpu/tts-server --label gpu-exec -n 3
KALI_QWEN_TXEXEC=0 tests/bench_baseline.py --bin build-gpu/tts-server --label gpu-legacy -n 3
tests/bench_baseline.py --bin build-cpu/tts-server --label cpu --port 8897 -n 2

# Primer byte de una petición suelta contra un server ya levantado
tests/bench_stream.py --port 8899 --text-file tests/texts/medio.txt --seed 42

# Regresión FINDINGS §6 (+ chequeo de contenido por ASR opcional)
VOSK_PY=/ruta/python-con-vosk VOSK_MODEL=/ruta/vosk-model-es tests/regression.sh
tests/regression.sh --backend cpu
```

Definiciones (todas medidas por petición, desde el cliente salvo `[Perf]`):

- **primer byte (TTFB)**: `POST` enviado → primer byte de PCM recibido en el
  cliente (`response_format: "pcm"`, streaming chunked).
- **RTF**: tiempo total de la respuesta / duración del audio (menor = mejor).
- **CPU**: delta `utime+stime` del proceso (de `/proc/<pid>/task/*/stat`) /
  tiempo de pared, en % de **un** núcleo (equivalente a `pidstat -u`).
  "hilo top" = el hilo más cargado (equivalente a `pidstat -t`).
- **[Perf]**: desglose que loguea el propio server por síntesis.

## Entorno

| | |
|---|---|
| Commit | `d2a579c` (rama `development`; binarios recompilados en ese commit) |
| GPU | NVIDIA RTX 3060 12 GB (sm_86), driver 595.99.02, CUDA 12.4.131 |
| CPU | AMD Ryzen 7 3700X (8C/16T), 15 GB RAM |
| SO | Linux 7.0.0-38, gcc 15.2 (host de nvcc: g++ 13.4) |
| Modelos | `qwen-talker-0.6b-customvoice-Q4_K_M.gguf` + `qwen-tokenizer-12hz-Q4_K_M.gguf` (de `/mnt/data2/models/voice`) |
| Petición | voz `serena`, `seed: 42`, `lang=auto`, `response_format: "pcm"` |
| Build GPU | `build-gpu/` limpia: `-DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DCMAKE_CUDA_ARCHITECTURES=86`, Release `-O3` |
| Build CPU | `build-cpu/`: `GGML_BLAS=ON` (OpenBLAS), `GGML_NATIVE=ON`, Release; 8 hilos ggml |
| Flags runtime | defaults: `KALI_QWEN_TXEXEC` ON (W=512), `KALI_QWEN_CP_EXEC` ON pero gated a legacy en 0.6B; carga: `fa=on kv_talker=f16` (GPU) / `fa=off kv_talker=f32` (CPU) |
| `GGML_CUDA_GRAPHS` en CMakeCache | **ON** en `build-cuda/` y en `build-gpu/` (verificado, gotcha FINDINGS §7.2) |

Ruido de la máquina: es la PC de uso diario; durante las mediciones había
otros procesos (load avg 2-8, otro `tts-server` y un `llama-server` residentes
en la GPU, ambos ociosos). Las repeticiones con seed fija dieron md5 idénticos
y dispersión de tiempos < 3 % en GPU, salvo un outlier anotado.

Textos (`tests/texts/`, español):

| texto | chars | audio GPU | audio CPU |
|---|---|---|---|
| corto | 54 | 7,0 s | 7,1 s |
| medio | 315 | 21,4 s | 27,0 s |
| largo | 739 | 58,4 s (legacy) / **30,0 s truncado** (exec) | 67,4 s |

(La numérica CPU f32 difiere de la GPU f16, así que con la misma seed la
duración también difiere; ver FINDINGS §5.)

## Resultados

### GPU, defaults (`build-gpu`, talker exec W=512) — medianas de 3

| texto | audio | frames | **primer byte** | total | primer byte / total | **RTF** | CPU proceso | hilo top |
|---|---|---|---|---|---|---|---|---|
| corto | 7,04 s | 88 | **1217 ms** | 1218 ms | 0,999 | 0,173 | 99 % | 99 % |
| medio | 21,36 s | 267 | **3501 ms** | 3504 ms | 0,999 | 0,164 | 100 % | 100 % |
| largo | 30,00 s ⚠ | 375 | **3971 ms** | 4972 ms | 0,798 | 0,166 | 100 % | 100 % |

⚠ `largo` sale **truncado** con el exec por defecto — ver hallazgo 1.

Desglose `[Perf]` (medianas):

| texto | Prefill | TTFA interno | TalkerDecode | CodePredictor | CodecDecode | HostCompose |
|---|---|---|---|---|---|---|
| corto | 8,7 ms | 26,7 ms | 2,18 ms/f | 10,11 ms/f | 91 ms | 5,0 ms |
| medio | 13,6 ms | 34,4 ms | 2,18 ms/f | 9,82 ms/f | 267 ms | 15,3 ms |
| largo | 20,6 ms | 34,9 ms | 2,19 ms/f | 9,80 ms/f | 403 ms | 21,5 ms |

CPU del proceso en reposo (entre peticiones): **0,0 %**.

### GPU, legacy (`KALI_QWEN_TXEXEC=0`) — medianas de 3

| texto | audio | frames | **primer byte** | total | primer byte / total | **RTF** | CPU proceso | hilo top |
|---|---|---|---|---|---|---|---|---|
| corto | 7,04 s | 88 | **1325 ms** | 1326 ms | 0,999 | 0,188 | 99 % | 99 % |
| medio | 21,36 s | 267 | **4023 ms** | 4026 ms | 0,999 | 0,188 | 100 % | 100 % |
| largo | 58,40 s | 730 | **4910 ms** | 12552 ms | 0,389 | 0,215 | 99 % | 99 % |

| texto | Prefill | TTFA interno | TalkerDecode | CodePredictor | CodecDecode | HostCompose |
|---|---|---|---|---|---|---|
| corto | 8,6 ms | 30,0 ms | 4,00 ms/f | 9,94 ms/f | 91 ms | 5,1 ms |
| medio | 13,6 ms | 34,8 ms | 4,17 ms/f | 9,74 ms/f | 270 ms | 15,5 ms |
| largo | 21,8 ms | 44,3 ms | 4,56 ms/f | 11,36 ms/f | 782 ms | 45,5 ms |

`corto` y `medio` dan el **mismo md5** con exec y con legacy (no hay wrap de
la ventana): el exec es bit-exacto mientras `T_ctx + frames < 512`.
Outlier: una repetición de `largo` legacy tardó 16,3 s (RTF 0,278) por carga
externa; la mediana no la incluye.

### CPU (`build-cpu`, OpenBLAS, 8 hilos) — medianas de 2

| texto | audio | frames | **primer byte** | total | primer byte / total | **RTF** | CPU proceso | hilo top |
|---|---|---|---|---|---|---|---|---|
| corto | 7,12 s | 89 | **11,19 s** | 11,19 s | 1,000 | 1,572 | 679 % | 87 % |
| medio | 27,04 s | 338 | **41,78 s** | 48,57 s | 0,861 | 1,796 | 699 % | 90 % |
| largo | 67,44 s | 843 | **42,34 s** | 127,35 s | 0,333 | 1,889 | 688 % | 88 % |

| texto | Prefill | TTFA interno | TalkerDecode | CodePredictor | CodecDecode | HostCompose |
|---|---|---|---|---|---|---|
| corto | 132 ms | 180 ms | 20,0 ms/f | 48,3 ms/f | 4970 ms | 6,6 ms |
| medio | 346 ms | 400 ms | 24,9 ms/f | 47,5 ms/f | 23745 ms | 25,1 ms |
| largo | 824 ms | 884 ms | 35,7 ms/f | 47,8 ms/f | 56071 ms | 63,9 ms |

CPU en reposo: 0,3 %. En CPU el modo no es tiempo real (RTF > 1) y el
**CodecDecode se lleva ~44 % del total** (≈ 0,8 s de cómputo por segundo de
audio), más que el talker.

### Build `build-cuda/` (la que había en el disco)

`build-cuda/CMakeCache.txt` tiene `GGML_CUDA_GRAPHS=ON`, pero también
**`GGML_CUDA_DEBUG=ON`** y **`CMAKE_CXX_FLAGS=-DTX_NAN_PROBE`** (restos de la
sesión de debug del NaN en 0.6B), y su binario estaba compilado desde
`5fa1ead` (rama de debug), no desde `development`. Lo recompilé en `d2a579c`
sin reconfigurar. Con esa build, cada síntesis escribe ~95 000 líneas
`nodes_fused: ...` al log (~4 MB por petición de 20 s) y el RTF empeora ~4 %
(medio: 0,171 vs 0,164; corto: 0,182 vs 0,173). Para producción usar una
build limpia.

## Hipótesis del roadmap

### (a) "Con el default de 24 s, el primer byte llega al final de la síntesis en frases cortas" — **CONFIRMADA**

`codec_chunk_sec` vale 24 s por defecto (`src/qwen.cpp:222`) y el server no
lo cambia, así que el decoder de streaming sólo emite cuando se juntan 300
frames o en el EOS:

- GPU: en `corto` y `medio` (< 24 s de audio) el primer byte llega al
  **99,9 % del tiempo total** (1,2 s y 3,5 s), todo el audio en una sola
  ráfaga. El TTFA interno (primer frame de códigos) es de 27-35 ms: hay un
  factor **~35-100×** entre lo que el motor podría entregar y lo que recibe
  el cliente.
- En `largo` legacy (58 s) el primer byte llega a los 4,9 s, justo cuando se
  completan los primeros 24 s de audio (300 frames × ~16 ms/f), y luego llegan
  ráfagas de 24 s.
- En CPU es peor: 11 s de silencio para una frase de 7 s, 42 s para la
  primera ráfaga en textos largos.

El "TTFA real ~35 ms" de FINDINGS §7.5 es interno al pipeline; **no es lo que
ve un cliente HTTP** con los defaults actuales.

### (b) "Un núcleo queda girando al 100 % durante la generación en GPU" — **CONFIRMADA**

- Durante toda síntesis GPU el proceso usa 99-100 % de un núcleo, todo en un
  solo hilo (el worker de httplib que atiende la petición), y casi todo en
  modo usuario: `pidstat -u -t -p <pid> 1` muestra el hilo en `%usr 100,
  %system 0` segundo a segundo; por `/proc`, 348 ticks de usuario vs 1 de
  sistema en 3,5 s de síntesis.
- El trabajo real del host es mínimo (HostCompose: 15 ms de 3500 ms), y la GPU
  promedió ~75 % de utilización en la ventana medida (`nvidia-smi -lms 100`).
  Lo más consistente es una espera activa (spin) en las sincronizaciones
  CUDA (16 pasos del CodePredictor + talker por frame), que es el
  comportamiento por defecto del runtime CUDA. No pude confirmarlo a nivel
  de símbolo: `perf` está bloqueado (`kernel.perf_event_paranoid=4`) y no hay
  `gdb`.
- En reposo el proceso no consume CPU (0,0 %): sólo gira mientras sintetiza.

## Hallazgos adicionales (fuera del alcance de medir, para la fase siguiente)

1. **Regresión de contenido en la ventana circular W=512 (commit `d2439f8`)**.
   Cuando `T_ctx + frames generados > 512`, la ventana circular sobrescribe
   las posiciones más viejas del KV del talker, que son justamente las del
   **prompt con el texto**: el modelo deja de "ver" lo que le falta por leer.
   Resultado con `largo` (739 chars, `T_ctx`≈220):

   | modo | audio | recall ASR del texto |
   |---|---|---|
   | exec W=512 (default) | 30,0 s | **0,41** — deja de leer en "…los álamos que la rodeaban habían…" |
   | legacy (`KALI_QWEN_TXEXEC=0`) | 58,4 s | 0,99 |

   Con un texto de 617 chars pasa lo mismo (exec 41,6 s, corta en el mismo
   punto; legacy 45 s completo). El A/B por oído de FINDINGS §7.2 se hizo con
   textos cortos, donde no hay wrap. `tests/regression.sh` lo detecta con el
   chequeo por ASR (falla en `largo`; con `KALI_QWEN_TXEXEC=0` pasa 36/36).
   Mitigación mientras tanto: `KALI_QWEN_TXEXEC=0` (o subir W) para textos
   largos; arreglo: no desalojar el prompt (anclar las posiciones del
   prompt y hacer circular sólo la zona generada).
2. **Texto extralargo (1170 chars) también falla en legacy**: lee ~1000
   chars bien y luego divaga sin EOS hasta el techo de 2048 frames (163,8 s).
   Con el exec, en cambio, se corta a los 20 s. Hace falta partir el texto
   por oraciones en el server (ya está en el roadmap).
3. **`scripts/build-gpu.sh` no compila en esta máquina**: fuerza
   `-DCMAKE_CUDA_HOST_COMPILER=$(command -v g++)` (g++ 15) y nvcc 12.4 sólo
   acepta hasta gcc 13. Sin ese flag nvcc usa su g++-13 y compila bien (así se
   hizo `build-gpu/`).

## Resultado de `tests/regression.sh` (2026-10-14, con ASR)

| backend / modo | resultado | nota |
|---|---|---|
| GPU `build-gpu`, defaults (exec W=512) | 35 PASS, **1 FAIL** | `largo`: recall ASR 0,41 (hallazgo 1) |
| GPU `build-gpu`, `KALI_QWEN_TXEXEC=0` | 36 PASS, 0 FAIL | |
| CPU `build-cpu` | 36 PASS, 0 FAIL | md5 medio `9e59e8cb…`, estable x3 y tras la secuencia hostil |

md5 GPU (seed 42, ggml v0.24): corto `fa6e0764…`, medio `392db29d…`
(idénticos exec/legacy), largo `645ac909…` (exec, truncado) / `244e3c13…` (legacy).

## Fase 1 — antes / después (rama `claude/fase-1-streaming`)

Misma máquina, modelos, textos, voz y seed. "Antes" = línea base de arriba
(GPU defaults). "Después" = `build-gpu/` compilada con `scripts/build-gpu.sh`
corregido, defaults del server nuevos. Primer byte con servidor caliente
(medianas de 3).

### GPU

| texto | primer byte antes | **primer byte después** | RTF antes | RTF después | audio antes → después | CPU proceso |
|---|---|---|---|---|---|---|
| corto | 1217 ms | **57 ms** | 0,173 | 0,196 | 7,04 → 7,04 s | 99 % → 99 % |
| medio | 3501 ms | **58 ms** | 0,164 | 0,180 | 21,36 → 21,36 s | 100 % → 100 % |
| largo | 3971 ms | **68 ms** | 0,166 ⚠ truncado | 0,184 | 30,0 ⚠ → 62,1 s (recall 0,41 → 1,0) | 100 % → 100 % |

Primera petición tras levantar el server (texto corto): 179-196 ms sin
warmup → **39-57 ms** con warmup (el arranque tarda +1,0 s).

Desglose `[Perf]` después (medianas): Prefill 9-22 ms, TTFA 29-44 ms,
TalkerDecode 2,2 ms/f (3,3 en largo: pasado W=512 sigue en legacy),
CodePredictor ~10 ms/f, **CodecDecode 225 / 552 / 1134 ms** (antes 91 / 267
/ 403): el arranque progresivo y los 6 s de contexto izquierdo
re-decodifican contexto en cada emit. Es el costo del +6-13 % de RTF en
textos cortos y medios; con `--left-context-sec 2` baja, pero el audio se
aleja más de la decodificación en un bloque (ver abajo).

Meta de la fase (< 150 ms de primer byte en GPU, servidor caliente):
**cumplida** (57-68 ms).

Hipótesis (b) sigue igual: un núcleo al 100 % mientras sintetiza (la fase 1
no lo ataca).

### CPU

Defaults sin cambios en CPU (framing upstream: 24 s, 2 s de contexto),
porque el codec domina (~60 ms/frame) y con RTF > 1 los decodes extra sólo
alargan la síntesis. Medido con `corto`/`medio`:

| config CPU | primer byte | RTF |
|---|---|---|
| default (`--first-chunk-frames 0 --left-context-sec 2`) | 11,7 s / 40,0 s | 1,65 / 1,71 |
| `--first-chunk-frames 2 --left-context-sec 2` | 0,34 s / 0,57 s | 2,10 / 1,93 |
| `--first-chunk-frames 8 --left-context-sec 2` | 1,03 s / 1,28 s | 2,03 / 1,85 |
| `--first-chunk-frames 8 --left-context-sec 6` | 1,15 s / 1,35 s | 2,33 / 2,17 |

`--threads 4 / 8 / 16` (corto, progresivo): RTF 2,79 / 2,53 / 2,64; el
default (8 = núcleos físicos) sigue siendo el mejor en el 3700X.

### Qué cambia el audio (para el A/B por oído)

Muestras WAV en `samples/fase1/` (no versionadas, `*.wav` está en
`.gitignore`):

| archivo | qué es |
|---|---|
| `00-ventana-largo-ANTES-truncado.wav` | `largo` con la ventana circular W=512 (corta a mitad del texto) |
| `00-ventana-largo-DESPUES.wav` | `largo` con el fix (exec hasta W, luego legacy) |
| `00-ventana-largo-REFERENCIA-legacy.wav` | `largo` con `KALI_QWEN_TXEXEC=0` |
| `01-chunks-{corto,medio,largo}-ANTES-24s.wav` | framing anterior (24 s, 2 s de contexto) |
| `01-chunks-{corto,medio,largo}-DESPUES-progresivo.wav` | framing nuevo (2→4→8…, 6 s de contexto) |

1. **Fix de la ventana**: cambia el audio sólo cuando `T_ctx + frames > 512`
   (corto/medio: md5 idéntico al anterior). En `largo` el modelo ahora lee el
   texto completo; los códigos difieren de legacy puro (62,1 s vs 58,4 s)
   porque los primeros pasos van por el exec y el muestreo es caótico, pero
   el recall ASR es 1,0 en ambos.
2. **Chunks progresivos**: los códigos son idénticos; cambia cómo el codec
   decodifica por tramos. `tests/chunk_equivalence.py` contra la
   decodificación en un solo bloque: **sin clics** en ninguna unión (el salto
   entre muestras en cada unión es del orden del de la referencia), SNR
   global 24 dB con 2 s de contexto y **33-35 dB con 6 s** (12 s apenas
   mejora). Las diferencias están repartidas dentro de cada tramo, no en las
   uniones: vienen del contexto izquierdo más corto que la ventana de
   atención del codec, el mismo efecto que ya tenía el framing upstream de
   24 s en textos largos.
3. **Robustez**: `chunk_sec` llega ahora desde el JSON, así que el server lo
   topa en 24 s. Un bloque de 62 s pedía 4 GB de VRAM y ggml abortaba el
   proceso (FINDINGS Bug 4).

### A/B por oído (2026-10-14, usuario)

Escuchado con un reproductor A/B sincronizado (mismo punto de reproducción al
cambiar de versión), voz Serena, seed 42:

| prueba | veredicto |
|---|---|
| Fix de la ventana, texto largo (739 caracteres) | el fix lee todo y suena bien |
| Chunks progresivos, corto (7 s) | indistinguibles |
| Chunks progresivos, medio (21 s) | indistinguibles |
| Chunks progresivos, largo (62 s) | indistinguibles |

Los cambios de audio de la fase 1 quedan **aprobados por oído**.

### Otros cambios de la fase 1

- `language` en el JSON (`"spanish"`, `"english"`, …) llega al prompt
  (verificado: `lang=spanish (id=2054)`). Un idioma inválido devuelve 200
  con cuerpo vacío y el error sólo en el log, porque en streaming las
  cabeceras salen antes de que el ABI lo rechace; ya pasaba igual con una
  `voice` inválida.
- Trazas `[Sample]`/`[Sample-CP]` sólo con `QWEN_DEBUG=1` (0 líneas por
  defecto, 64 con debug).
- `--threads N`, `--chunk-sec`, `--first-chunk-frames`, `--left-context-sec`,
  `--no-warmup`; `KALI_QWEN_TX_WINDOW` para cambiar W del talker exec
  (medido: W=1024/2048 hacen el largo ~10 % más rápido pero el medio 3-9 %
  más lento; queda W=512).
- Web UI: arranque de la reproducción con 250 ms de audio (antes 1 s).
- `scripts/build-gpu.sh` compila con gcc 15 (elige g++-13 para nvcc 12.4).
- ABI v3: `qt_tts_params.codec_first_chunk_frames` y `qt_has_gpu()`.

`tests/regression.sh` después: GPU 36/36 (con ASR); CPU 30/30 (sin ASR) con md5 idénticos a la fase 0 (corto `ae21234d…`, medio `9e59e8cb…`, largo `ee31b22b…`): en CPU el audio no cambia.

## Fase 2 — costo por frame, CPU y robustez (rama `claude/fase-2-costo-por-frame`)

Misma máquina, modelos, textos, voz y seed. "Fase 1" = cifras de la sección
anterior; "Fase 2" = `build-gpu/` al final de la fase, GPU ociosa, medianas de 3.

### GPU, 0.6B CustomVoice

| texto | primer byte F1 → **F2** | RTF F1 → **F2** | CodecDecode F1 → F2 | margen mínimo F2 | CPU proceso |
|---|---|---|---|---|---|
| corto (7 s) | 57 → **58 ms** | 0,196 → **0,190** | 225 → 211 ms | 92 ms | 99 % |
| medio (21 s) | 58 → **61 ms** | 0,180 → **0,171** | 552 → 386 ms | 102 ms | 100 % |
| largo (62 s) | 68 → **69 ms** | 0,184 → **0,182** | 1134 → 1044 ms | 91 ms | 100 % |
| extralargo (1170 chars) | — → **50-86 ms** | 0,187-0,192 → **0,169-0,172** | — | 102 ms | — |

Referencia framing de 24 s (fase 0): RTF 0,173 / 0,164 / 0,180. El margen
mínimo es cuánto audio va por delante de la reproducción si el cliente
reproduce desde el primer byte (`bench_stream.py`, `min_slack_ms`).

### GPU, 1.7B VoiceDesign (predictor exec)

| texto | primer byte | RTF | CodePredictor | CPU proceso |
|---|---|---|---|---|
| corto (4 s) | 40 ms | 0,158 | 6,36 ms/frame | 99 % |
| medio (21 s) | 49 ms | 0,150 | 6,35 ms/frame | 100 % |

Meta del roadmap para el 1.7B (RTF ≤ 0,15): **alcanzada en medio**, 0,158 en
el texto corto.

### Qué se hizo y qué dio

1. **C2: modo de espera de CUDA** (`GGML_CUDA_SCHEDULE=auto|spin|yield|blocking`,
   default auto). Con `blocking`:

   | modelo | CPU auto → blocking | RTF auto → blocking |
   |---|---|---|
   | 1.7B (predictor exec) | 98-100 % → **27-28 %** | 0,160-0,169 → 0,188-0,198 |
   | 0.6B (predictor legacy) | ~100 % → 78-86 % | 0,179-0,197 → 0,214-0,228 |

   Refina la hipótesis (b) de la fase 0: en el 1.7B el núcleo sí se va en
   espera activa; en el 0.6B se va sobre todo en reconstruir los grafos del
   predictor legacy (gated por el NaN de FINDINGS Bug 3), así que L10 también
   es palanca de CPU. El default queda en `auto` (mejor latencia);
   `blocking` sirve si la máquina se comparte con otras cargas.
2. **L5: crecimiento de chunks x2 y luego x4** (2, 4, 16, 64, 256, tope). El
   codec cuesta ~1 ms por frame decodificado (lineal; build y alloc del grafo
   ~0,5 ms por llamada, medido con una sonda temporal), y cada emit
   re-decodifica hasta 6 s de contexto. Menos emits = menos contexto. Recupera
   ~60-100 % del RTF perdido en la fase 1. El primer paso queda en x2: con x4
   directo el 2o chunk llegaba justo cuando se acababa el primero (margen
   -9 ms con `blocking`).
3. **L8: embeddings del predictor desde el mmap del GGUF** en vez de 16 copias
   device→host por frame. Bit-exacto (md5 idéntico con y sin, en 0.6B y
   1.7B). 1.7B: CodePredictor 6,49 → 6,34 ms/frame (-2,3 %). 0.6B: dentro del
   ruido. **L11 descartado**: el embedding del talker ya se arma en host desde
   el mmap (HostCompose 0,06 ms/frame).
4. **Robustez**:
   - `voice`/`language` inválidos → **HTTP 400** con la lista de valores
     aceptados, antes de abrir el stream (antes: 200 con cuerpo vacío).
   - **OOM de VRAM → HTTP 503** (`QT_STATUS_OOM`) en vez de `abort()`. Eran
     dos bugs en el ggml vendido (`ggml_backend_sched_alloc_splits` ignoraba
     un reserve fallido; tras el fallo el gallocr no re-reservaba). Verificado
     forzando un bloque de codec de 6,8 GB: wav → 503 dos veces seguidas, el
     server sigue vivo y las peticiones normales posteriores dan el mismo
     md5. En pcm el 200 ya salió cuando ocurre el OOM: el cuerpo queda vacío.
   - **Textos largos por piezas de oraciones** (`--split-chars`, default 800
     bytes, piezas de hasta ~400). `extralargo` con 6 seeds:

     | | audio | recall ASR | RTF |
     |---|---|---|---|
     | sin partir | 67-143 s | 0,69-0,995 (seed 4: 0,69) | 0,187-0,192 |
     | por oraciones | 92-109 s | 0,98-0,995 en 5/6 (seed 6: 0,85) | 0,169-0,172 |

     Más estable y ~10 % más rápido (cada pieza queda dentro de la ventana
     del exec), pero no elimina el riesgo. Ojo: el "divaga sin EOS" de la
     fase 0 fue en legacy puro con seed 42; con el camino actual y seed 42
     el texto sin partir también termina bien.

### Qué cambia el audio (A/B)

- `02-crecimiento-*`: framing x2 (fase 1) vs x2+x4. Mismos códigos; sólo
  cambian las uniones del codec. Sin clics; SNR vs un bloque 31,6 / 33,6 dB
  (corto / medio), antes 34,6 / 32,7 dB.
- `03-extralargo-*`: sin partir vs por oraciones (seed 42). Cambian los
  códigos (cada pieza es una síntesis nueva) y hay una pausa natural entre
  piezas.
- C2, L8, la validación y el OOM no cambian el audio (md5 idénticos).

Regresión al final de la fase: GPU **41/41** con ASR (la secuencia hostil
suma `extralargo`; se exige un EOS por pieza). CPU **34/34** (sin ASR), con
md5 de corto/medio/largo idénticos a la fase 0. En CPU el split resultó
necesario: con un binario sin él, `extralargo` (seed 42) llegó al techo de
2048 frames sin EOS (163,8 s); partido, 4 piezas con EOS y 93,6 s.

## Archivos de esta fase

- `tests/bench_stream.py`: cliente de streaming (primer byte, RTF, ráfagas, md5).
- `tests/bench_baseline.py`: línea base completa (server propio + /proc + `[Perf]`).
- `tests/regression.sh`: FINDINGS §6 automatizado (+ golden y ASR opcionales).
- `tests/asr_check.py`: recall de palabras por ASR (vosk).
- `tests/texts/`: textos fijos `corto`, `medio`, `largo`, `extralargo`.
- `tests/chunk_equivalence.py` (fase 1): streaming por chunks vs un bloque (SNR, clics en uniones).
