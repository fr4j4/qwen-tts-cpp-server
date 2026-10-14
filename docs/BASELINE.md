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

## Archivos de esta fase

- `tests/bench_stream.py`: cliente de streaming (primer byte, RTF, ráfagas, md5).
- `tests/bench_baseline.py`: línea base completa (server propio + /proc + `[Perf]`).
- `tests/regression.sh`: FINDINGS §6 automatizado (+ golden y ASR opcionales).
- `tests/asr_check.py`: recall de palabras por ASR (vosk).
- `tests/texts/`: textos fijos `corto`, `medio`, `largo`, `extralargo`.
