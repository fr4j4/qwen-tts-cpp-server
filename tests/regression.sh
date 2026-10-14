#!/usr/bin/env bash
# tests/regression.sh — automatiza docs/FINDINGS.md §6 (regresión del server).
#
# Levanta su propio tts-server en un puerto dedicado (no toca otros
# procesos), sintetiza textos fijos con seed fija y verifica:
#   1. flags de carga: GPU -> fa=on kv_talker=f16 ; CPU -> fa=off kv_talker=f32
#   2. 0.6B (mtp identity): el gate loguea "unsupported geometry"
#   3. determinismo: texto medio x3 -> md5 del PCM idéntico
#   4. EOS presente, frames < max_new_tokens (2048), duración razonable
#   5. secuencia hostil corto -> largo -> medio: sin NaN, sin 2048 frames,
#      server vivo, y el md5 del medio igual al del paso 3 (sin fuga de
#      estado entre síntesis)
#   6. (opcional) md5 contra golden: --golden FILE / --update-golden
#      (los golden son POR versión de ggml y backend, ver FINDINGS §5)
#   7. (opcional) contenido por ASR: si VOSK_PY (python con vosk) y
#      VOSK_MODEL están definidos, cada audio debe contener >= ASR_MIN
#      (0.85) de las palabras del texto, en orden. Detecta truncamientos
#      que md5/duración no ven.
#
# Uso:
#   tests/regression.sh                       # GPU, binario y modelos por defecto
#   tests/regression.sh --backend cpu
#   tests/regression.sh --bin build-gpu/tts-server --port 8899
#   TALKER=/ruta/talker.gguf CODEC=/ruta/codec.gguf tests/regression.sh
#
# Variables: TALKER, CODEC (rutas a los gguf), MODELS_DIR (default: models,
# con fallback a /mnt/data2/models/voice), SEED (42), VOICE (serena),
# VOSK_PY, VOSK_MODEL, ASR_MIN (chequeo 7).
# Salida: 0 si todo pasa, 1 si algo falla. Log del server en $OUT_DIR.
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD

BACKEND=gpu
BIN=""
PORT=""
GOLDEN=""
UPDATE_GOLDEN=0
SEED=${SEED:-42}
VOICE=${VOICE:-serena}
OUT_DIR=${OUT_DIR:-$(mktemp -d -t qwen-regression.XXXXXX)}
mkdir -p "$OUT_DIR"

while [ $# -gt 0 ]; do
  case "$1" in
    --backend) BACKEND=$2; shift 2 ;;
    --bin) BIN=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --golden) GOLDEN=$2; shift 2 ;;
    --update-golden) UPDATE_GOLDEN=1; shift ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    *) echo "argumento desconocido: $1" >&2; exit 2 ;;
  esac
done

if [ "$BACKEND" = gpu ]; then
  PORT=${PORT:-8899}
  if [ -z "$BIN" ]; then
    for b in build-gpu/tts-server build-cuda/tts-server; do
      [ -x "$b" ] && { BIN=$b; break; }
    done
  fi
  EXPECT_FLAGS=("fa=on" "kv_talker=f16")
else
  PORT=${PORT:-8897}
  if [ -z "$BIN" ]; then
    for b in build/tts-server build-cpu/tts-server; do
      [ -x "$b" ] && { BIN=$b; break; }
    done
  fi
  EXPECT_FLAGS=("fa=off" "kv_talker=f32")
fi
[ -x "${BIN:-}" ] || { echo "ERROR: binario tts-server no encontrado (usa --bin)" >&2; exit 2; }

MODELS_DIR=${MODELS_DIR:-models}
[ -d "$MODELS_DIR" ] || MODELS_DIR=/mnt/data2/models/voice
TALKER=${TALKER:-$MODELS_DIR/qwen-talker-0.6b-customvoice-Q4_K_M.gguf}
CODEC=${CODEC:-$MODELS_DIR/qwen-tokenizer-12hz-Q4_K_M.gguf}
for f in "$TALKER" "$CODEC"; do
  [ -f "$f" ] || { echo "ERROR: falta modelo $f (ver scripts/download-*.sh)" >&2; exit 2; }
done

if ss -ltn 2>/dev/null | grep -q ":$PORT\b"; then
  echo "ERROR: el puerto $PORT ya está en uso; usa --port" >&2
  exit 2
fi

LOG=$OUT_DIR/server.log
BENCH="python3 $ROOT/tests/bench_stream.py --port $PORT --voice $VOICE --seed $SEED"
PASS=0
FAIL=0
ok()   { echo "  PASS  $*"; PASS=$((PASS + 1)); }
bad()  { echo "  FAIL  $*"; FAIL=$((FAIL + 1)); }
jget() { python3 -c "import json,sys; print(json.loads(sys.argv[1]).get(sys.argv[2]))" "$1" "$2"; }

echo "== regression: backend=$BACKEND bin=$BIN port=$PORT seed=$SEED voice=$VOICE"
echo "   talker=$TALKER"
echo "   codec=$CODEC"
echo "   out=$OUT_DIR"

[ "$BACKEND" = cpu ] && export GGML_BACKEND=cpu
"$BIN" --model "$TALKER" --codec "$CODEC" --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT

for _ in $(seq 120); do
  curl -sf "127.0.0.1:$PORT/health" > /dev/null && break
  kill -0 $SRV 2>/dev/null || { echo "ERROR: el server murió al cargar"; tail -20 "$LOG"; exit 1; }
  sleep 1
done
curl -sf "127.0.0.1:$PORT/health" > /dev/null || { echo "ERROR: server no responde"; exit 1; }
echo "   versión: $(grep -m1 'qwentts.cpp' "$LOG")"

# --- 1-2. flags de carga -----------------------------------------------------
echo "-- carga"
LOADED=$(grep -m1 '\[Pipeline\] Loaded:' "$LOG")
for f in "${EXPECT_FLAGS[@]}"; do
  if grep -q -- "$f" <<< "$LOADED"; then ok "flag $f"; else bad "flag $f ausente en: $LOADED"; fi
done
if grep -q 'mtp_proj identity' "$LOG"; then
  if grep -q 'unsupported geometry' "$LOG"; then ok "0.6B: gate CP_EXEC -> legacy"; else bad "0.6B sin log 'unsupported geometry'"; fi
fi

# Sintetiza un texto, valida EOS/frames/duración/NaN en su tramo de log.
# Deja el md5 en la variable LAST_MD5. Uso: synth <etiqueta> <archivo>
LAST_MD5=""
synth() {
  local label=$1 file=$2 from res err md5 audio chars frames seg
  from=$(($(wc -l < "$LOG") + 1))
  res=$($BENCH --text-file "$file" --label "$label" --save "$OUT_DIR/$label.pcm" 2>&1 | tail -1)
  err=$(jget "$res" error 2>/dev/null)
  if [ "$err" != "None" ]; then
    bad "$label: petición falló: $res"
    LAST_MD5=""
    return
  fi
  md5=$(jget "$res" md5); audio=$(jget "$res" audio_s); chars=$(jget "$res" chars)
  LAST_MD5=$md5
  sleep 0.2
  seg=$(tail -n +"$from" "$LOG")
  frames=$(grep -o 'Generation done : [0-9]*' <<< "$seg" | tail -1 | grep -o '[0-9]*$')
  echo "        $label: frames=${frames:-?} audio=${audio}s md5=$md5 ttfb=$(jget "$res" ttfb_ms)ms rtf=$(jget "$res" rtf)"
  if grep -q 'EOS at step' <<< "$seg"; then ok "$label: EOS presente"; else bad "$label: sin EOS"; fi
  if [ -n "$frames" ] && [ "$frames" -lt 2048 ]; then ok "$label: frames<2048"; else bad "$label: frames=${frames:-?} (techo)"; fi
  if grep -qiw 'nan' <<< "$seg"; then bad "$label: NaN en el log"; else ok "$label: sin NaN"; fi
  # Duración razonable: entre chars/40 y chars/6 segundos (español ~13-15 chars/s).
  if python3 -c "import sys; a,c=float(sys.argv[1]),int(sys.argv[2]); sys.exit(0 if c/40 <= a <= c/6 else 1)" "$audio" "$chars"; then
    ok "$label: duración razonable (${audio}s para $chars chars)"
  else
    bad "$label: duración ${audio}s fuera de rango para $chars chars"
  fi
  if [ -n "${VOSK_PY:-}" ] && [ -n "${VOSK_MODEL:-}" ]; then
    local asr recall
    asr=$("$VOSK_PY" -I "$ROOT/tests/asr_check.py" --model "$VOSK_MODEL" --text-file "$file" "$OUT_DIR/$label.pcm" 2>&1 | tail -1)
    recall=$(jget "$asr" recall 2>/dev/null)
    if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)" "$recall" "${ASR_MIN:-0.85}" 2>/dev/null; then
      ok "$label: ASR recall $recall"
    else
      bad "$label: ASR recall ${recall:-?} < ${ASR_MIN:-0.85} (contenido truncado o divagación)"
    fi
  fi
}

T=$ROOT/tests/texts

# --- 3-4. determinismo -------------------------------------------------------
echo "-- determinismo (medio x3, seed $SEED)"
MD5S=()
for i in 1 2 3; do
  synth "medio#$i" "$T/medio.txt"
  MD5S+=("$LAST_MD5")
done
if [ -n "${MD5S[0]}" ] && [ "${MD5S[0]}" = "${MD5S[1]}" ] && [ "${MD5S[1]}" = "${MD5S[2]}" ]; then
  ok "determinismo md5 x3 (${MD5S[0]})"
else
  bad "md5 distintos: ${MD5S[*]}"
fi
MEDIO_MD5=${MD5S[0]}

# --- 5. secuencia hostil -----------------------------------------------------
echo "-- secuencia hostil corto -> largo -> medio"
synth corto "$T/corto.txt"; CORTO_MD5=$LAST_MD5
synth largo "$T/largo.txt"; LARGO_MD5=$LAST_MD5
synth medio "$T/medio.txt"
if [ -n "$LAST_MD5" ] && [ "$LAST_MD5" = "$MEDIO_MD5" ]; then
  ok "medio tras hostil = medio inicial (sin fuga de estado)"
else
  bad "medio tras hostil ($LAST_MD5) != inicial ($MEDIO_MD5)"
fi
if kill -0 $SRV 2>/dev/null; then ok "server vivo tras la secuencia"; else bad "server muerto (crash)"; fi
if grep -qiE 'segmentation|SIGSEGV|GGML_ASSERT|cudaMalloc failed' "$LOG"; then bad "crash/assert en el log"; fi

# --- 6. golden ---------------------------------------------------------------
if [ -n "$GOLDEN" ]; then
  CUR=$(printf 'corto %s\nmedio %s\nlargo %s\n' "$CORTO_MD5" "$MEDIO_MD5" "$LARGO_MD5")
  if [ "$UPDATE_GOLDEN" = 1 ]; then
    printf '%s\n' "$CUR" > "$GOLDEN"; echo "   golden actualizado: $GOLDEN"
  elif [ -f "$GOLDEN" ]; then
    if diff <(printf '%s\n' "$CUR") "$GOLDEN" > /dev/null; then ok "md5 = golden ($GOLDEN)"; else
      bad "md5 != golden ($GOLDEN)"; diff <(printf '%s\n' "$CUR") "$GOLDEN" | sed 's/^/        /'; fi
  else
    echo "   (golden $GOLDEN no existe; usa --update-golden para crearlo)"
  fi
fi

echo "== resultado: $PASS PASS, $FAIL FAIL (log: $LOG)"
[ "$FAIL" -eq 0 ]
