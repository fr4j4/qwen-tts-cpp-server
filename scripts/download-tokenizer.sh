#!/bin/bash
# Download the tokenizer (codec) GGUF model.
# Required by ALL talker models — the tokenizer converts 12Hz codes to 24kHz audio.
#
# Usage: ./scripts/download-tokenizer.sh [quantization]
#   quantization: Q4_K_M (default) | Q8_0 | BF16 | F32
set -eu
DIR="models"
mkdir -p "$DIR"
QUANT="${1:-Q4_K_M}"
REPO="Serveurperso/Qwen3-TTS-GGUF"
FILE="qwen-tokenizer-12hz-${QUANT}.gguf"

if [ -f "$DIR/$FILE" ]; then
  echo "[OK] $FILE already exists"
  exit 0
fi

echo "[Download] $FILE"
if command -v hf &>/dev/null; then
  hf download --quiet "$REPO" "$FILE" --local-dir "$DIR"
elif command -v huggingface-cli &>/dev/null; then
  huggingface-cli download --quiet "$REPO" "$FILE" --local-dir "$DIR"
else
  wget -c -q "https://huggingface.co/${REPO}/resolve/main/${FILE}" -O "$DIR/$FILE"
fi
echo "Done: $DIR/$FILE"