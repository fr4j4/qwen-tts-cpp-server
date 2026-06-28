#!/bin/bash
# Download Qwen3-TTS 0.6B Base talker GGUF.
# Zero-shot TTS with named speakers. No voice cloning.
# Smallest model — best for CPU or GPU < 2GB VRAM.
#
# Usage: ./scripts/download-0.6b-base.sh [quantization]
set -eu
DIR="models"
mkdir -p "$DIR"
QUANT="${1:-Q4_K_M}"
REPO="Serveurperso/Qwen3-TTS-GGUF"
FILE="qwen-talker-0.6b-base-${QUANT}.gguf"

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
echo ""
echo "Also download the tokenizer: ./scripts/download-tokenizer.sh $QUANT"