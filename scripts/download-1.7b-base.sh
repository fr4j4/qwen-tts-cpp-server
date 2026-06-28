#!/bin/bash
# Download Qwen3-TTS 1.7B Base talker GGUF.
# Higher quality zero-shot TTS with named speakers. No voice cloning.
# Larger model — requires GPU with >= 4GB VRAM or 8GB+ RAM for CPU.
#
# Usage: ./scripts/download-1.7b-base.sh [quantization]
set -eu
DIR="models"
mkdir -p "$DIR"
QUANT="${1:-Q4_K_M}"
REPO="Serveurperso/Qwen3-TTS-GGUF"
FILE="qwen-talker-1.7b-base-${QUANT}.gguf"

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