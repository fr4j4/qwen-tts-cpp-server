#!/bin/bash
# Download Qwen3-TTS 1.7B VoiceDesign talker GGUF.
# Voice synthesis from text attribute description (e.g. "young female, warm voice").
# Only available in 1.7B size. Requires GPU with >= 4GB VRAM or 8GB+ RAM for CPU.
#
# Usage: ./scripts/download-1.7b-voicedesign.sh [quantization]
set -eu
DIR="models"
mkdir -p "$DIR"
QUANT="${1:-Q4_K_M}"
REPO="Serveurperso/Qwen3-TTS-GGUF"
FILE="qwen-talker-1.7b-voicedesign-${QUANT}.gguf"

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