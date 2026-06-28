#!/bin/bash
# Download Qwen3-TTS 1.7B VoiceDesign models (talker + tokenizer) in Q4_K_M.
# VoiceDesign allows synthesizing voices from text descriptions (e.g. "young female, warm voice").
# Requires ~2GB disk space. Model size: ~1.8GB (talker 1.5GB + tokenizer 255MB).
#
# Usage: ./scripts/download-voicedesign.sh [quantization]
set -eu
DIR="models"
mkdir -p "$DIR"
QUANT="${1:-Q4_K_M}"
REPO="Serveurperso/Qwen3-TTS-GGUF"

TALKER_FILE="qwen-talker-1.7b-voicedesign-${QUANT}.gguf"
TOKENIZER_FILE="qwen-tokenizer-12hz-${QUANT}.gguf"

download_file() {
  local file="$1"
  if [ -f "$DIR/$file" ]; then
    echo "[OK] $file already exists"
    return 0
  fi
  echo "[Download] $file"
  if command -v hf &>/dev/null; then
    hf download --quiet "$REPO" "$file" --local-dir "$DIR"
  elif command -v huggingface-cli &>/dev/null; then
    huggingface-cli download --quiet "$REPO" "$file" --local-dir "$DIR"
  else
    wget -c -q "https://huggingface.co/${REPO}/resolve/main/${file}" -O "$DIR/$file"
  fi
}

download_file "$TALKER_FILE"
download_file "$TOKENIZER_FILE"

echo ""
echo "Done! Models saved to $DIR/"
echo "  $DIR/$TALKER_FILE"
echo "  $DIR/$TOKENIZER_FILE"
echo ""
echo "To start with VoiceDesign model:"
echo "  ./start-voicedesign.sh          (CPU mode)"
echo "  GGML_BACKEND=CUDA0 ./start-voicedesign.sh  (GPU mode)"