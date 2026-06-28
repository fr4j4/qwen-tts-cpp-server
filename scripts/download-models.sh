#!/bin/bash
# Download Qwen3-TTS GGUF models from HuggingFace.
#
# Usage:
#   ./scripts/download-models.sh                  # default: 0.6b customvoice Q4_K_M
#   ./scripts/download-models.sh 0.6b-customvoice # talker + tokenizer (Q4_K_M)
#   ./scripts/download-models.sh 1.7b-base        # larger model
#   ./scripts/download-models.sh 0.6b-customvoice Q8_0  # specific quantization
#
# Available talker models:
#   0.6b-base          629 MB (Q4_K_M)  — zero-shot TTS, no voice cloning
#   0.6b-customvoice   605 MB (Q4_K_M)  — zero-shot TTS + voice cloning (9 named speakers)
#   1.7b-base          1.2 GB (Q4_K_M)  — higher quality, no voice cloning
#   1.7b-customvoice   1.2 GB (Q4_K_M)  — higher quality + voice cloning
#   1.7b-voicedesign   1.2 GB (Q4_K_M)  — voice synthesis from text description
#
# Quantization variants:
#   Q4_K_M   smallest, lowest VRAM     (recommended for GPU < 4GB or CPU)
#   Q8_0     recommended default       (best quality/size ratio)
#   BF16     source faithful, max precision
#   F32      reference, debug only
#
# Tokenizer is shared across all talker models.
set -eu

REPO="Serveurperso/Qwen3-TTS-GGUF"
DIR="models"
mkdir -p "$DIR"

TALKER="${1:-0.6b-customvoice}"
QUANT="${2:-Q4_K_M}"

# Validate talker choice
case "$TALKER" in
  0.6b-base|0.6b-customvoice|1.7b-base|1.7b-customvoice|1.7b-voicedesign) ;;
  *)
    echo "Unknown talker model: $TALKER"
    echo "Valid: 0.6b-base | 0.6b-customvoice | 1.7b-base | 1.7b-customvoice | 1.7b-voicedesign"
    exit 1
    ;;
esac

# Validate quantization
case "$QUANT" in
  Q4_K_M|Q8_0|BF16|F32) ;;
  *)
    echo "Unknown quantization: $QUANT"
    echo "Valid: Q4_K_M | Q8_0 | BF16 | F32"
    exit 1
    ;;
esac

TALKER_FILE="qwen-talker-${TALKER}-${QUANT}.gguf"
TOKENIZER_FILE="qwen-tokenizer-12hz-${QUANT}.gguf"

dl() {
  local file="$1"
  if [ -f "$DIR/$file" ]; then
    echo "[OK] $file already exists"
    return
  fi
  echo "[Download] $file"
  # Try hf CLI first, fall back to wget
  if command -v hf &>/dev/null; then
    hf download --quiet "$REPO" "$file" --local-dir "$DIR"
  elif command -v huggingface-cli &>/dev/null; then
    huggingface-cli download --quiet "$REPO" "$file" --local-dir "$DIR"
  else
    echo "Neither 'hf' nor 'huggingface-cli' found. Trying wget..."
    wget -c -q "https://huggingface.co/${REPO}/resolve/main/${file}" -O "$DIR/$file"
  fi
}

echo "=== Downloading Qwen3-TTS GGUF models ==="
echo "Talker:    $TALKER ($QUANT)"
echo "Tokenizer: 12Hz ($QUANT)"
echo "Target:    $DIR/"
echo ""

dl "$TALKER_FILE"
dl "$TOKENIZER_FILE"

echo ""
echo "Done! Models saved to $DIR/"
echo ""
echo "File sizes:"
ls -lh "$DIR/$TALKER_FILE" "$DIR/$TOKENIZER_FILE" 2>/dev/null | awk '{print $5, $9}'
echo ""
echo "To use a different talker model, edit server.py or set environment variables:"
echo "  TALKER_MODEL=models/$TALKER_FILE"
echo "  CODEC_MODEL=models/$TOKENIZER_FILE"