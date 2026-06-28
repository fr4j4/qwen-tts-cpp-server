#!/bin/bash
# Derive lighter GGUFs from the F32 source-of-truth produced by convert.py.
# Each F32 model under models/ is quantized to BF16, Q8_0 and Q4_K_M.
#
# Three target precisions cover the useful range : BF16 for max precision
# on CUDA, Q8_0 as the balanced default, Q4_K_M as the smallest variant
# that still sounds correct.
#
# Quantization policy is centralized in tools/quantize.cpp should_quantize :
# RVQ codebooks (quantizer.quantizers.*) and any embedding/projection layer
# wrapping them stay at F32 in every variant. Nearest-neighbor lookup is
# sensitive to per-row quantization noise ; even BF16 mantissa truncation
# drifts codes enough to break voice fidelity. Conv weights stay at source
# dtype and are cast to F16 at load time by gf_load_conv_f16 (ARM im2col
# strict). Same policy as omnivoice.cpp / acestep.cpp keeping audio-critical
# paths intact.

set -eu

Q="./build/quantize"

quantize() {
    local src="$1" type="$2"
    local out="${src/-F32.gguf/-${type}.gguf}"
    if [ -f "$out" ]; then
        echo "[Skip] $out"
    else
        $Q "$src" "$out" "$type"
    fi
}

for src in models/qwen-*-F32.gguf; do
    [ -f "$src" ] || continue
    quantize "$src" BF16
    quantize "$src" Q8_0
    quantize "$src" Q4_K_M
done
