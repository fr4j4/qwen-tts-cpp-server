#!/bin/bash

set -eu

../build/qwen-tts \
    --model ../models/qwen-talker-1.7b-base-Q8_0.gguf \
    --codec ../models/qwen-tokenizer-12hz-Q8_0.gguf \
    --lang English \
    -o base.wav < prompt.txt
