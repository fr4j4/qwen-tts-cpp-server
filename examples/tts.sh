#!/bin/bash

set -eu

../build/qwen-tts \
    --model ../models/qwen-talker-1.7b-voicedesign-Q8_0.gguf \
    --codec ../models/qwen-tokenizer-12hz-Q8_0.gguf \
    --instruct "male, young adult, moderate pitch" \
    --lang English \
    -o tts.wav < prompt.txt
