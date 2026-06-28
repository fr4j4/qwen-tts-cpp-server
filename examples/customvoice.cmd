@echo off

set PATH=%~dp0..\build\Release;%PATH%

qwen-tts.exe ^
    --model ..\models\qwen-talker-1.7b-customvoice-Q8_0.gguf ^
    --codec ..\models\qwen-tokenizer-12hz-Q8_0.gguf ^
    --speaker vivian ^
    --lang English ^
    -o customvoice.wav < prompt.txt

pause
