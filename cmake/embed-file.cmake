# embed-file.cmake : turn a file into a C++ header holding its bytes.
# Usage: cmake -DINPUT=<file> -DOUTPUT=<header> -DNAME=<symbol> -P embed-file.cmake
# Produces `static const unsigned char NAME[]` and `NAME_len`. Used to
# ship the web UI (static/index.html) inside tts-server, so the binary
# serves it without any file next to it.
file(READ "${INPUT}" HEX HEX)
string(LENGTH "${HEX}" HEX_LEN)
math(EXPR N "${HEX_LEN} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," BYTES "${HEX}")
string(REGEX REPLACE "(0x[0-9a-f][0-9a-f],){32}" "\\0\n" BYTES "${BYTES}")
file(WRITE "${OUTPUT}.tmp"
"// Generated from ${INPUT} by cmake/embed-file.cmake. Do not edit.\n#pragma once\n#include <cstddef>\nstatic const unsigned char ${NAME}[] = {\n${BYTES}\n};\nstatic const size_t ${NAME}_len = ${N};\n")
file(COPY_FILE "${OUTPUT}.tmp" "${OUTPUT}" ONLY_IF_DIFFERENT)
file(REMOVE "${OUTPUT}.tmp")
