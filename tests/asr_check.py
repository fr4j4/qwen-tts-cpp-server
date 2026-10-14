#!/usr/bin/env python3
"""Chequeo de contenido por ASR (opcional): ¿el audio dice el texto?

Transcribe un PCM s16le 24 kHz mono con vosk y calcula el recall de
palabras del texto de referencia (fracción de palabras del texto que
aparecen, en orden, en la transcripción; LCS a nivel palabra).

Detecta truncamientos/divagaciones que el md5 y la duración no ven
(p. ej. el modelo deja de leer el texto a mitad y rellena con ruido).

Uso (con el python que tenga vosk instalado):
  python -I tests/asr_check.py --model /ruta/vosk-model-es --text-file t.txt audio.pcm
Imprime JSON {"recall": ..., "hyp": ...}. Requiere ffmpeg en PATH.
"""
import argparse
import json
import re
import subprocess
import unicodedata


def norm_words(s):
    s = unicodedata.normalize("NFC", s.lower())
    return re.findall(r"[a-záéíóúüñ]+", s)


def lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text-file", required=True)
    ap.add_argument("pcm")
    a = ap.parse_args()
    import vosk
    vosk.SetLogLevel(-1)
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", "24000", "-ac", "1",
                          "-i", a.pcm, "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    rec = vosk.KaldiRecognizer(vosk.Model(a.model), 16000)
    rec.AcceptWaveform(raw)
    hyp = json.loads(rec.FinalResult())["text"]
    ref = norm_words(open(a.text_file, encoding="utf-8").read())
    recall = lcs_len(ref, norm_words(hyp)) / max(1, len(ref))
    print(json.dumps({"recall": round(recall, 3), "ref_words": len(ref), "hyp": hyp}, ensure_ascii=False))


if __name__ == "__main__":
    main()
