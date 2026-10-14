#!/usr/bin/env python3
"""Banco de pruebas de streaming para tts-server (solo stdlib).

Mide, del lado del cliente, para un POST /v1/audio/speech con
response_format=pcm:
  - ttfb_ms : POST enviado -> primer byte de audio PCM recibido
  - hdr_ms  : POST enviado -> cabeceras HTTP recibidas
  - total_ms: POST enviado -> fin del cuerpo
  - audio_s : duración del audio (s16le 24 kHz mono)
  - rtf     : total_ms / audio_s
  - chunks  : número de lecturas de red con datos y su cronología
  - md5     : md5 del PCM (determinismo con seed fija)

Una línea JSON por petición en stdout. Opcionalmente guarda el PCM.

Uso:
  tests/bench_stream.py --port 8899 --text "Hola." --voice serena --seed 42 [-n 3] [--save out.pcm]
  tests/bench_stream.py --port 8899 --text-file textos.txt ...
"""
import argparse
import hashlib
import http.client
import json
import sys
import time

SR = 24000
BYTES_PER_SAMPLE = 2


def run_once(host, port, text, voice, seed, save=None, timeout=600):
    body = json.dumps({"input": text, "voice": voice, "seed": seed,
                       "response_format": "pcm"}).encode()
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    conn.connect()
    t0 = time.perf_counter()
    conn.request("POST", "/v1/audio/speech", body=body,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    t_hdr = time.perf_counter()
    if resp.status != 200:
        err = resp.read().decode(errors="replace")
        conn.close()
        return {"error": f"HTTP {resp.status}: {err}"}
    md5 = hashlib.md5()
    n_bytes = 0
    t_first = None
    events = []  # (ms desde t0, bytes acumulados)
    out = open(save, "wb") if save else None
    while True:
        data = resp.read1(65536)
        if not data:
            break
        now = time.perf_counter()
        if t_first is None:
            t_first = now
        n_bytes += len(data)
        md5.update(data)
        if out:
            out.write(data)
        events.append((round((now - t0) * 1000, 1), n_bytes))
    t_end = time.perf_counter()
    conn.close()
    if out:
        out.close()
    audio_s = n_bytes / (SR * BYTES_PER_SAMPLE)
    total_ms = (t_end - t0) * 1000
    # Agrupa lecturas en "ráfagas" (huecos > 50 ms) = chunks del codec.
    bursts = []
    last = None
    for ms, acc in events:
        if last is None or ms - last > 50:
            bursts.append([ms, acc])
        else:
            bursts[-1][1] = acc
        last = ms
    return {
        "chars": len(text),
        "hdr_ms": round((t_hdr - t0) * 1000, 1),
        "ttfb_ms": round((t_first - t0) * 1000, 1) if t_first else None,
        "total_ms": round(total_ms, 1),
        "audio_s": round(audio_s, 3),
        "rtf": round(total_ms / 1000 / audio_s, 3) if audio_s else None,
        "ttfb_frac": round((t_first - t0) * 1000 / total_ms, 3) if t_first else None,
        "bursts": [{"t_ms": b[0], "audio_s": round(b[1] / (SR * BYTES_PER_SAMPLE), 2)} for b in bursts],
        "md5": md5.hexdigest(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--text-file")
    ap.add_argument("--voice", default="serena")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-n", type=int, default=1, help="repeticiones")
    ap.add_argument("--label", default="")
    ap.add_argument("--save", help="guarda el PCM de la última repetición")
    a = ap.parse_args()
    text = a.text if a.text is not None else open(a.text_file, encoding="utf-8").read().strip()
    for i in range(a.n):
        r = run_once(a.host, a.port, text, a.voice, a.seed,
                     save=a.save if i == a.n - 1 else None)
        r = {"label": a.label, "rep": i, **r}
        print(json.dumps(r, ensure_ascii=False), flush=True)
        if "error" in r:
            sys.exit(1)


if __name__ == "__main__":
    main()
