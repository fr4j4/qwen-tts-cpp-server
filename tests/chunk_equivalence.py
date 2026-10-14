#!/usr/bin/env python3
"""Compara un PCM en streaming por chunks contra la referencia de un solo bloque.

Ambos PCM s16le 24 kHz mono, misma petición (texto, voz, seed): los
códigos son idénticos, sólo cambia cómo el codec los decodificó (de una
vez vs por chunks con 2 s de contexto izquierdo). Reporta:

  - largo idéntico (muestras)
  - SNR global del chunked vs referencia (dB)
  - por cada unión entre chunks: SNR local en ±40 ms y "salto" en la
    unión (|x[j]-x[j-1]|) comparado con el de la referencia en el mismo
    punto. Un clic aparece como salto >> referencia o SNR local baja.

Las uniones se calculan con el mismo plan que el server: primer chunk de
--first frames, luego el doble, hasta --chunk frames (1 frame = 1920 muestras).

Uso: tests/chunk_equivalence.py ref.pcm test.pcm --first 2 --chunk 300
Sale con 1 si el SNR global < --min-snr o alguna unión parece un clic.
"""
import argparse
import json
import math
import struct
import sys

HOP = 1920


def load(path):
    data = open(path, "rb").read()
    return list(struct.unpack("<%dh" % (len(data) // 2), data))


def snr_db(ref, test):
    sig = sum(x * x for x in ref)
    err = sum((a - b) ** 2 for a, b in zip(ref, test))
    if err == 0:
        return float("inf")
    if sig == 0:
        return -float("inf")
    return 10 * math.log10(sig / err)


def joins(n_frames, first, chunk):
    out, pos, size = [], 0, first if 0 < first < chunk else chunk
    while pos + size < n_frames:
        pos += size
        out.append(pos)
        size = min(size * 2, chunk)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--first", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=300)
    ap.add_argument("--min-snr", type=float, default=30.0)
    ap.add_argument("--win", type=int, default=960, help="muestras a cada lado de la unión (960 = 40 ms)")
    a = ap.parse_args()
    ref, test = load(a.ref), load(a.test)
    res = {"n_ref": len(ref), "n_test": len(test), "same_len": len(ref) == len(test)}
    n = min(len(ref), len(test))
    res["snr_db"] = round(snr_db(ref[:n], test[:n]), 2)
    bad = []
    rows = []
    for f in joins(n // HOP, a.first, a.chunk):
        j = f * HOP
        lo, hi = max(1, j - a.win), min(n, j + a.win)
        local = snr_db(ref[lo:hi], test[lo:hi])
        jump_t = abs(test[j] - test[j - 1])
        jump_r = abs(ref[j] - ref[j - 1])
        # Escala típica de salto entre muestras en la ventana de la referencia.
        typ = sorted(abs(ref[i] - ref[i - 1]) for i in range(lo, hi))[int(0.99 * (hi - lo - 1))] or 1
        click = jump_t > 2 * max(typ, jump_r) and jump_t > 300
        rows.append({"frame": f, "local_snr_db": round(local, 1), "jump": jump_t, "jump_ref": jump_r,
                     "p99_ref": typ, "click": click})
        if click:
            bad.append(f)
    res["joins"] = rows
    res["clicks"] = bad
    res["min_local_snr_db"] = min((r["local_snr_db"] for r in rows), default=None)
    ok = res["same_len"] and res["snr_db"] >= a.min_snr and not bad
    res["ok"] = ok
    print(json.dumps(res))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
