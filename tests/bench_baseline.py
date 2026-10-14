#!/usr/bin/env python3
"""Línea base de latencia/CPU del tts-server (FASE 0 — medir).

Levanta su propio tts-server, hace un warmup descartado y luego, por cada
texto (corto/medio/largo) y repetición, mide desde el cliente (ver
bench_stream.py) y desde /proc del server:

  ttfb_ms     POST -> primer byte PCM en el cliente
  rtf         total / duración del audio
  cpu_pct     CPU del proceso durante la petición, en % de UN núcleo
              (equivalente a pidstat -u: delta utime+stime / wall)
  top_thr_pct CPU del hilo más cargado, en % de un núcleo (pidstat -t)
  perf        desglose [Perf] del server para esa petición

También mide la CPU del proceso en reposo (idle_cpu_pct) entre peticiones.
Imprime una línea JSON por petición y, al final, una tabla Markdown con
medianas.

Uso:
  tests/bench_baseline.py --bin build-gpu/tts-server --label gpu [-n 3] [--port 8899]
  GGML_BACKEND=cpu tests/bench_baseline.py --bin build-cpu/tts-server --label cpu --port 8897
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_stream import run_once  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CLK = os.sysconf("SC_CLK_TCK")


def thread_ticks(pid):
    """{tid: utime+stime} de todos los hilos del proceso."""
    out = {}
    for t in Path(f"/proc/{pid}/task").iterdir():
        try:
            f = (t / "stat").read_text()
        except OSError:
            continue
        fields = f[f.rindex(")") + 2:].split()
        out[int(t.name)] = int(fields[11]) + int(fields[12])
    return out


def cpu_delta(a, b, wall):
    d = {tid: b[tid] - a.get(tid, 0) for tid in b}
    total = sum(d.values()) / CLK / wall * 100
    top = max(d.values()) / CLK / wall * 100 if d else 0.0
    busy = sum(1 for v in d.values() if v / CLK / wall > 0.5)
    return round(total, 1), round(top, 1), busy


PERF_RE = {
    "prefill_ms": r"\[Perf\] Prefill ([\d.]+) ms",
    "ttfa_ms": r"\[Perf\] TTFA ([\d.]+) ms",
    "talker_ms_f": r"\[Perf\] TalkerDecode [\d.]+ ms \(\d+ frames, ([\d.]+) ms/frame\)",
    "cp_ms_f": r"\[Perf\] CodePredictor [\d.]+ ms \(([\d.]+) ms/frame\)",
    "host_ms": r"\[Perf\] HostCompose ([\d.]+) ms",
    "codec_ms": r"\[Perf\] CodecDecode ([\d.]+) ms",
    "frames": r"\[Perf\] Total [\d.]+ ms \((\d+) frames",
    "ar_ms_f": r"\[Perf\] Total [\d.]+ ms \(\d+ frames, ([\d.]+) ms/frame AR",
    "srv_total_ms": r"\[Perf\] Total ([\d.]+) ms",
}


def parse_perf(seg):
    out = {}
    for k, rx in PERF_RE.items():
        m = re.findall(rx, seg)
        if m:
            out[k] = float(m[-1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--texts", default="corto,medio,largo")
    ap.add_argument("--voice", default="serena")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "/mnt/data2/models/voice"))
    ap.add_argument("--talker", default="qwen-talker-0.6b-customvoice-Q4_K_M.gguf")
    ap.add_argument("--codec", default="qwen-tokenizer-12hz-Q4_K_M.gguf")
    ap.add_argument("--log", default=None, help="log del server (default: /tmp)")
    ap.add_argument("--extra", default="{}", help='campos JSON extra por petición, p. ej. \'{"instructions": "..."}\'')
    a = ap.parse_args()
    extra = json.loads(a.extra)

    log_path = a.log or f"/tmp/qwen-baseline-{a.label}.log"
    logf = open(log_path, "w")
    md = Path(a.models_dir)
    proc = subprocess.Popen([a.bin, "--model", str(md / a.talker), "--codec", str(md / a.codec),
                             "--host", "127.0.0.1", "--port", str(a.port)],
                            stdout=logf, stderr=subprocess.STDOUT)
    try:
        for _ in range(180):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=1)
                break
            except Exception:
                if proc.poll() is not None:
                    sys.exit(f"server murió al cargar; ver {log_path}")
                time.sleep(1)
        loaded = next((l.strip() for l in open(log_path) if "[Pipeline] Loaded:" in l), "")
        print(json.dumps({"label": a.label, "loaded": loaded}), flush=True)

        # Warmup descartado (CUDA init / primeras capturas de grafo).
        run_once("127.0.0.1", a.port, "Hola.", a.voice, a.seed, extra=extra)

        # CPU en reposo.
        t0, w0 = thread_ticks(proc.pid), time.perf_counter()
        time.sleep(3)
        idle, _, _ = cpu_delta(t0, thread_ticks(proc.pid), time.perf_counter() - w0)
        print(json.dumps({"label": a.label, "idle_cpu_pct": idle}), flush=True)

        rows = {}
        for name in a.texts.split(","):
            text = (ROOT / "tests" / "texts" / f"{name}.txt").read_text(encoding="utf-8").strip()
            for rep in range(a.n):
                logf.flush()
                start = os.path.getsize(log_path)
                ta, wa = thread_ticks(proc.pid), time.perf_counter()
                r = run_once("127.0.0.1", a.port, text, a.voice, a.seed, extra=extra)
                wall = time.perf_counter() - wa
                cpu, top, busy = cpu_delta(ta, thread_ticks(proc.pid), wall)
                time.sleep(0.2)
                with open(log_path, errors="replace") as f:
                    f.seek(start)
                    perf = parse_perf(f.read())
                r.pop("bursts", None)
                row = {"label": a.label, "text": name, "rep": rep, **r,
                       "cpu_pct": cpu, "top_thr_pct": top, "busy_threads": busy, **perf}
                print(json.dumps(row, ensure_ascii=False), flush=True)
                rows.setdefault(name, []).append(row)

        def med(k, rs):
            v = [x[k] for x in rs if x.get(k) is not None]
            return round(statistics.median(v), 3) if v else None

        print(f"\n### {a.label}  (idle {idle}% de un núcleo)\n")
        cols = ["audio_s", "frames", "ttfb_ms", "ttfb_frac", "total_ms", "rtf", "cpu_pct", "top_thr_pct",
                "prefill_ms", "ttfa_ms", "talker_ms_f", "cp_ms_f", "codec_ms", "host_ms"]
        print("| texto | " + " | ".join(cols) + " |")
        print("|---|" + "---|" * len(cols))
        for name, rs in rows.items():
            print(f"| {name} | " + " | ".join(str(med(c, rs)) for c in cols) + " |")
    finally:
        proc.terminate()
        proc.wait(timeout=30)


if __name__ == "__main__":
    main()
