// Web Audio PCM player — BUFFER-FIRST playback for s16le 24 kHz mono.
//
// Why buffer-first: with a fast generator (RTF ~0.18, audio arrives ~5x
// real-time) fine-grained streaming playback creates dozens of
// AudioBufferSourceNodes with per-source resamplers; the audio thread
// overloads and the tail of the stream collapses into noise, while the
// exact same bytes saved to a .wav play perfectly. So this player:
//   1. Accumulates chunks (zero playback) until it holds MIN_START_SECONDS
//      of audio — or the stream ends, whichever comes first.
//   2. Starts with a single big source; later arrivals are appended in
//      MAX_SOURCE_SECONDS blocks scheduled contiguously (no gaps, no
//      pile-up: a block starts exactly when the previous ends).
//   3. The AudioContext is created at 24000 Hz → no per-source resampler
//      even when the device mixes at 44.1/48 kHz.

const SAMPLE_RATE = 24000

/** Minimum buffered audio (samples) before playback starts (1 s). */
export const MIN_START_SAMPLES = 24000

/** Max samples per scheduled source (10 s). */
const MAX_SOURCE_SAMPLES = 24000 * 10

export class PcmPlayer {
  private ctx: AudioContext | null = null
  private pending: Float32Array = new Float32Array(0)
  private nextTime = 0
  private started = false
  private ended = false
  private lastSource: AudioBufferSourceNode | null = null
  private onended: (() => void) | null = null

  setOnEnded(fn: (() => void) | null) {
    this.onended = fn
  }

  get isStarted() {
    return this.started
  }

  private ensureCtx(): AudioContext | null {
    if (this.ctx) return this.ctx
    const AC =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!AC) return null
    try {
      this.ctx = new AC({ sampleRate: SAMPLE_RATE, latencyHint: 'interactive' })
    } catch {
      this.ctx = new AC()
    }
    return this.ctx
  }

  /** Feed raw little-endian s16 PCM bytes. Accumulates until start(). */
  push(pcm: ArrayBuffer | Uint8Array) {
    const bytes = pcm instanceof Uint8Array ? pcm : new Uint8Array(pcm)
    const n = Math.floor(bytes.length / 2)
    if (n === 0) return
    const f = new Float32Array(n)
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.length)
    for (let i = 0; i < n; i++) f[i] = dv.getInt16(i * 2, true) / 32768

    this.pending = this.concat(this.pending, f)
    const ctx = this.ensureCtx()
    if (!ctx) return
    if (ctx.state === 'suspended') void ctx.resume()
    // Started already? schedule the next block(s) as data arrives —
    // otherwise playback stalls after the first MAX_SOURCE_SAMPLES.
    if (this.started) {
      this.scheduleBlock(ctx, MAX_SOURCE_SAMPLES)
    }
  }

  /**
   * Start playback with everything buffered so far (no-op if already
   * started). Call when MIN_START_SAMPLES is reached, or on stream end —
   * whichever comes first. Subsequent pushes are scheduled automatically
   * in MAX_SOURCE_SAMPLES blocks right after the previous ones.
   */
  start() {
    if (this.started) return
    const ctx = this.ensureCtx()
    if (!ctx || this.pending.length === 0) return
    this.started = true
    this.nextTime = ctx.currentTime + 0.08
    this.scheduleBlock(ctx, MAX_SOURCE_SAMPLES)
  }

  private scheduleBlock(ctx: AudioContext, maxSamples: number) {
    const now = ctx.currentTime
    // Never schedule into the past — sources firing late pile up and sum.
    if (this.nextTime < now) this.nextTime = now + 0.03

    while (this.pending.length > 0) {
      const take = Math.min(this.pending.length, maxSamples)
      const f = this.pending.slice(0, take)
      this.pending = this.pending.slice(take)
      if (f.length === 0) break

      const src = ctx.createBufferSource()
      this.lastSource = src
      const ab = ctx.createBuffer(1, f.length, SAMPLE_RATE)
      ab.copyToChannel(f as Float32Array<ArrayBuffer>, 0)
      src.buffer = ab
      src.connect(ctx.destination)
      src.start(this.nextTime)
      this.nextTime += f.length / SAMPLE_RATE

      src.onended = () => {
        src.disconnect()
        if (src === this.lastSource && this.ended && this.pending.length === 0) {
          this.started = false
          this.onended?.()
        }
      }
    }
  }

  private concat(a: Float32Array, b: Float32Array): Float32Array {
    const r = new Float32Array(a.length + b.length)
    r.set(a, 0)
    r.set(b, a.length)
    return r
  }

  /** Stream finished: flush what remains and arm the ended callback. */
  markEnded() {
    this.ended = true
    const ctx = this.ensureCtx()
    if (ctx) {
      if (!this.started) this.start()
      else this.scheduleBlock(ctx, MAX_SOURCE_SAMPLES) // drain anything left
    }
    if (!this.started) {
      this.onended?.()
    }
  }

  pause() {
    void this.ctx?.suspend()
  }

  resume() {
    void this.ctx?.resume()
  }

  stop() {
    try {
      this.ctx?.close()
    } catch {
      /* ignore */
    }
    this.ctx = null
    this.pending = new Float32Array(0)
    this.nextTime = 0
    this.started = false
    this.ended = false
    this.lastSource = null
  }

  buffered(): Float32Array {
    return this.pending
  }

  get hasWebAudio() {
    return !!this.ctx
  }
}
