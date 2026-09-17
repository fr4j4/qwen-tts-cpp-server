// Web Audio PCM player: plays s16le 24 kHz mono chunks as they arrive
// (real-time streaming). Falls back to buffered playback if AudioContext
// is unavailable.

const SAMPLE_RATE = 24000

export class PcmPlayer {
  private ctx: AudioContext | null = null
  private buf: Float32Array = new Float32Array(0)
  private playing = false
  private queue: Float32Array<ArrayBuffer>[] = []
  private ended = false
  private onended: (() => void) | null = null

  get endedFlag() {
    return this.ended
  }

  setOnEnded(fn: (() => void) | null) {
    this.onended = fn
  }

  private ensureCtx(): AudioContext | null {
    if (this.ctx) return this.ctx
    const AC =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!AC) return null
    this.ctx = new AC()
    return this.ctx
  }

  /** Feed raw little-endian s16 PCM bytes. */
  push(pcm: ArrayBuffer | Uint8Array) {
    const bytes = pcm instanceof Uint8Array ? pcm : new Uint8Array(pcm)
    const n = Math.floor(bytes.length / 2)
    const f = new Float32Array(n)
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.length)
    for (let i = 0; i < n; i++) {
      f[i] = dv.getInt16(i * 2, true) / 32768
    }
    this.queue.push(f)
    this.maybeStart()
  }

  private maybeStart() {
    if (this.playing || this.queue.length === 0) return
    const ctx = this.ensureCtx()
    if (!ctx) {
      // No Web Audio: accumulate into a flat buffer for later play.
      this.buf = this.concat(this.buf, this.queue.shift()!)
      return
    }
    this.playing = true
    if (ctx.state === 'suspended') void ctx.resume()
    this.pump(ctx)
  }

  private pump(ctx: AudioContext) {
    const src = ctx.createBufferSource()
    const chunk: Float32Array<ArrayBuffer> = this.queue.shift() ?? new Float32Array(0)
    if (chunk.length > 0) {
      const ab = ctx.createBuffer(1, chunk.length, SAMPLE_RATE)
      ab.copyToChannel(chunk, 0)
      src.buffer = ab
    }
    src.connect(ctx.destination)
    src.onended = () => {
      src.disconnect()
      if (this.queue.length > 0) {
        this.pump(ctx)
      } else if (this.ended) {
        this.playing = false
        this.onended?.()
      } else {
        this.playing = false
      }
    }
    const t = ctx.currentTime + 0.01
    src.start(t)
  }

  private concat(a: Float32Array, b: Float32Array): Float32Array {
    const r = new Float32Array(a.length + b.length)
    r.set(a, 0)
    r.set(b, a.length)
    return r
  }

  markEnded() {
    this.ended = true
    if (!this.playing && this.queue.length === 0 && this.ctx) {
      this.onended?.()
    }
  }

  /** Full buffered audio (only when Web Audio fell back). */
  buffered(): Float32Array {
    return this.buf
  }

  get hasWebAudio() {
    return !!this.ctx
  }

  stop() {
    try {
      this.ctx?.close()
    } catch {
      /* ignore */
    }
    this.ctx = null
    this.playing = false
    this.queue = []
    this.ended = false
  }
}
