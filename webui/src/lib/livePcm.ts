// Growable live PCM buffer (Float32, normalized) that lives OUTSIDE React:
// chunks append cheaply on arrival; the waveform canvas reads the view at
// 60 fps via requestAnimationFrame — no React re-renders, no per-tick copies.

export class LivePcm {
  private buf = new Float32Array(24000 * 60) // 60 s initial capacity
  private len = 0

  /** Append raw little-endian s16 bytes (must be sample-aligned). */
  append(bytes: Uint8Array) {
    const n = Math.floor(bytes.length / 2)
    if (n === 0) return
    if (this.len + n > this.buf.length) this.grow(this.len + n)
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.length)
    for (let i = 0; i < n; i++) this.buf[this.len + i] = dv.getInt16(i * 2, true) / 32768
    this.len += n
  }

  private grow(need: number) {
    let cap = this.buf.length
    while (cap < need) cap *= 2
    const nb = new Float32Array(cap)
    nb.set(this.buf.subarray(0, this.len))
    this.buf = nb
  }

  reset() {
    this.len = 0
  }

  get length() {
    return this.len
  }

  /** Live view of the valid samples (no copy). */
  view(): Float32Array {
    return this.buf.subarray(0, this.len)
  }
}
