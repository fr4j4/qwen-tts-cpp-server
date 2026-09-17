import { useEffect, useRef } from 'react'
import type { LivePcm } from '../lib/livePcm'

interface Props {
  /** Live buffer: redraws at 60 fps via rAF while it grows. */
  liveBuffer?: LivePcm | null
  /** Static fallback (finished stream / stable mode snapshot). */
  pcm?: Uint8Array
}

const BgColor = 'rgba(10,14,23,0.9)'
const WaveColor = '#38bdf8'
const MidColor = 'rgba(56, 189, 248, 0.25)'
const CursorColor = '#e2f4ff'

function toFloat(pcm: Uint8Array): Float32Array {
  const n = Math.floor(pcm.length / 2)
  const f = new Float32Array(n)
  const dv = new DataView(pcm.buffer, pcm.byteOffset, pcm.length)
  for (let i = 0; i < n; i++) f[i] = dv.getInt16(i * 2, true) / 32768
  return f
}

/** Per-pixel min/max envelope; strided reads keep it O(cols) per frame. */
function drawEnvelope(
  ctx: CanvasRenderingContext2D,
  cv: HTMLCanvasElement,
  data: Float32Array
): { x: number } {
  const dpr = window.devicePixelRatio || 1
  const W = cv.clientWidth
  const H = cv.clientHeight
  if (cv.width !== W * dpr || cv.height !== H * dpr) {
    cv.width = W * dpr
    cv.height = H * dpr
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.fillStyle = BgColor
  ctx.fillRect(0, 0, W, H)

  const mid = H / 2
  ctx.fillStyle = MidColor
  ctx.fillRect(0, mid - 0.5, W, 1)

  const n = data.length
  if (n === 0) return { x: 0 }

  const cols = Math.max(1, Math.floor(W))
  const per = n / cols
  ctx.fillStyle = WaveColor
  for (let x = 0; x < cols; x++) {
    const s = Math.floor(x * per)
    const e = Math.min(n, Math.floor((x + 1) * per) + 1)
    if (e <= s) continue
    const step = Math.max(1, Math.floor((e - s) / 6))
    let mn = 1
    let mx = -1
    for (let i = s; i < e; i += step) {
      const v = data[i]
      if (v < mn) mn = v
      if (v > mx) mx = v
    }
    const y1 = mid - mx * (H * 0.47)
    const y2 = mid - mn * (H * 0.47)
    ctx.fillRect(x, y1, 1, Math.max(1, y2 - y1))
  }
  // x del borde derecho (cursor de crecimiento)
  return { x: W }
}

/**
 * Waveform canvas: fluid (60 fps rAF) with a liveBuffer. The REVEALED
 * length eases toward the real buffer length, so the wave grows as a
 * continuous sweep — decoupled from the discrete network chunk arrivals.
 */
export default function Waveform({ liveBuffer, pcm }: Props) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const cv = ref.current
    if (!cv) return
    const ctx = cv.getContext('2d')
    if (!ctx) return

    let raf = 0
    let stopped = false
    let displayLen = 0 // muestras "reveladas" (eased)
    let lastT = performance.now()

    const draw = (t: number) => {
      if (stopped) return
      const dt = Math.min(0.1, (t - lastT) / 1000)
      lastT = t

      let data: Float32Array = new Float32Array(0)
      let cursorX = 0
      let totalLen = 0

      if (liveBuffer) {
        const target = liveBuffer.length
        if (displayLen > target) displayLen = target // reset entre generaciones
        const gap = target - displayLen
        // Chase continuo: converge ~90% en ~250 ms — el borde avanza
        // como barrido suave aunque los chunks lleguen en bocados.
        if (gap < 120) displayLen = target
        else displayLen += gap * Math.min(1, dt * 12)
        totalLen = target
        data = liveBuffer.view().subarray(0, Math.floor(displayLen))
      } else if (pcm) {
        data = toFloat(pcm)
        displayLen = data.length
        totalLen = data.length
      }

      const W = cv.clientWidth || 1
      drawEnvelope(ctx, cv, data)
      cursorX = totalLen > 0 ? (data.length / totalLen) * W : 0
      // cursor de crecimiento (solo en vivo)
      if (liveBuffer) {
        ctx.fillStyle = CursorColor
        ctx.fillRect(Math.min(W - 2, cursorX), 0, 2, cv.clientHeight)
        raf = requestAnimationFrame(draw)
      }
    }

    raf = requestAnimationFrame(draw)
    return () => {
      stopped = true
      cancelAnimationFrame(raf)
    }
  }, [liveBuffer, pcm])

  return (
    <div className="waveform-wrap">
      <canvas ref={ref} className="waveform" style={{ height: 120 }} />
    </div>
  )
}
