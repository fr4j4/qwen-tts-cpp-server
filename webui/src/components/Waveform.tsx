import { useEffect, useRef } from 'react'

interface Props {
  pcm: Uint8Array
}

/** Draws a live waveform of the accumulated PCM (s16le 24 kHz). */
export default function Waveform({ pcm }: Props) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const cv = ref.current
    if (!cv) return
    const ctx = cv.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const W = cv.clientWidth
    const H = cv.clientHeight
    cv.width = W * dpr
    cv.height = H * dpr
    ctx.scale(dpr, dpr)

    ctx.fillStyle = 'rgba(10,14,23,0.9)'
    ctx.fillRect(0, 0, W, H)

    const n = Math.floor(pcm.length / 2)
    if (n === 0) return

    // downsample to < 3000 buckets
    const buckets = Math.min(3000, Math.max(64, W / 2))
    const step = Math.max(1, Math.floor(n / buckets))
    const dv = new DataView(pcm.buffer, pcm.byteOffset, pcm.length)
    const mid = H / 2
    ctx.strokeStyle = '#38bdf8'
    ctx.lineWidth = 1

    let lastX = 0
    let lastY = mid
    ctx.beginPath()
    ctx.moveTo(0, mid)
    for (let b = 0; b < n; b += step) {
      const v = dv.getInt16(b * 2, true)
      const x = (b / n) * W
      const y = mid - (v / 32768) * (H * 0.45)
      ctx.lineTo(x, y)
      lastX = x
      lastY = y
    }
    ctx.stroke()

    // progress fill
    ctx.strokeStyle = '#0ea5e9'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.moveTo(0, mid)
    ctx.lineTo(lastX, lastY)
    ctx.stroke()
  }, [pcm])

  return (
    <div className="waveform-wrap">
      <canvas ref={ref} className="waveform" style={{ height: 110 }} />
    </div>
  )
}
