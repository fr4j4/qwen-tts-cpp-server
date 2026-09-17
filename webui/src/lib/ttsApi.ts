// TTS server client (same-origin proxy /proxy/tts).
import type { Mode, SynthEvent } from './types'

export interface TtsOptions {
  mode: Mode
  speaker: string
  instruct: string
  seed: string
  /** TTS server base URL (same-origin proxy forwards to it via ?upstream=) */
  upstream: string
  onEvent: (e: SynthEvent) => void
  signal: AbortSignal
}

export interface TtsResult {
  /** total s16 samples */
  samples: number
  pcmBytes: Uint8Array
  /** object URL of a wav-converted blob; revoke when done */
  wavUrl: string
}

const SAMPLE_RATE = 24000

function buildPayload(o: TtsOptions, text: string): Record<string, unknown> {
  if (o.mode === 'voicedesign') {
    return { input: text, instructions: o.instruct, response_format: 'pcm' }
  }
  return { input: text, voice: o.speaker, response_format: 'pcm' }
}

/** s16le PCM bytes -> wav blob object URL */
export function pcmToWavUrl(bytes: Uint8Array, sampleRate = SAMPLE_RATE): string {
  const n = bytes.length
  const buf = new ArrayBuffer(44 + n)
  const dv = new DataView(buf)
  const w = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i))
  }
  w(0, 'RIFF')
  dv.setUint32(4, 36 + n, true)
  w(8, 'WAVE')
  w(12, 'fmt ')
  dv.setUint32(16, 16, true)
  dv.setUint16(20, 1, true)
  dv.setUint16(22, 1, true)
  dv.setUint32(24, sampleRate, true)
  dv.setUint32(28, sampleRate * 2, true)
  dv.setUint16(32, 2, true)
  dv.setUint16(34, 16, true)
  w(36, 'data')
  dv.setUint32(40, n, true)
  new Uint8Array(buf, 44).set(bytes)
  return URL.createObjectURL(new Blob([buf], { type: 'audio/wav' }))
}

/**
 * Streams raw PCM (s16le 24 kHz mono) from the tts-server. The server
 * flushes frames as they are produced (real-time). onEvent('pcm') fires
 * per network chunk so the UI can feed the player live; 'end' carries
 * the total sample count. Resolves with full PCM + wav URL at the end.
 */
export async function streamTts(text: string, o: TtsOptions): Promise<TtsResult> {
  const payload = buildPayload(o, text)
  if (o.seed) payload.seed = Number(o.seed)

  const q = o.upstream ? `?upstream=${encodeURIComponent(o.upstream)}` : ''
  const res = await fetch(`/proxy/tts/v1/audio/speech${q}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
    signal: o.signal,
  })
  if (!res.ok) {
    let detail = ''
    try {
      detail = (await res.text()).slice(0, 400)
    } catch {
      /* no body */
    }
    o.onEvent({ kind: 'error', error: `TTS HTTP ${res.status}: ${detail || res.statusText}` })
    throw new Error(`TTS HTTP ${res.status}`)
  }
  if (!res.body) {
    o.onEvent({ kind: 'error', error: 'TTS: sin body' })
    throw new Error('TTS: sin body')
  }

  const reader = res.body.getReader()
  const chunks: Uint8Array[] = []
  let total = 0
  // CRÍTICO: los chunks de red pueden terminar ENTRE las 2 bytes de una
  // muestra s16. Si se descarta el byte impar (floor(len/2) por chunk),
  // todas las muestras siguientes quedan desalineadas = ruido blanco.
  // Se acumula el byte sobrante en `carry` y se emiten solo chunks
  // alineados a muestras completas.
  let carry: Uint8Array = new Uint8Array(0)

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    if (!value || value.length === 0) continue

    let buf = value
    if (carry.length > 0) {
      buf = new Uint8Array(carry.length + value.length)
      buf.set(carry, 0)
      buf.set(value, carry.length)
    }
    const usable = buf.length - (buf.length % 2)
    if (usable === 0) {
      carry = buf.slice()
      continue
    }
    const aligned = usable === buf.length ? buf : buf.slice(0, usable)
    chunks.push(aligned)
    total += usable
    // entrega el chunk ALINEADO para playback en vivo
    o.onEvent({ kind: 'pcm', bytes: aligned, count: Math.floor(total / 2) })
    carry = usable < buf.length ? buf.slice(usable) : new Uint8Array(0)
  }
  // (si queda 1 byte suelto al final, es media muestra — inaudible, se omite)

  const pcm = new Uint8Array(total)
  let off = 0
  for (const c of chunks) {
    pcm.set(c, off)
    off += c.length
  }
  const samples = Math.floor(total / 2)
  o.onEvent({ kind: 'end', count: samples })
  return { samples, pcmBytes: pcm, wavUrl: pcmToWavUrl(pcm) }
}

export async function health(upstream?: string): Promise<boolean> {
  try {
    const q = upstream ? `?upstream=${encodeURIComponent(upstream)}` : ''
    const r = await fetch(`/proxy/health${q}`, { signal: AbortSignal.timeout(3000) })
    return r.ok
  } catch {
    return false
  }
}
