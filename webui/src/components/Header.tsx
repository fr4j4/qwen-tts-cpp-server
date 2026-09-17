import { useEffect, useState } from 'react'
import { health } from '../lib/ttsApi'

export default function Header({ healthUrl }: { healthUrl: string }) {
  const [ok, setOk] = useState<boolean | null>(null)
  const upstream = healthUrl

  useEffect(() => {
    let alive = true
    const t = setInterval(async () => {
      const h = await health(upstream)
      if (alive) setOk(h)
    }, 4000)
    void health(upstream).then((h) => alive && setOk(h))
    return () => {
      alive = false
      clearInterval(t)
    }
  }, [upstream])

  return (
    <header className="header">
      <div className="brand">
        <span className="logo">🎙️</span>
        <div>
          <h1>Qwen TTS Lab</h1>
          <p className="sub">Motor local qwen-tts-cpp · real-time pipeline lab</p>
        </div>
      </div>
      <div className="health">
        <span className={'dot ' + (ok === null ? 'unknown' : ok ? 'up' : 'down')} />
        {ok === null ? '…' : ok ? `TTS server OK (${healthUrl})` : 'TTS server NO disponible'}
      </div>
    </header>
  )
}
