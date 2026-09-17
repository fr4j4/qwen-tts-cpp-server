interface Props {
  playing: boolean
  setPlaying: (v: boolean) => void
  wavUrl: string | null
  frames: number
  onStop: () => void
  /** Web Audio: pause/resume the live player */
  onPauseToggle: () => void
}

export default function AudioControls({ playing, setPlaying, wavUrl, frames, onStop, onPauseToggle }: Props) {
  if (frames === 0) {
    return (
      <div className="audio-controls idle">
        <span className="badge">Aún sin audio — genera una síntesis.</span>
      </div>
    )
  }
  return (
    <div className="audio-controls">
      <button
        className={'primary' + (playing ? ' dim' : '')}
        onClick={() => {
          onPauseToggle()
          setPlaying(!playing)
        }}
        disabled={!wavUrl && !playing}
      >
        {playing ? '⏸ Pausar' : '▶ Reproducir'}
      </button>
      <button className="ghost" onClick={onStop}>
        ⏹ Detener
      </button>
      <button
        className="ghost"
        disabled={!wavUrl}
        onClick={() => {
          const a = document.createElement('a')
          a.href = wavUrl!
          a.download = `qwen-tts-${Date.now()}.wav`
          a.click()
        }}
      >
        💾 Descargar WAV
      </button>
      <span className="badge">{frames} frames · {(frames / 100).toFixed(2)}s</span>
    </div>
  )
}
