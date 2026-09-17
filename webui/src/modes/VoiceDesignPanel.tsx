import { useState } from 'react'
import type { TtsSettings } from '../lib/types'
import { STYLE_CHIPS } from '../lib/types'

interface Props {
  tts: TtsSettings
  setTts: (v: TtsSettings) => void
  onGenerate: () => void
}

/**
 * VoiceDesign (1.7B): the model needs a style instruction. The panel
 * offers quick style chips + a free text area, plus a few demo
 * instructions to try.
 */
export default function VoiceDesignPanel({ tts, setTts, onGenerate }: Props) {
  // demo instructions live in the text field only when the user picks one?
  // Keep the instruct state separate.
  const [instruct, setInstruct] = useState(tts.instruct ?? '')
  const [demo, setDemo] = useState(0)

  const applyInstruct = (v: string) => {
    setInstruct(v)
    setTts({ ...tts, instruct: v })
  }

  const samples = [
    'Narración de cuento para dormir a un niño',
    'Noticiero serio de las 9 de la noche',
    'Vendedor entusiasta en una feria',
    'Poesía susurrada, lenta y emotiva',
  ]

  return (
    <div className="panel">
      <div className="panel-title">
        <span>🎭 Dirección de voz</span>
        <span className="badge">1.7B · instrucción</span>
      </div>

      <div className="settings-block">
        <div className="block-head">
          <span>Estilo rápido</span>
        </div>
        <div className="chips">
          {STYLE_CHIPS.map((c) => (
            <button
              key={c}
              className={'chip' + (instruct === c ? ' on' : '')}
              onClick={() => applyInstruct(instruct === c ? '' : c)}
            >
              {c}
            </button>
          ))}
        </div>
      </div>

      <div className="settings-block">
        <div className="block-head">
          <span>Instrucción libre</span>
          <button
            className="chip"
            onClick={() => {
              setDemo((d) => (d + 1) % samples.length)
              applyInstruct(samples[demo])
            }}
            title="Insertar ejemplo"
          >
            ✨ ejemplo
          </button>
        </div>
        <textarea
          rows={3}
          value={instruct}
          onChange={(e) => applyInstruct(e.target.value)}
          placeholder="Ej: Narración de cuento para dormir a un niño…"
        />
        {instruct.trim() === '' && (
          <p className="hint">⚠️ VoiceDesign requiere una instrucción de estilo — el server rechaza síntesis sin ella.</p>
        )}
      </div>

      <div className="actions" style={{ marginTop: 10 }}>
        <button className="primary" onClick={onGenerate} disabled={instruct.trim() === ''}>
          🔊 Sintetizar con esta dirección
        </button>
      </div>
    </div>
  )
}
