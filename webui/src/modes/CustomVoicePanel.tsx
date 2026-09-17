import type { TtsSettings } from '../lib/types'
import { SPEAKERS } from '../lib/types'

interface Props {
  tts: TtsSettings
  setTts: (v: TtsSettings) => void
  onGenerate: () => void
}

const AVATARS: Record<string, string> = {
  serena: '👩🏻',
  vivian: '👩🏼',
  uncle_fu: '👨🏻🦳',
  ryan: '👨🏻',
  aiden: '👦🏻',
  ono_anna: '👩🏻🦰',
  sohee: '👧🏻',
  eric: '👨🏼',
  dylan: '🧑🏻',
}

const LANG: Record<string, string> = {
  serena: 'es · CL',
  vivian: 'zh · CN',
  uncle_fu: 'zh · CN',
  ryan: 'en · US',
  aiden: 'en · US',
  ono_anna: 'ja · JP',
  sohee: 'ko · KR',
  eric: 'en · US',
  dylan: 'en · UK',
}

/**
 * CustomVoice (0.6B): fixed speaker table read from the GGUF. The grid
 * mirrors the model's own speaker names + dialects, so this panel is
 * "graphically adapted" to the 0.6B family.
 */
export default function CustomVoicePanel({ tts, setTts, onGenerate }: Props) {
  return (
    <div className="panel">
      <div className="panel-title">
        <span>🗣️ Voces del modelo</span>
        <span className="badge">0.6B · speaker table</span>
      </div>

      <div className="settings-block">
        <div className="block-head">
          <span>Elige una voz fija</span>
        </div>
        <div className="speakers-grid">
          {SPEAKERS.map((s) => (
            <button
              key={s}
              className={'speaker-card' + (tts.speaker === s ? ' on' : '')}
              onClick={() => setTts({ ...tts, speaker: s })}
            >
              <span className="avatar">{AVATARS[s] ?? '🗣️'}</span>
              <span className="name">{s}</span>
              <span className="lang">{LANG[s] ?? ''}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="settings-block">
        <div className="block-head">
          <span>Semilla (determinismo)</span>
        </div>
        <input
          value={tts.seed}
          onChange={(e) => setTts({ ...tts, seed: e.target.value })}
          placeholder="vacío = aleatoria"
          type="number"
        />
        <p className="hint">Misma semilla + mismo texto = audio idéntico (usado en las pruebas de regresión).</p>
      </div>

      <div className="actions" style={{ marginTop: 10 }}>
        <button className="primary" onClick={onGenerate}>
          🔊 Sintetizar con {tts.speaker}
        </button>
      </div>
    </div>
  )
}
