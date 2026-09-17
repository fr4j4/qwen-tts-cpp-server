import type { LlmSettings, TtsSettings } from '../lib/types'

interface Props {
  llm: LlmSettings
  setLlm: (v: LlmSettings) => void
  tts: TtsSettings
  setTts: (v: TtsSettings) => void
}

export default function SettingsPanel({ llm, setLlm, tts, setTts }: Props) {
  const u = (patch: Partial<LlmSettings>) => setLlm({ ...llm, ...patch })
  return (
    <details className="panel" open>
      <summary className="panel-title">⚙️ Conexiones</summary>

      <div className="settings-block">
        <div className="block-head">
          <span>🔊 TTS server</span>
          <span className="badge">local</span>
        </div>
        <label className="s">
          URL base
          <input value={tts.url} onChange={(e) => setTts({ ...tts, url: e.target.value })} placeholder="http://127.0.0.1:8898" />
        </label>
        <p className="hint">La app habla al mismo origen ({window.location.origin}); el proxy reenvía a esta URL — editable por si mueves el server.</p>
      </div>

      <div className="settings-block">
        <div className="block-head">
          <span>🤖 LLM (OpenAI-compatible)</span>
          <label className="switch">
            <input
              type="checkbox"
              checked={llm.enabled}
              onChange={(e) => u({ enabled: e.target.checked })}
            />
            <span>{llm.enabled ? 'ON' : 'OFF'}</span>
          </label>
        </div>
        {llm.enabled && (
          <>
            <label className="s">
              Endpoint base
              <input
                value={llm.baseUrl}
                onChange={(e) => u({ baseUrl: e.target.value })}
                placeholder="http://127.0.0.1:8081/v1"
              />
            </label>
            <label className="s">
              API key (opcional)
              <input
                type="password"
                value={llm.apiKey}
                onChange={(e) => u({ apiKey: e.target.value })}
                placeholder="sk-…"
              />
            </label>
            <label className="s">
              Modelo
              <input
                value={llm.model}
                onChange={(e) => u({ model: e.target.value })}
                placeholder="qwen2.5-7b-instruct"
              />
            </label>
            <label className="s">
              Max tokens
              <input
                type="number"
                min={16}
                max={1024}
                value={llm.maxTokens}
                onChange={(e) => u({ maxTokens: clampNum(e.target.value) })}
              />
            </label>
            <label className="s">
              Temperatura
              <input
                type="number"
                min={0}
                max={2}
                step={0.1}
                value={llm.temperature}
                onChange={(e) => u({ temperature: Math.max(0, Math.min(2, Number(e.target.value) || 0)) })}
              />
            </label>
            <label className="s">
              System prompt
              <textarea
                rows={3}
                value={llm.systemPrompt}
                onChange={(e) => u({ systemPrompt: e.target.value })}
              />
            </label>
            <p className="hint">El texto que escribes arriba se envía como prompt; la respuesta del LLM es lo que se sintetiza.</p>
          </>
        )}
      </div>
    </details>
  )
}

function clampNum(v: string): number {
  const n = Number(v)
  if (v === '') return 0
  return Math.max(1, Math.min(1024, Math.round(n || 0)))
}
