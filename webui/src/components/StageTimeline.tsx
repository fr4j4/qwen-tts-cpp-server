import type { Stage, StageTiming } from '../lib/types'

interface Props {
  timing: StageTiming[]
  stage: Stage
  llmEnabled: boolean
  /** ms from TTS start to first audio chunk (real-time indicator) */
  ttfaMs: number | null
  /** ms from TTS start to stream end */
  totalMs: number | null
}

const STAGES: { key: Stage; label: string; icon: string; enabled: (llm: boolean) => boolean }[] = [
  { key: 'llm', label: 'LLM procesando', icon: '🤖', enabled: (l) => l },
  { key: 'tts', label: 'Texto → TTS', icon: '📤', enabled: () => true },
  { key: 'audio', label: 'Stream de audio', icon: '🔊', enabled: () => true },
]

export default function StageTimeline({ timing, stage, llmEnabled, ttfaMs, totalMs }: Props) {
  const stages = STAGES.filter((s) => s.enabled(llmEnabled))
  const idx = (s: Stage) => timing.findIndex((t) => t.stage === s)
  const activeIdx = stages.findIndex((s) => s.key === stage)
  const baseAt = timing.find((t) => t.stage === 'llm')?.at ?? timing[0]?.at ?? 0

  return (
    <div className="panel">
      <div className="timeline">
        {stages.map((s, i) => {
          const t = timing[idx(s.key)]
          const state =
            t || (llmEnabled ? i < activeIdx : i <= activeIdx)
              ? 'done'
              : i === activeIdx
                ? 'active'
                : 'todo'
          return (
            <div key={s.key} className={'tl-step ' + state}>
              <div className="tl-icon">{state === 'done' ? '✅' : state === 'active' ? '⏳' : s.icon}</div>
              <div className="tl-info">
                <div className="tl-label">{s.label}</div>
                {t && (
                  <div className="tl-time">
                    {((t.at - baseAt) / 1000).toFixed(2)}s
                    {t.detail ? ` · ${t.detail}` : ''}
                  </div>
                )}
              </div>
              {i < stages.length - 1 && <div className="tl-arrow">→</div>}
            </div>
          )
        })}
      </div>
      <div className="rt-metrics">
        {ttfaMs !== null && (
          <span className="metric">
            <span className="metric-label">TTFA (1er audio)</span>
            <span className="metric-value">{ttfaMs} ms</span>
          </span>
        )}
        {totalMs !== null && (
          <span className="metric">
            <span className="metric-label">Stream completo</span>
            <span className="metric-value">{totalMs} ms</span>
          </span>
        )}
        {ttfaMs !== null && ttfaMs > 1000 && (
          <span className="metric warn">⚠️ &gt;1 s de primer audio — revisa carga GPU o upstream</span>
        )}
        {ttfaMs !== null && ttfaMs <= 1000 && (
          <span className="metric ok">real-time ✔</span>
        )}
      </div>
    </div>
  )
}
