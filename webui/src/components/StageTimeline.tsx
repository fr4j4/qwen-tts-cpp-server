import type { Stage, StageTiming } from '../lib/types'

interface Props {
  timing: StageTiming[]
  stage: Stage
  llmEnabled: boolean
}

const STAGES: { key: Stage; label: string; icon: string; enabled: (llm: boolean) => boolean }[] = [
  { key: 'llm', label: 'LLM procesando', icon: '🤖', enabled: (l) => l },
  { key: 'tts', label: 'Texto → TTS', icon: '📤', enabled: () => true },
  { key: 'audio', label: 'Stream de audio', icon: '🔊', enabled: () => true },
]

export default function StageTimeline({ timing, stage, llmEnabled }: Props) {
  const stages = STAGES.filter((s) => s.enabled(llmEnabled))
  const idx = (s: Stage) => timing.findIndex((t) => t.stage === s)
  const activeIdx = stages.findIndex((s) => s.key === stage)

  return (
    <div className="timeline panel">
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
                  {((t.at - (timing[timing.findIndex((x) => x.stage === 'llm') ?? 0]?.at ?? t.at)) / 1000).toFixed(2)}s
                  {t.detail ? ` · ${t.detail}` : ''}
                </div>
              )}
            </div>
            {i < stages.length - 1 && <div className="tl-arrow">→</div>}
          </div>
        )
      })}
    </div>
  )
}
