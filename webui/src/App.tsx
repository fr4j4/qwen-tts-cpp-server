import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import Header from './components/Header'
import SettingsPanel from './components/SettingsPanel'
import StageTimeline from './components/StageTimeline'
import Waveform from './components/Waveform'
import AudioControls from './components/AudioControls'
import VoiceDesignPanel from './modes/VoiceDesignPanel'
import CustomVoicePanel from './modes/CustomVoicePanel'
import { streamChat } from './lib/llmApi'
import { streamTts } from './lib/ttsApi'
import { PcmPlayer } from './lib/pcmPlayer'
import { loadLlm, loadTts, saveLlm, saveTts } from './lib/storage'
import type { LlmSettings, Mode, TtsSettings, UiState } from './lib/types'
import { SAMPLE_TEXTS } from './lib/types'

const initialTts: TtsSettings = {
  mode: 'voicedesign',
  url: 'http://127.0.0.1:8898',
  speaker: 'serena',
  seed: '',
  instruct: 'Natural y tranquila',
}

const initialLlm: LlmSettings = {
  enabled: false,
  baseUrl: '',
  apiKey: '',
  model: '',
  systemPrompt:
    'Eres un asistente de voz. Responde en una frase breve (máx. 2 frases), natural y clara, como voz hablada. No uses markdown.',
  maxTokens: 120,
  temperature: 0.8,
}

const initialUi: UiState = {
  phase: 'idle',
  stage: 'idle',
  llmText: '',
  llmDone: false,
  synthText: '',
  frames: 0,
  timing: [],
  error: null,
}

type UiAction =
  | { type: 'reset' }
  | { type: 'start' }
  | { type: 'llm_tick'; text: string }
  | { type: 'llm_done' }
  | { type: 'tts_start'; text: string }
  | { type: 'audio_tick'; frames: number }
  | { type: 'finish' }
  | { type: 'fail'; error: string }

function reducer(s: UiState, a: UiAction): UiState {
  switch (a.type) {
    case 'reset':
      return { ...initialUi }
    case 'start':
      return {
        ...initialUi,
        phase: 'generating',
        timing: [{ stage: 'llm', at: performance.now(), detail: 'inicio' }],
      }
    case 'llm_tick':
      return { ...s, llmText: a.text, llmDone: false }
    case 'llm_done':
      return { ...s, llmDone: true }
    case 'tts_start':
      return {
        ...s,
        stage: 'tts',
        synthText: a.text,
        timing: [...s.timing, { stage: 'tts', at: performance.now(), detail: a.text.slice(0, 60) + (a.text.length > 60 ? '…' : '') }],
      }
    case 'audio_tick': {
      let timing = s.timing
      if (s.stage !== 'audio') {
        timing = [...s.timing, { stage: 'audio', at: performance.now(), detail: 'primer chunk PCM' }]
      }
      return { ...s, stage: 'audio', frames: a.frames, timing }
    }
    case 'finish':
      return { ...s, phase: 'done', stage: 'audio' }
    case 'fail':
      return { ...s, phase: 'error', stage: 'idle', error: a.error }
    default:
      return s
  }
}

export default function App() {
  const [tts, setTts] = useState<TtsSettings>(() =>
    loadTts({ ...initialTts })
  )
  const [llm, setLlm] = useState<LlmSettings>(() => loadLlm({ ...initialLlm }))
  const [text, setText] = useState(SAMPLE_TEXTS[0])
  const [ui, dispatch] = useReducer(reducer, initialUi)

  const abortRef = useRef<AbortController | null>(null)
  const pcmRef = useRef<Uint8Array>(new Uint8Array(0))
  const playerRef = useRef<PcmPlayer | null>(null)
  const [playing, setPlaying] = useState(false)
  const [wavUrl, setWavUrl] = useState<string | null>(null)
  const [pcmDuration, setPcmDuration] = useState(0)
  const llmRef = useRef(llm)
  useEffect(() => {
    llmRef.current = llm
  }, [llm])

  useEffect(() => saveTts(tts), [tts])
  useEffect(() => saveLlm(llm), [llm])

  const run = useCallback(async () => {
    if (ui.phase === 'generating') return
    playerRef.current?.stop()
    setPlaying(false)
    setWavUrl(null)
    pcmRef.current = new Uint8Array(0)
    setPcmDuration(0)
    dispatch({ type: 'start' })

    const ctrl = new AbortController()
    abortRef.current = ctrl
    let final = text

    try {
      if (llmRef.current.enabled) {
        let acc = ''
        for await (const d of streamChat(llmRef.current, text, (sig) => {
          sig.addEventListener('abort', () => ctrl.abort())
        })) {
          if (ctrl.signal.aborted) throw new Error('cancelado')
          if (d.kind === 'chunk') {
            acc += d.text ?? ''
            dispatch({ type: 'llm_tick', text: acc })
          } else if (d.kind === 'error') {
            throw new Error(d.error ?? 'LLM error')
          }
        }
        if (!acc.trim()) throw new Error('LLM no generó texto')
        dispatch({ type: 'llm_done' })
        final = acc
      }

      dispatch({ type: 'tts_start', text: final })

      const result = await streamTts(final, {
        mode: tts.mode,
        speaker: tts.speaker,
        instruct: tts.instruct,
        seed: tts.seed,
        upstream: tts.url,
        signal: ctrl.signal,
        onEvent: (e) => {
          if (e.kind === 'frame') {
            dispatch({ type: 'audio_tick', frames: e.count })
          } else if (e.kind === 'error' && e.error) {
            dispatch({ type: 'fail', error: e.error })
          }
        },
      })
      pcmRef.current = result.pcmBytes
      setPcmDuration(result.frames / 100)
      setWavUrl(result.wavUrl)
      dispatch({ type: 'finish' })

      // Playback: feed the player progressively (real-time feel).
      const player = new PcmPlayer()
      playerRef.current = player
      player.setOnEnded(() => setPlaying(false))
      // Replay from the accumulated buffer.
      const chunk = 4096
      for (let i = 0; i < result.pcmBytes.length; i += chunk) {
        player.push(result.pcmBytes.slice(i, i + chunk))
      }
      player.markEnded()
      setPlaying(true)
    } catch (e) {
      if (!ctrl.signal.aborted) {
        dispatch({ type: 'fail', error: (e as Error).message || String(e) })
      } else {
        dispatch({ type: 'reset' })
      }
    }
  }, [text, tts.mode, tts.speaker, tts.seed, ui.phase])

  const stop = useCallback(() => {
    abortRef.current?.abort()
    playerRef.current?.stop()
    setPlaying(false)
    dispatch({ type: 'reset' })
  }, [])

  const setMode = (m: Mode) => setTts((p) => ({ ...p, mode: m }))

  const modeProps = {
    tts,
    setTts,
    onGenerate: () => void run(),
  }

  return (
    <div className="app">
      <Header healthUrl={tts.url} />
      <div className="layout">
        <aside className="side">
          <SettingsPanel llm={llm} setLlm={setLlm} tts={tts} setTts={setTts} />
          <div className="tabs">
            <button
              className={'tab' + (tts.mode === 'voicedesign' ? ' active' : '')}
              onClick={() => setMode('voicedesign')}
              title="Modelo 1.7B VoiceDesign — instrucción de estilo"
            >
              🎭 VoiceDesign
            </button>
            <button
              className={'tab' + (tts.mode === 'customvoice' ? ' active' : '')}
              onClick={() => setMode('customvoice')}
              title="Modelo 0.6B CustomVoice — voces fijas"
            >
              🗣️ CustomVoice
            </button>
          </div>
          {tts.mode === 'voicedesign' ? <VoiceDesignPanel {...modeProps} /> : <CustomVoicePanel {...modeProps} />}

          <div className="field">
            <label>Texto (o prompt para el LLM)</label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={6}
              placeholder="Escribe el texto a sintetizar…"
            />
            <div className="samples">
              {SAMPLE_TEXTS.map((s) => (
                <button key={s} className="chip" onClick={() => setText(s)}>
                  {s.slice(0, 30)}…
                </button>
              ))}
            </div>
          </div>

          <div className="actions">
            <button
              className="primary"
              disabled={ui.phase === 'generating' || !text.trim()}
              onClick={() => void run()}
            >
              {ui.phase === 'generating' ? '⏳ Generando…' : llm.enabled ? '⚡ LLM → TTS' : '🔊 Generar audio'}
            </button>
            {ui.phase === 'generating' && (
              <button className="danger" onClick={stop}>
                ✋ Detener
              </button>
            )}
          </div>
        </aside>

        <main className="main">
          <StageTimeline timing={ui.timing} stage={ui.stage} llmEnabled={llm.enabled} />
          {ui.error && <div className="alert error">{ui.error}</div>}
          {llm.enabled && (ui.llmText || ui.phase !== 'idle') && (
            <div className="panel llm-out">
              <div className="panel-title">
                <span>🤖 Salida del LLM</span>
                <span className="badge">{ui.llmDone ? '✓ completo' : ui.llmText ? '… generando' : 'esperando'}</span>
              </div>
              <div className="llm-text">{ui.llmText || '—'}</div>
            </div>
          )}
          <div className="panel synth-out">
            <div className="panel-title">
              <span>🎙️ Texto enviado al TTS</span>
              {ui.synthText && (
                <span className="badge">{ui.frames} frames · {pcmDuration.toFixed(2)} s audio</span>
              )}
            </div>
            <div className="llm-text">{ui.synthText || (ui.phase === 'generating' && !llm.enabled ? text : '—')}</div>
          </div>
          {ui.frames > 0 && (
            <div className="panel">
              <Waveform pcm={pcmRef.current} />
            </div>
          )}
          <AudioControls
            playing={playing}
            setPlaying={setPlaying}
            wavUrl={wavUrl}
            frames={ui.frames}
            onStop={() => playerRef.current?.stop()}
          />
        </main>
      </div>
    </div>
  )
}
