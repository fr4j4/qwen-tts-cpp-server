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
import { PcmPlayer, MIN_START_SAMPLES } from './lib/pcmPlayer'
import { LivePcm } from './lib/livePcm'
import { loadLlm, loadTts, saveLlm, saveTts } from './lib/storage'
import type { LlmSettings, Mode, TtsSettings, UiState } from './lib/types'
import { SAMPLE_TEXTS } from './lib/types'

const SAMPLE_RATE = 24000

const initialTts: TtsSettings = {
  mode: 'voicedesign',
  url: 'http://127.0.0.1:8898',
  speaker: 'serena',
  seed: '',
  instruct: 'Natural y tranquila',
  playback: 'live',
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
  ttfaMs: null,
  totalMs: null,
  error: null,
}

type UiAction =
  | { type: 'reset' }
  | { type: 'start' }
  | { type: 'llm_tick'; text: string }
  | { type: 'llm_done' }
  | { type: 'tts_start'; text: string }
  | { type: 'audio_tick'; frames: number }
  | { type: 'finish'; totalMs: number }
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
      let ttfa = s.ttfaMs
      if (s.stage !== 'audio') {
        timing = [...s.timing, { stage: 'audio', at: performance.now(), detail: 'primer chunk PCM' }]
        ttfa = Math.round(performance.now() - (timing.find((t) => t.stage === 'tts')?.at ?? performance.now()))
      }
      return { ...s, stage: 'audio', frames: a.frames, timing, ttfaMs: ttfa }
    }
    case 'finish':
      return { ...s, phase: 'done', stage: 'audio', totalMs: a.totalMs }
    case 'fail':
      return { ...s, phase: 'error', stage: 'idle', error: a.error }
    default:
      return s
  }
}

export default function App() {
  const [tts, setTts] = useState<TtsSettings>(() => {
    const loaded = loadTts({ ...initialTts })
    // Migración one-time: 'stable' se repartió como default durante el
    // workaround del ruido (causa real: desalineación PCM, ya arreglada).
    // El realtime es el comportamiento intended — se restaura una vez.
    const anyLoaded = loaded as TtsSettings & { migratedLive?: boolean }
    if (!anyLoaded.migratedLive) {
      anyLoaded.playback = 'live'
      anyLoaded.migratedLive = true
    }
    return anyLoaded
  })
  const [llm, setLlm] = useState<LlmSettings>(() => loadLlm({ ...initialLlm }))
  const [text, setText] = useState(SAMPLE_TEXTS[0])
  const [ui, dispatch] = useReducer(reducer, initialUi)

  const abortRef = useRef<AbortController | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const pcmRef = useRef<Uint8Array>(new Uint8Array(0))
  const livePcmRef = useRef<LivePcm>(new LivePcm())
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
    livePcmRef.current.reset()
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
      const ttsT0 = performance.now()

      // Two playback paths:
      //  - 'live':  Web Audio buffer-first (starts at ~2.5 s buffered;
      //             experimental — degrades on some Linux/Pulse setups)
      //  - 'stable': no Web Audio at all; when the stream ends, the same
      //             wav the download serves is played via a native
      //             <audio> element (the path already proven clean).
      const stable = tts.playback !== 'live'
      const player = stable ? null : new PcmPlayer()
      if (player) playerRef.current = player
      let lastUi = 0

      const result = await streamTts(final, {
        mode: tts.mode,
        speaker: tts.speaker,
        instruct: tts.instruct,
        seed: tts.seed,
        upstream: tts.url,
        signal: ctrl.signal,
        onEvent: (e) => {
          if (e.kind === 'pcm' && e.bytes) {
            player?.push(e.bytes)
            // Waveform fluida: append O(n) al buffer vivo (sin copias ni
            // re-renders); el canvas se redibuja solo a 60 fps vía rAF.
            livePcmRef.current.append(e.bytes)
            // Start once the first ~1 s of audio are buffered.
            if (player && !player.isStarted && (e.count ?? 0) >= MIN_START_SAMPLES) {
              player.start()
            }
            const now = performance.now()
            if (now - lastUi > 200 || lastUi === 0) {
              lastUi = now
              if (!stable) setPlaying(true)
              dispatch({ type: 'audio_tick', frames: e.count ?? 0 })
            }
          } else if (e.kind === 'error' && e.error) {
            dispatch({ type: 'fail', error: e.error })
          }
        },
      })

      player?.markEnded() // drains whatever accumulated below 2.5 s
      player?.setOnEnded(() => setPlaying(false))
      pcmRef.current = result.pcmBytes
      setPcmDuration(result.samples / SAMPLE_RATE)
      setWavUrl(result.wavUrl)
      dispatch({ type: 'finish', totalMs: Math.round(performance.now() - ttsT0) })

      if (stable) {
        // Proven path: native <audio> on the finished wav blob.
        const audio = audioRef.current
        if (audio) {
          audio.src = result.wavUrl
          try {
            await audio.play()
            setPlaying(true)
            audio.onended = () => setPlaying(false)
          } catch {
            /* autoplay block: user presses Reproducir */
          }
        }
      } else if (result.samples > 0 && !player!.hasWebAudio) {
        setPlaying(true)
      }
    } catch (e) {
      if (!ctrl.signal.aborted) {
        dispatch({ type: 'fail', error: (e as Error).message || String(e) })
      } else {
        dispatch({ type: 'reset' })
      }
    }
  }, [text, tts.mode, tts.speaker, tts.instruct, tts.seed, tts.url, tts.playback, ui.phase])

  const stop = useCallback(() => {
    abortRef.current?.abort()
    playerRef.current?.stop()
    audioRef.current?.pause()
    setPlaying(false)
    pcmRef.current = new Uint8Array(0)
    livePcmRef.current.reset()
    setPcmDuration(0)
    dispatch({ type: 'reset' })
  }, [])

  const setMode = (m: Mode) =>
    setTts((p) => {
      // Auto-swap the upstream when toggling modes on the default ports
      // (1.7B VoiceDesign :8898 <-> 0.6B CustomVoice :8870). Custom URLs
      // are left untouched.
      let url = p.url
      if (m === 'customvoice' && /:8898($|\/)/.test(url)) url = 'http://127.0.0.1:8870'
      else if (m === 'voicedesign' && /:8870($|\/)/.test(url)) url = 'http://127.0.0.1:8898'
      return { ...p, mode: m, url }
    })

  const modeProps = {
    tts,
    setTts,
    onGenerate: () => void run(),
  }

  return (
    <div className="app">
      <audio ref={audioRef} style={{ display: 'none' }} />
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
          <StageTimeline timing={ui.timing} stage={ui.stage} llmEnabled={llm.enabled} ttfaMs={ui.ttfaMs} totalMs={ui.totalMs} />
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
              <Waveform liveBuffer={livePcmRef.current} pcm={ui.phase === 'done' ? pcmRef.current : undefined} />
            </div>
          )}
          <AudioControls
            playing={playing}
            setPlaying={setPlaying}
            wavUrl={wavUrl}
            frames={ui.frames}
            onStop={() => {
              playerRef.current?.stop()
              audioRef.current?.pause()
              setPlaying(false)
            }}
            onPauseToggle={() => {
              const a = audioRef.current
              if (tts.playback !== 'live' && a && a.src) {
                if (a.paused) void a.play()
                else a.pause()
              } else if (playing) playerRef.current?.pause()
              else playerRef.current?.resume()
            }}
          />
        </main>
      </div>
    </div>
  )
}
