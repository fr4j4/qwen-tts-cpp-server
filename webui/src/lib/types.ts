// Domain types for the TTS lab.

export type Mode = 'voicedesign' | 'customvoice'

export interface TtsSettings {
  mode: Mode
  url: string // tts-server base (same-origin proxy; informational)
  speaker: string // CustomVoice only
  seed: string // '' = random
  instruct: string // VoiceDesign only
}

export interface LlmSettings {
  enabled: boolean
  baseUrl: string // OpenAI-compatible base, e.g. http://127.0.0.1:8081/v1
  apiKey: string
  model: string
  systemPrompt: string
  maxTokens: number
  temperature: number
}

export type Stage = 'idle' | 'llm' | 'tts' | 'audio'

export interface StageTiming {
  stage: Stage
  at: number // ms epoch
  detail?: string
}

export interface SynthEvent {
  kind: 'frame' | 'chunk' | 'end' | 'error'
  /** cumulative pcm frames (1 frame = 1/100 s) or bytes for chunk */
  count: number
  ms?: number
  text?: string
  error?: string
}

export type Phase = 'idle' | 'generating' | 'done' | 'error'

export interface UiState {
  phase: Phase
  stage: Stage
  llmText: string
  llmDone: boolean
  synthText: string // text actually sent to TTS
  frames: number
  timing: StageTiming[]
  error: string | null
}

export const SPEAKERS = [
  'serena',
  'vivian',
  'uncle_fu',
  'ryan',
  'aiden',
  'ono_anna',
  'sohee',
  'eric',
  'dylan',
]

export const STYLE_CHIPS = [
  'Natural y tranquila',
  'Entusiasta y energética',
  'Suave y susurrante',
  'Seria y profesional',
  'Alegre y cálida',
  'Narración pausada',
  'Urgente y nerviosa',
  'Fría y distante',
]

export const SAMPLE_TEXTS = [
  'Hola, ¿cómo estás? Bienvenido al laboratorio de voz.',
  'Este es el modelo Qwen TTS funcionando en tiempo real.',
  'La inteligencia artificial habla con voz humana, casi.',
]
