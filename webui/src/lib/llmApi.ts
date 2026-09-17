/** Converts a user-supplied base URL into the absolute chat endpoint. */
function buildChatUrl(base: string): string {
  let b = base.trim().replace(/\/+$/, '')
  if (!b) return ''
  // URL completa del endpoint (ej. Z.AI docs: https://api.z.ai/api/paas/v4/chat/completions)
  if (/\/chat\/completions$/.test(b) || /\/completions$/.test(b)) return b
  // Estilo OpenAI: base termina en /v1 -> /v1/chat/completions
  if (/\/v\d+$/.test(b)) return b + '/chat/completions'
  // Z.AI: base termina en /paas/v4 -> /paas/v4/chat/completions
  if (/\/paas\/v\d+$/.test(b)) return b + '/chat/completions'
  // base plana (host raíz): convención OpenAI
  return b + '/v1/chat/completions'
}

function buildCompletionUrl(base: string): string {
  const chat = buildChatUrl(base)
  return chat.replace(/\/chat\/completions$/, '/completions')
}
import type { LlmSettings } from './types'

export interface LlmDelta {
  kind: 'chunk' | 'done' | 'error'
  text?: string
  error?: string
}

interface StreamResult {
  ok: boolean
  status: number
  body?: ReadableStream<Uint8Array> | null
  text?: string
}

async function postStream(
  url: string,
  headers: Record<string, string>,
  body: string,
  signal: AbortSignal
): Promise<StreamResult> {
  const res = await fetch(url, {
    method: 'POST',
    headers,
    body,
    signal,
  })
  if (res.ok) return { ok: true, status: res.status, body: res.body }
  const text = await res.text().catch(() => '')
  return { ok: false, status: res.status, text: text.slice(0, 400) }
}

function chatBody(settings: LlmSettings, prompt: string) {
  const body: Record<string, unknown> = {
    model: settings.model,
    messages: [
      { role: 'system', content: settings.systemPrompt },
      { role: 'user', content: prompt },
    ],
    max_tokens: settings.maxTokens,
    temperature: settings.temperature,
    stream: true,
  }
  // GLM 4.5+ piensa por defecto (thinking ON): el bloque de razonamiento
  // puede filtrarse al texto que después lee el TTS (estáticos/símbolos).
  // Como hace Hermes (plugins/model-providers/zai): thinking OFF.
  if (/^glm-([4-9]\.\d+|[5-9])/i.test(settings.model)) {
    body.extra_body = { thinking: { type: 'disabled' } }
  }
  return body
}

function completionBody(settings: LlmSettings, prompt: string) {
  // llama.cpp "completion" has no system role: embed the system prompt
  // inline, chat-style, as plain text.
  return {
    model: settings.model,
    prompt: `${settings.systemPrompt}\n\nUsuario: ${prompt}\nAsistente:`,
    max_tokens: settings.maxTokens,
    temperature: settings.temperature,
    stream: true,
  }
}

function parseSseLine(line: string, isChat: boolean): { text: string; done: boolean } | null {
  const t = line.trim()
  if (!t.startsWith('data:')) return null
  const payload = t.slice(5).trim()
  if (payload === '[DONE]') return { text: '', done: true }
  try {
    const j = JSON.parse(payload)
    if (isChat) {
      // GLM thinking: el razonamiento llega en reasoning_content — se
      // ignora (NUNCA alimentar el TTS con símbolos de razonamiento).
      const text = (j.choices?.[0]?.delta?.content as string | undefined) ?? ''
      const fin = j.choices?.[0]?.finish_reason as string | undefined
      return { text, done: !!fin }
    }
    const text = (j.choices?.[0]?.text as string | undefined) ?? ''
    const fin = j.choices?.[0]?.finish_reason as string | undefined
    return { text, done: !!fin }
  } catch {
    return null
  }
}

/** Returns an async generator of deltas. Throws on transport errors. */
export async function* streamChat(
  settings: LlmSettings,
  prompt: string,
  onSignal?: (signal: AbortSignal) => void
): AsyncGenerator<LlmDelta, void, unknown> {
  const ctrl = new AbortController()
  onSignal?.(ctrl.signal)
  const enc = encodeURIComponent
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'accept-language': 'en-US,en', // Z.AI lo recomienda explícitamente
    ...(settings.apiKey ? { authorization: `Bearer ${settings.apiKey}` } : {}),
  }

  // The proxy forwards to an absolute URL via ?target= ; llmApi builds
  // the full endpoint from the base (handles /v1, /paas/v4, or a full
  // /chat/completions URL pasted from vendor docs).
  const chatTarget = buildChatUrl(settings.baseUrl)
  if (!chatTarget) throw new Error('LLM: endpoint base vacío')
  const chatUrl = `/proxy/llm?target=${enc(chatTarget)}`
  const tryChat = await postStream(chatUrl, headers, JSON.stringify(chatBody(settings, prompt)), ctrl.signal)

  let res: StreamResult
  let isChat = true
  if (tryChat.ok) {
    res = tryChat
  } else if (tryChat.status === 404 || tryChat.status === 400 || tryChat.status === 401) {
    // 404: endpoint not exposed; 400: model rejects chat format;
    // 401/403: auth may pass on the alternative endpoint.
    const compUrl = `/proxy/llm?target=${enc(buildCompletionUrl(settings.baseUrl))}`
    res = await postStream(compUrl, headers, JSON.stringify(completionBody(settings, prompt)), ctrl.signal)
    isChat = false
  } else {
    throw new Error(`LLM HTTP ${tryChat.status}: ${tryChat.text || 'unknown'}`)
  }

  if (!res.ok) {
    throw new Error(`LLM HTTP ${res.status}: ${res.text || 'unknown'}`)
  }
  if (!res.body) {
    throw new Error('LLM: sin body de streaming')
  }

  const reader = res.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      const lines = buf.split('\n')
      buf = lines.pop() ?? ''
      for (const line of lines) {
        const parsed = parseSseLine(line, isChat)
        if (!parsed) continue
        if (parsed.text) yield { kind: 'chunk', text: parsed.text }
        if (parsed.done) {
          yield { kind: 'done' }
          return
        }
      }
    }
    yield { kind: 'done' }
  } catch (e) {
    if ((e as Error).name === 'AbortError') {
      yield { kind: 'error', error: 'abortado' }
      return
    }
    throw e
  }
}
