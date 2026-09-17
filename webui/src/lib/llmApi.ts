// OpenAI-compatible chat streaming (SSE) via the same-origin proxy.
// Tries /v1/chat/completions first; falls back to /v1/completions
// automatically (llama.cpp servers commonly expose only "completion").
// Both are streamed token-by-token.
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
  return {
    model: settings.model,
    messages: [
      { role: 'system', content: settings.systemPrompt },
      { role: 'user', content: prompt },
    ],
    max_tokens: settings.maxTokens,
    temperature: settings.temperature,
    stream: true,
  }
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
      const text = j.choices?.[0]?.delta?.content as string | undefined
      const fin = j.choices?.[0]?.finish_reason as string | undefined
      return { text: text ?? '', done: !!fin }
    }
    const text = j.choices?.[0]?.text as string | undefined
    const fin = j.choices?.[0]?.finish_reason as string | undefined
    return { text: text ?? '', done: !!fin }
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
  // Normalise: OpenAI-compatible bases usually end in /v1; the client
  // paths below are relative (/chat/completions), so ensure /v1 once.
  let target = settings.baseUrl.replace(/\/+$/, '')
  if (!/\/v1$/.test(target)) target += '/v1'
  const enc = encodeURIComponent
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...(settings.apiKey ? { authorization: `Bearer ${settings.apiKey}` } : {}),
  }

  const chatUrl = `/proxy/llm/chat/completions?target=${enc(target)}`
  const tryChat = await postStream(chatUrl, headers, JSON.stringify(chatBody(settings, prompt)), ctrl.signal)

  let res: StreamResult
  let isChat = true
  if (tryChat.ok) {
    res = tryChat
  } else if (tryChat.status === 404 || tryChat.status === 400) {
    // 404: endpoint not exposed; 400: model rejects chat format.
    const compUrl = `/proxy/llm/completions?target=${enc(target)}`
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
