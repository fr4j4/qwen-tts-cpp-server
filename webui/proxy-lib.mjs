// Same-origin proxy shared by Vite (dev) and server.js (prod).
// Routes:
//   /proxy/tts/*   -> upstream TTS server (default http://127.0.0.1:8898)
//   /proxy/llm/*   -> the OpenAI-compatible LLM endpoint configured in the UI
//                     (client sends ?target=<url>; the proxy strips it and
//                     forwards, so the browser never sees CORS)
//   /proxy/health  -> upstream /health passthrough
//
// LAN: the webui server binds 0.0.0.0. Any machine can reach the UI at
// http://<webui-ip>:<port> and point it at the TTS/LLM servers via the
// settings (the browser still only talks to the webui origin).
import http from 'node:http'
import https from 'node:https'
import { URL } from 'node:url'

/**
 * Forward a request to an absolute URL (`target` already includes the
 * upstream path). The browser request body/headers stream through.
 */
function forward(req, res, target) {
  let url
  try {
    url = new URL(target)
  } catch {
    res.writeHead(502, { 'content-type': 'application/json' })
    res.end(JSON.stringify({ error: { message: `invalid proxy target: ${target}` } }))
    return
  }
  const lib = url.protocol === 'https:' ? https : http
  const headers = { ...req.headers }
  delete headers.host
  delete headers['content-length']

  const out = lib.request(
    {
      hostname: url.hostname,
      port: url.port || (url.protocol === 'https:' ? 443 : 80),
      path: url.pathname + url.search,
      method: req.method,
      headers,
    },
    (up) => {
      res.writeHead(up.statusCode || 502, up.headers)
      up.pipe(res)
    }
  )
  out.on('error', (e) => {
    res.writeHead(502, { 'content-type': 'application/json' })
    res.end(JSON.stringify({ error: { message: `upstream error: ${e.message}` } }))
  })
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    req.pipe(out)
  } else {
    out.end()
  }
}

// Proxy handler for a raw node req/res (Vite connect middleware or http server).
export function proxyHandler(req, res, ttsUpstream) {
  const u = req.url || ''
  if (u.startsWith('/proxy/tts/') || u.startsWith('/proxy/health')) {
    // ?upstream=<url> overrides the server default: lets a browser on
    // ANY machine in the LAN point the UI at the TTS server's IP
    // without restarting the webui server.
    const parsed = new URL(u, 'http://local')
    const upstream = parsed.searchParams.get('upstream') || ttsUpstream
    parsed.searchParams.delete('upstream')
    const rest = (parsed.pathname || '')
      .replace(/^\/proxy\/tts/, '')
      .replace(/^\/proxy\/health/, '/health') + parsed.search
    return forward(req, res, upstream.replace(/\/+$/, '') + rest)
  }
  if (u.startsWith('/proxy/llm/')) {
    // ?target=<openai-compatible base, may include /v1> ; path after
    // /proxy/llm is the API path WITHOUT /v1 (e.g. /chat/completions).
    // llmApi normalises the base so /v1 appears exactly once upstream.
    const parsed = new URL(u, 'http://local')
    const target = parsed.searchParams.get('target')
    parsed.searchParams.delete('target')
    const rest = (parsed.pathname || '').replace(/^\/proxy\/llm/, '') + parsed.search
    if (!target) {
      res.writeHead(400, { 'content-type': 'application/json' })
      res.end(JSON.stringify({ error: { message: 'missing ?target= LLM base URL' } }))
      return
    }
    return forward(req, res, target.replace(/\/+$/, '') + rest)
  }
  res.writeHead(404, { 'content-type': 'application/json' })
  res.end(JSON.stringify({ error: { message: 'not found' } }))
}
