// Production server for the built webui: serves dist/ and proxies
// /proxy/* to the TTS upstream and the configured LLM endpoint.
import http from 'node:http'
import { readFile, stat } from 'node:fs/promises'
import { extname, join, normalize } from 'node:path'
import { proxyHandler } from './proxy-lib.mjs'

const PORT = Number(process.env.WEBUI_PORT || 9898)
const TTS_UPSTREAM = process.env.TTS_UPSTREAM || 'http://127.0.0.1:8898'
const DIST = join(process.cwd(), 'dist')

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.css': 'text/css',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.json': 'application/json',
  '.map': 'application/json',
  '.woff2': 'font/woff2',
}

async function serveStatic(req, res, pathname) {
  let rel = pathname === '/' ? '/index.html' : pathname
  const file = normalize(join(DIST, rel))
  if (!file.startsWith(DIST)) {
    res.writeHead(403)
    res.end('forbidden')
    return
  }
  try {
    const st = await stat(file)
    if (st.isDirectory()) throw new Error('dir')
    const buf = await readFile(file)
    res.writeHead(200, {
      'content-type': MIME[extname(file)] || 'application/octet-stream',
      'cache-control': pathname === '/' ? 'no-cache' : 'public, max-age=3600',
    })
    res.end(buf)
  } catch {
    // SPA fallback
    try {
      const buf = await readFile(join(DIST, 'index.html'))
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
      res.end(buf)
    } catch {
      res.writeHead(404)
      res.end('not built yet: run `pnpm build` in webui/')
    }
  }
}

const server = http.createServer(async (req, res) => {
  const u = new URL(req.url || '/', 'http://local')
  if (u.pathname.startsWith('/proxy/')) {
    proxyHandler(req, res, TTS_UPSTREAM)
    return
  }
  await serveStatic(req, res, u.pathname)
})

server.listen(PORT, '0.0.0.0', () => {
  console.log(`[webui] http://0.0.0.0:${PORT}  (tts upstream: ${TTS_UPSTREAM})`)
})
