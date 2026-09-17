import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { proxyHandler } from './proxy-lib.mjs'

// Same-origin proxy so the browser never hits CORS. Works from any
// machine in the LAN: the UI points at the webui server's IP, and the
// proxy forwards to the TTS upstream (default here, overridable per
// request via ?upstream=).
export default defineConfig({
  plugins: [
    react(),
    {
      name: 'same-origin-proxy',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url && req.url.startsWith('/proxy/')) {
            proxyHandler(req, res, process.env.TTS_UPSTREAM || 'http://127.0.0.1:8898')
          } else {
            next()
          }
        })
      },
    },
  ],
  server: {
    port: 5173,
    host: true, // LAN access via http://<ip>:5173
  },
})
