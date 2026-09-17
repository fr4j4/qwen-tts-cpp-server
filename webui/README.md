# 🎙️ Qwen TTS Lab — Web UI

Laboratorio web interactivo para probar el motor local de TTS (**qwen-tts-cpp-server**) con todos sus modos y capacidades, incluido un pipeline **LLM → TTS en tiempo real** contra cualquier proveedor OpenAI-compatible.

## Requisitos

- Node.js ≥ 20 (probado con v26)
- pnpm (o npm/yarn)
- El server TTS corriendo: `./build-cuda/tts-server --model <1.7b-voicedesign> --codec <12hz> --host 0.0.0.0 --port 8898`

> ⚠️ Para que la webui alcance al TTS desde **cualquier PC de la LAN**, el `tts-server` debe escuchar en `0.0.0.0` (el `--host` predeterminado es `127.0.0.1`, solo local).

## Arranque rápido

```bash
cd webui
pnpm install
pnpm build          # compila dist/
pnpm start          # sirve en http://0.0.0.0:9898 (LAN: http://<ip-de-esta-pc>:9898)
```

- **Dev**: `pnpm dev` → http://localhost:5173 (también accesible por IP).
- **Config de arranque**: `TTS_UPSTREAM` (default `http://127.0.0.1:8898`) y `WEBUI_PORT` (default `9898`).

## Qué se puede probar

### 🎭 Modo VoiceDesign (1.7B)
- Instrucción de estilo obligatoria: chips rápidos + texto libre + ejemplos.
- El server rechaza síntesis sin instrucción (400) — la UI lo previene.

### 🗣️ Modo CustomVoice (0.6B)
- Grid de las 9 voces fijas del modelo (serena, vivian, uncle_fu, ryan, aiden, ono_anna, sohee, eric, dylan), con avatar e idioma.
- Semilla para reproducción determinística.

### 🤖 Pipeline LLM → TTS (tiempo real)
En **Conexiones → LLM (OpenAI-compatible)**:
- **Endpoint base** (ej. `http://127.0.0.1:8081/v1`), **API key** (opcional) y **nombre de modelo**.
- El texto de abajo se envía como prompt; la respuesta del LLM es lo que se sintetiza con **streaming de tokens**.
- Autodetecta: si el endpoint no expone `/chat/completions` (llama.cpp con solo `completion`), hace **fallback automático** a `/v1/completions`.
- **Timeline de 3 etapas con tiempos**: 🤖 LLM procesando → 📤 texto enviado al TTS → 🔊 inicio del stream de audio PCM (24 kHz s16), reproducido en vivo con Web Audio.

### 🎛️ General
- Waveform del PCM acumulado, reproducir/pausar/detener, descargar WAV.
- Estado de salud del TTS server en el header (polling).
- Ajustes persistidos en localStorage.

## Arquitectura (red local)

```
Browser (cualquier PC)  ──►  http://<ip>:9898  (webui: Node server.js / Vite dev)
                                 │  /proxy/tts/*   ──►  tts-server (URL configurable en UI, ?upstream=)
                                 │  /proxy/llm/*   ──►  LLM OpenAI-compatible (URL en UI, ?target=)
                                 ▼
                          sin CORS — el browser sólo habla con la webui
```

- El frontend nunca toca los servidores de voz/LLM directamente: el proxy same-origin de la webui reenvía, así que **no hay problemas de CORS desde ninguna máquina**.
- En la UI puedes cambiar la URL del TTS (campo "URL base" en Conexiones) a cualquier IP de la LAN sin reiniciar la webui (cada request lleva `?upstream=`).

## Estructura

```
webui/
├── index.html            # shell HTML
├── vite.config.js       # dev + proxy same-origin (LAN: host:true)
├── server.js            # prod: sirve dist/ + proxy compartido
├── proxy-lib.mjs        # proxy /proxy/tts + /proxy/llm (compartido dev/prod)
├── src/
│   ├── main.tsx         # entrada React
│   ├── App.tsx          # orquestación del pipeline LLM→TTS (reducer de etapas)
│   ├── styles.css       # tema oscuro
│   ├── components/      # Header, SettingsPanel, StageTimeline, Waveform, AudioControls
│   ├── modes/           # VoiceDesignPanel, CustomVoicePanel
│   └── lib/             # types, storage, llmApi (SSE + fallback), ttsApi, pcmPlayer (Web Audio)
└── dist/                # build de producción (lo sirve server.js)
```
