// Tiny typed localStorage helper with JSON fallback.

const K = {
  tts: 'qttl:tts',
  llm: 'qttl:llm',
}

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return fallback
    return { ...fallback, ...JSON.parse(raw) }
  } catch {
    return fallback
  }
}

function write<T>(key: string, v: T) {
  try {
    localStorage.setItem(key, JSON.stringify(v))
  } catch {
    /* ignore */
  }
}

export function loadTts<T extends object>(fallback: T): T {
  return read(K.tts, fallback)
}
export function saveTts<T extends object>(v: T) {
  write(K.tts, v)
}
export function loadLlm<T extends object>(fallback: T): T {
  return read(K.llm, fallback)
}
export function saveLlm<T extends object>(v: T) {
  write(K.llm, v)
}
