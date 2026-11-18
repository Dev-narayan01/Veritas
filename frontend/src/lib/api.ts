/** Typed client for the Veritas API. All network calls go through `request()`, which
 * attaches the resolved base URL and turns error bodies into readable messages. */
import type {
  AIResult,
  AnalysisOptions,
  AnalysisResult,
  AnalysisSummary,
  Health,
  MetricsResponse,
  ProgressEvent,
  Verdict,
} from '../types'
import { fetchJson, modeIsAvailable, resolveApiBase, type ApiBase } from './apiBase'

let current: ApiBase = { base: '', mode: 'offline', health: null, message: null }

/** Lite deployment result cache. It lives in sessionStorage only, is keyed by a single
 * id, and is cleared when the tab closes: the Lite functions keep no history. */
const LITE_ID = 'lite-latest'
const LITE_KEY = `veritas:${LITE_ID}`

export function writeCachedResult(result: AnalysisResult): void {
  try {
    sessionStorage.setItem(LITE_KEY, JSON.stringify(result))
  } catch {
    /* storage full or blocked: the results page will say so */
  }
}

export function readCachedResult(id: string): AnalysisResult | null {
  if (id !== LITE_ID) return null
  try {
    const raw = sessionStorage.getItem(LITE_KEY)
    return raw ? (JSON.parse(raw) as AnalysisResult) : null
  } catch {
    return null
  }
}

export function clearCachedResult(id: string): void {
  if (id !== LITE_ID) return
  try {
    sessionStorage.removeItem(LITE_KEY)
  } catch {
    /* nothing to clear */
  }
}

export function getApiState(): ApiBase {
  return current
}

export async function initApi(onWaiting?: (message: string) => void): Promise<ApiBase> {
  current = await resolveApiBase({ onWaiting })
  return current
}

export function setApiState(state: ApiBase): void {
  current = state
}

/** Base URL for links the browser navigates to directly (downloads, plot images).
 * Uses the same resolution as the JSON calls, so a Lite or offline deployment does not
 * build links against a host it never contacted. */
export function apiUrl(path: string): string {
  return `${current.base}${path}`
}

function url(path: string): string {
  return `${current.base}${path}`
}

async function request<T>(path: string, init?: RequestInit, timeoutMs = 120_000): Promise<T> {
  try {
    return await fetchJson<T>(url(path), init, timeoutMs)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    let parsed = message
    try {
      const body = JSON.parse(message.replace(/^\d+\s*/, ''))
      parsed = body?.detail?.error?.message ?? body?.error?.message ?? message
    } catch {
      /* not JSON: keep the raw message */
    }
    throw new Error(parsed)
  }
}

export const api = {
  health: () => request<Health>('/api/health'),
  samples: () => request<Array<{ label: string; text: string }>>('/api/samples'),

  /** True when only the Lite functions answer: pasted text, classical detector only. */
  isLite: () => current.mode === 'lite',

  createAnalysis: async (payload: {
    files: File[]
    text?: string
    url?: string
    options: AnalysisOptions
  }): Promise<{ id: string }> => {
    // Lite deployment: one pasted text, classical detector, no server-side history.
    // Accepting the request and then quietly doing less would be dishonest, so the
    // restriction is enforced here and surfaced with the server's own message.
    if (current.mode === 'lite') {
      if (payload.files.length || payload.url) {
        throw new Error(
          'This deployment can only analyse pasted text. Upload files on the full server (the site connects automatically when it is awake).',
        )
      }
      if (!payload.text) throw new Error('Paste some text to analyse.')
      const result = await request<AnalysisResult>(
        '/api/analyze',
        {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          // Honour the mode the user picked. Sending 'lite' unconditionally made the
          // mode selector decorative on this deployment. The server degrades fast->lite
          // and says so in mode_used, so an unavailable mode cannot be faked here.
          body: JSON.stringify({
            text: payload.text,
            mode:
              payload.options.mode && modeIsAvailable(current.health, payload.options.mode)
                ? payload.options.mode
                : 'lite',
          }),
        },
        60_000,
      )
      writeCachedResult(result)
      return { id: LITE_ID }
    }
    const form = new FormData()
    payload.files.forEach((file) => form.append('files', file))
    if (payload.text) form.append('text', payload.text)
    if (payload.url) form.append('url', payload.url)
    form.append('options', JSON.stringify(payload.options))
    return request<{ id: string }>('/api/analyses', { method: 'POST', body: form }, 180_000)
  },

  analysis: async (id: string): Promise<AnalysisResult & { error?: string | null }> => {
    if (current.mode === 'lite') {
      const cached = readCachedResult(id)
      if (cached) return cached
      throw new Error('Lite mode keeps no history: analyse the text again on this deployment, or use the full server.')
    }
    return request<AnalysisResult & { error?: string | null }>(`/api/analyses/${id}`)
  },
  history: async (): Promise<AnalysisSummary[]> => {
    if (current.mode === 'lite') {
      const cached = readCachedResult(LITE_ID)
      return cached ? [{ id: LITE_ID, created_at: cached.created_at, status: 'done', filenames: cached.documents.map((d) => d.filename), mode: 'lite', verdicts: cached.documents.map((d) => d.ai?.verdict ?? 'too_short'), similarity: null }] : []
    }
    return request<AnalysisSummary[]>('/api/analyses')
  },
  deleteAnalysis: async (id: string): Promise<void> => {
    if (current.mode === 'lite') {
      clearCachedResult(id)
      return
    }
    return request<void>(`/api/analyses/${id}`, { method: 'DELETE' })
  },
  metrics: () => request<MetricsResponse>('/api/metrics'),

  /** SSE progress with a polling fallback (1.5 s) when EventSource is unavailable. */
  watchProgress: (
    id: string,
    onEvent: (event: ProgressEvent) => void,
    onDone: () => void,
    onError: (message: string) => void,
  ): (() => void) => {
    if (current.mode === 'lite') {
      // The Lite function answers in one request, so there is nothing to stream.
      onEvent({ stage: 'done', progress: 1, message: 'lite analysis complete' })
      const timer = setTimeout(onDone, 0)
      return () => clearTimeout(timer)
    }
    let closed = false
    let pollTimer: ReturnType<typeof setInterval> | null = null
    let source: EventSource | null = null

    const finish = () => {
      closed = true
      source?.close()
      if (pollTimer) clearInterval(pollTimer)
      onDone()
    }

    const startPolling = () => {
      let lastProgress = -1
      pollTimer = setInterval(async () => {
        if (closed) return
        try {
          const result = await api.analysis(id)
          const status = (result as unknown as { status: string }).status
          if (status === 'done') {
            onEvent({ stage: 'done', progress: 1, message: 'analysis complete' })
            finish()
            return
          }
          if (status === 'error') {
            onError((result as unknown as { error?: string }).error ?? 'analysis failed')
            finish()
            return
          }
          const progress = Math.min(0.95, lastProgress + 0.05)
          lastProgress = progress
          onEvent({ stage: status || 'running', progress, message: 'working' })
        } catch (error) {
          onError(error instanceof Error ? error.message : String(error))
          finish()
        }
      }, 1500)
    }

    if (typeof EventSource === 'undefined') {
      startPolling()
      return () => {
        closed = true
        if (pollTimer) clearInterval(pollTimer)
      }
    }

    try {
      source = new EventSource(url(`/api/analyses/${id}/events`))
      source.addEventListener('progress', (event) => {
        const payload = JSON.parse((event as MessageEvent).data) as ProgressEvent
        onEvent(payload)
      })
      source.addEventListener('done', () => finish())
      source.addEventListener('error', () => {
        // A transport error is not fatal: fall back to polling once.
        if (!closed && !pollTimer) {
          source?.close()
          startPolling()
        }
      })
      source.addEventListener('timeout', () => {
        if (!closed && !pollTimer) startPolling()
      })
    } catch {
      startPolling()
    }

    return () => {
      closed = true
      source?.close()
      if (pollTimer) clearInterval(pollTimer)
    }
  },
}

/** Reason a mode cannot run on this server, or null when it can. Never guessed: the
 * text comes from /api/health, which reports the component that failed to load. */
export function reasonForMode(health: Health | null, mode: string): string | null {
  if (!health) return 'Waiting for the server'
  const entry = health.modes.find((item) => item.mode === mode)
  if (!entry) return 'This server does not offer that mode'
  return entry.available ? null : (entry.reason ?? 'Required components are unavailable')
}

export function verdictLabel(verdict: Verdict): string {
  switch (verdict) {
    case 'likely_human':
      return 'Likely human'
    case 'likely_ai':
      return 'Likely AI'
    case 'uncertain':
      return 'Uncertain'
    default:
      return 'Too short'
  }
}

export function verdictColor(verdict: Verdict): string {
  switch (verdict) {
    case 'likely_human':
      return 'var(--human)'
    case 'likely_ai':
      return 'var(--ai)'
    case 'uncertain':
      return 'var(--uncertain)'
    default:
      return 'var(--muted)'
  }
}

export function topVerdict(documents: Array<{ ai: AIResult | null }>): Verdict {
  const order: Verdict[] = ['likely_ai', 'uncertain', 'likely_human', 'too_short']
  const present = documents.map((d) => d.ai?.verdict).filter(Boolean) as Verdict[]
  for (const verdict of order) if (present.includes(verdict)) return verdict
  return 'too_short'
}
