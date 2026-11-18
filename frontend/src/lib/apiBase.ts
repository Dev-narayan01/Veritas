/**
 * API base resolution with wake-up and Lite fallback (spec 14.4).
 *
 * Order of preference:
 *  1. `VITE_API_BASE_URL` (the Hugging Face Space) — polled for up to 90 s while
 *     showing the wake-up message, because free Spaces sleep when idle;
 *  2. same-origin `/api` (the Vercel Lite functions) when the Space does not answer;
 *  3. if neither answers, the UI stays usable in read-only mode and says so.
 *
 * The decision is pure-ish and testable: `decideBase(url, healthLookup)` takes the
 * probe function so unit tests never touch the network.
 */
import type { Health } from '../types'

export const WAKEUP_TIMEOUT_MS = 90_000
export const HEALTH_TIMEOUT_MS = 8_000

export type ApiMode = 'full' | 'lite' | 'offline'

export interface ApiBase {
  base: string
  mode: ApiMode
  health: Health | null
  message: string | null
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

export async function fetchJson<T>(url: string, init?: RequestInit, timeoutMs = HEALTH_TIMEOUT_MS): Promise<T> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(url, { ...init, signal: controller.signal })
    if (!response.ok) {
      const body = await response.text().catch(() => '')
      throw new Error(`${response.status} ${body.slice(0, 200)}`)
    }
    return (await response.json()) as T
  } finally {
    clearTimeout(timer)
  }
}

export async function probeHealth(base: string): Promise<Health> {
  return fetchJson<Health>(`${base}/api/health`, undefined, HEALTH_TIMEOUT_MS)
}

/**
 * Resolve the API base. `onWaiting` receives the wake-up message so the UI can show
 * it; `deadlineMs` bounds the total wait.
 */
export async function resolveApiBase(options?: {
  onWaiting?: (message: string) => void
  deadlineMs?: number
  probe?: (base: string) => Promise<Health>
}): Promise<ApiBase> {
  const configured = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '')
  const probe = options?.probe ?? probeHealth
  const deadline = Date.now() + (options?.deadlineMs ?? WAKEUP_TIMEOUT_MS)

  if (configured) {
    options?.onWaiting?.('Waking up the analysis server, this can take up to a minute')
    let warned = false
    while (Date.now() < deadline) {
      try {
        const health = await probe(configured)
        return { base: configured, mode: 'full', health, message: null }
      } catch {
        if (!warned) {
          warned = true
          options?.onWaiting?.('The full analysis server is still starting — trying again')
        }
        await sleep(3000)
      }
    }
  }

  // Same-origin Lite functions (Vercel) or the Vite dev proxy.
  try {
    const health = await probe('')
    const modes = health.modes ?? []
    // The question is not "is anything available" but "is the *full server* here". `fast`
    // becoming available lit up that check, so the site would have offered file uploads
    // and server-side history that these functions do not implement. Anything that is not
    // the full server means reduced capabilities, whatever detectors it happens to run.
    const full = modes.some((m) => (m.mode === 'standard' || m.mode === 'deep') && m.available)
    const fast = modes.some((m) => m.mode === 'fast' && m.available)
    return {
      base: '',
      mode: full ? 'full' : 'lite',
      health,
      message: full
        ? null
        : fast
          ? 'Lite deployment with fast mode available: no paraphrase detection, no files, no history'
          : 'Lite mode: reduced accuracy, no paraphrase detection',
    }
  } catch {
    return {
      base: '',
      mode: 'offline',
      health: null,
      message: 'The analysis server is unreachable. Check your connection and reload.',
    }
  }
}

export function modeIsAvailable(health: Health | null, mode: string): boolean {
  if (!health) return false
  return health.modes.some((m) => m.mode === mode && m.available)
}

export function reasonForMode(health: Health | null, mode: string): string | null {
  if (!health) return 'health information unavailable'
  return health.modes.find((m) => m.mode === mode)?.reason ?? null
}
