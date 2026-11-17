/**
 * Graphics tier detection (spec 13.2).
 *
 * Tiers: `high` (full scenes, bloom), `low` (fewer instances, no postprocessing),
 * `off` (no WebGL at all, 2D fallbacks). The decision uses WebGL2 availability,
 * `prefers-reduced-motion`, core count, device memory, touch input and a short
 * frame-time probe, and the user can override it with Auto / High / Low / Off.
 */
export type GraphicsTier = 'high' | 'low' | 'off'
export type GraphicsPreference = 'auto' | GraphicsTier

export interface GraphicsEnv {
  webgl2: boolean
  reducedMotion: boolean
  hardwareConcurrency: number
  deviceMemoryGb: number
  touch: boolean
  /** Median frame time in ms from a short probe, or null when not measured. */
  frameTimeMs: number | null
}

const MIN_FPS_FRAME_MS = 1000 / 40

export function detectWebgl2(): boolean {
  if (typeof document === 'undefined') return false
  try {
    const canvas = document.createElement('canvas')
    const gl = canvas.getContext('webgl2')
    if (!gl) return false
    // Release the context immediately: only one WebGL canvas may be mounted at a time.
    const lose = (gl as WebGL2RenderingContext).getExtension('WEBGL_lose_context')
    lose?.loseContext()
    return true
  } catch {
    return false
  }
}

export function readEnv(): GraphicsEnv {
  const nav = typeof navigator === 'undefined' ? undefined : navigator
  const reducedMotion =
    typeof window !== 'undefined' &&
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  return {
    webgl2: detectWebgl2(),
    reducedMotion,
    hardwareConcurrency: nav?.hardwareConcurrency ?? 4,
    deviceMemoryGb: (nav as unknown as { deviceMemory?: number })?.deviceMemory ?? 4,
    touch: typeof window !== 'undefined' && 'ontouchstart' in window,
    frameTimeMs: null,
  }
}

/** Pure decision function — unit tested. */
export function decideTier(env: GraphicsEnv): GraphicsTier {
  if (!env.webgl2) return 'off'
  if (env.reducedMotion) return 'off'
  const slowProbe = env.frameTimeMs !== null && env.frameTimeMs > MIN_FPS_FRAME_MS
  const weak =
    env.hardwareConcurrency <= 4 ||
    env.deviceMemoryGb <= 4 ||
    slowProbe ||
    env.touch
  return weak ? 'low' : 'high'
}

export function resolveTier(preference: GraphicsPreference, env: GraphicsEnv): GraphicsTier {
  if (preference === 'off') return 'off'
  if (preference === 'high') return env.webgl2 && !env.reducedMotion ? 'high' : 'off'
  if (preference === 'low') return env.webgl2 && !env.reducedMotion ? 'low' : 'off'
  return decideTier(env)
}

/** Instance budgets per tier, shared by every scene so scenes stay consistent. */
export const BUDGETS: Record<GraphicsTier, { sheets: number; particles: number; dpr: [number, number]; postprocessing: boolean }> = {
  high: { sheets: 60, particles: 1500, dpr: [1, 2], postprocessing: true },
  low: { sheets: 20, particles: 300, dpr: [1, 1], postprocessing: false },
  off: { sheets: 0, particles: 0, dpr: [1, 1], postprocessing: false },
}
