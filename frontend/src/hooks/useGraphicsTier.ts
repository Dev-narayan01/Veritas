/** Reads the user's Effects preference (persisted in localStorage) and resolves it
 * against the detected environment. Non-3D behaviour is identical for every tier. */
import { useEffect, useMemo, useState } from 'react'
import { resolveTier, readEnv, type GraphicsEnv, type GraphicsPreference, type GraphicsTier } from '../lib/graphics'

const STORAGE_KEY = 'veritas.graphics'

export function useGraphicsPreference(): [GraphicsPreference, (value: GraphicsPreference) => void] {
  const [preference, setPreference] = useState<GraphicsPreference>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      return (stored as GraphicsPreference) ?? 'auto'
    } catch {
      return 'auto'
    }
  })
  const update = (value: GraphicsPreference) => {
    setPreference(value)
    try {
      localStorage.setItem(STORAGE_KEY, value)
    } catch {
      /* storage disabled: the choice still applies to this session */
    }
  }
  return [preference, update]
}

export function useGraphicsTier(preference: GraphicsPreference): { tier: GraphicsTier; env: GraphicsEnv } {
  const [env, setEnv] = useState<GraphicsEnv>(() => ({
    webgl2: false,
    reducedMotion: false,
    hardwareConcurrency: 4,
    deviceMemoryGb: 4,
    touch: false,
    frameTimeMs: null,
  }))

  useEffect(() => {
    const base = readEnv()
    setEnv(base)
    // Short frame-time probe: ten frames, keep the median.
    if (!base.webgl2) return
    let frames = 0
    let last = performance.now()
    const times: number[] = []
    let handle = 0
    const tick = (now: number) => {
      times.push(now - last)
      last = now
      frames += 1
      if (frames < 10) {
        handle = requestAnimationFrame(tick)
      } else {
        const sorted = [...times].sort((a, b) => a - b)
        setEnv({ ...base, frameTimeMs: sorted[Math.floor(sorted.length / 2)] })
      }
    }
    handle = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(handle)
  }, [])

  const tier = useMemo(() => resolveTier(preference, env), [preference, env])
  return { tier, env }
}
