import { describe, expect, it } from 'vitest'
import { BUDGETS, decideTier, resolveTier, type GraphicsEnv } from '../../src/lib/graphics'

const strong: GraphicsEnv = {
  webgl2: true,
  reducedMotion: false,
  hardwareConcurrency: 16,
  deviceMemoryGb: 16,
  touch: false,
  frameTimeMs: 8,
}

describe('decideTier', () => {
  it('is off without WebGL2 — the page must still work', () => {
    expect(decideTier({ ...strong, webgl2: false })).toBe('off')
  })

  it('is off when the user prefers reduced motion', () => {
    expect(decideTier({ ...strong, reducedMotion: true })).toBe('off')
  })

  it('is high on a strong desktop', () => {
    expect(decideTier(strong)).toBe('high')
  })

  it('drops to low on weak, slow-probing or touch devices', () => {
    expect(decideTier({ ...strong, hardwareConcurrency: 2 })).toBe('low')
    expect(decideTier({ ...strong, deviceMemoryGb: 2 })).toBe('low')
    expect(decideTier({ ...strong, frameTimeMs: 40 })).toBe('low')
    expect(decideTier({ ...strong, touch: true })).toBe('low')
  })
})

describe('resolveTier', () => {
  it('honours an explicit Off preference even with WebGL2', () => {
    expect(resolveTier('off', strong)).toBe('off')
  })

  it('downgrades an explicit High to off when WebGL2 is missing', () => {
    expect(resolveTier('high', { ...strong, webgl2: false })).toBe('off')
  })

  it('auto delegates to the decision', () => {
    expect(resolveTier('auto', strong)).toBe('high')
    expect(resolveTier('auto', { ...strong, touch: true })).toBe('low')
  })
})

describe('BUDGETS', () => {
  it('allocates nothing to the off tier', () => {
    expect(BUDGETS.off.sheets).toBe(0)
    expect(BUDGETS.off.particles).toBe(0)
    expect(BUDGETS.off.postprocessing).toBe(false)
  })

  it('is strictly smaller for the low tier', () => {
    expect(BUDGETS.low.sheets).toBeLessThan(BUDGETS.high.sheets)
    expect(BUDGETS.low.particles).toBeLessThan(BUDGETS.high.particles)
    expect(BUDGETS.low.postprocessing).toBe(false)
  })
})
