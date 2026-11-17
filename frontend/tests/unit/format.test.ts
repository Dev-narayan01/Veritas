import { describe, expect, it } from 'vitest'
import { bytes, percent, plainPercent, stageLabel, truncate } from '../../src/lib/format'

describe('formatting', () => {
  it('renders a missing probability as an em dash rather than 0%', () => {
    expect(percent(null)).toBe('—')
    expect(percent(undefined)).toBe('—')
  })

  it('formats fractions and already-percent values separately', () => {
    expect(percent(0.1234)).toBe('12.3%')
    expect(plainPercent(12.34)).toBe('12.3%')
  })

  it('truncates with an ellipsis only when needed', () => {
    expect(truncate('short', 10)).toBe('short')
    expect(truncate('abcdefghijk', 5)).toEqual('abcd…')
  })

  it('formats byte sizes', () => {
    expect(bytes(900)).toBe('900 B')
    expect(bytes(2048)).toBe('2 kB')
    expect(bytes(5 * 1024 * 1024)).toBe('5.0 MB')
  })

  it('maps unknown stages to themselves', () => {
    expect(stageLabel('ai')).toBe('AI-authorship analysis')
    expect(stageLabel('mystery')).toBe('mystery')
  })
})
