import { describe, expect, it } from 'vitest'
import { buildSpans, coverage, filterMatches, heatColor, matchesToRanges, sentencesToRanges } from '../../src/lib/spans'
import type { PlagMatch, SentenceScore } from '../../src/types'

const TEXT = 'Alpha beta gamma. Delta epsilon zeta. Eta theta iota.'

const sentence = (text: string, start: number, end: number, aiProb: number): SentenceScore => ({
  index: 0,
  start,
  end,
  text,
  ai_prob: aiProb,
  flagged_tokens: [],
})

const match = (overrides: Partial<PlagMatch>): PlagMatch => ({
  query_span: [0, 50],
  source_span: null,
  query_text: '',
  source_text: '',
  source_id: 's1',
  source_title: 'Source one',
  source_url: null,
  color_index: 0,
  match_type: 'exact',
  score: 1,
  words: 10,
  in_quote: false,
  in_bibliography: false,
  ...overrides,
})

describe('buildSpans', () => {
  it('returns the whole text when there are no ranges', () => {
    expect(buildSpans(TEXT, [])).toEqual([{ text: TEXT, start: 0, end: TEXT.length }])
  })

  it('never loses or duplicates a character', () => {
    const spans = buildSpans(TEXT, [
      { start: 0, end: 25, matchIndexes: [0] },
      { start: 10, end: 40, matchIndexes: [1] },
      { start: 44, end: 52, matchIndexes: [2] },
    ])
    expect(spans.map((span) => span.text).join('')).toBe(TEXT)
  })

  it('clamps ranges that run past the end of the text', () => {
    const spans = buildSpans(TEXT, [{ start: 40, end: 9999, matchIndexes: [0] }])
    expect(spans.at(-1)?.end).toBe(TEXT.length)
    expect(spans.map((span) => span.text).join('')).toBe(TEXT)
  })

  it('ignores empty and inverted ranges', () => {
    expect(buildSpans(TEXT, [{ start: 10, end: 10, matchIndexes: [] }, { start: 30, end: 4, matchIndexes: [] }])).toHaveLength(1)
  })
})

describe('coverage', () => {
  it('counts each character once for overlapping spans', () => {
    expect(coverage('x'.repeat(100), [[0, 40], [20, 60]])).toBeCloseTo(60)
  })

  it('handles empty text', () => {
    expect(coverage('', [[0, 10]])).toBe(0)
  })
})

describe('heatColor', () => {
  it('returns an rgba string that gets more opaque with probability', () => {
    expect(heatColor(0.1)).toMatch(/^rgba\(/)
    expect(heatColor(0.1)).toContain('0.28')
  })

  it('moves from green to red as probability rises', () => {
    expect(heatColor(0)).toContain('52, 211, 153')
    expect(heatColor(1)).toContain('248, 113, 113')
  })
})

describe('range builders', () => {
  it('maps sentences to their own index', () => {
    const ranges = sentencesToRanges([sentence('Alpha beta gamma.', 0, 17, 0.2), sentence('Delta epsilon zeta.', 18, 36, 0.9)])
    expect(ranges.map((range) => range.matchIndexes)).toEqual([[0], [1]])
  })

  it('uses query_span for matches', () => {
    const ranges = matchesToRanges([match({ query_span: [5, 15] }), match({ query_span: [30, 40] })])
    expect(ranges).toEqual([
      { start: 5, end: 15, matchIndexes: [0] },
      { start: 30, end: 40, matchIndexes: [1] },
    ])
  })
})

describe('filterMatches', () => {
  const matches = [
    match({ source_id: 'a', words: 30 }),
    match({ source_id: 'b', words: 30, in_quote: true }),
    match({ source_id: 'c', words: 30, in_bibliography: true }),
    match({ source_id: 'd', words: 4 }),
  ]

  it('keeps everything with no filters active', () => {
    expect(filterMatches(matches, { quotes: false, bibliography: false, minWords: 0, hideBelowPercent: 0 })).toHaveLength(4)
  })

  it('drops short matches, quotes and bibliography entries when asked', () => {
    const kept = filterMatches(matches, { quotes: true, bibliography: true, minWords: 6, hideBelowPercent: 0 })
    expect(kept.map((m) => m.source_id)).toEqual(['a'])
  })

  it('drops sources whose share of the document is below the threshold', () => {
    const kept = filterMatches(matches, { quotes: false, bibliography: false, minWords: 0, hideBelowPercent: 40 })
    expect(kept.every((m) => m.source_id === 'a')).toBe(true)
  })
})
