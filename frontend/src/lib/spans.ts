/** Span utilities shared by the heatmap, the plagiarism viewer and the filters.
 * Everything is expressed in character offsets into the document text the API
 * returned, so highlights always line up with what the backend measured. */
import type { PlagMatch, SentenceScore, TokenScore } from '../types'

export interface Segment {
  text: string
  start: number
  end: number
  color?: string
  sentenceIndex?: number
  matchIndexes?: number[]
}

/** Split text into a flat list of segments with no overlaps, later entries winning. */
export function buildSpans(
  text: string,
  ranges: Array<{ start: number; end: number; matchIndexes: number[] }>,
): Segment[] {
  if (!text) return []
  const sorted = [...ranges]
    .filter((r) => r.end > r.start)
    .sort((a, b) => a.start - b.start || b.end - a.end)
  const out: Segment[] = []
  let cursor = 0
  for (const range of sorted) {
    const start = Math.max(cursor, Math.min(range.start, text.length))
    const end = Math.min(Math.max(range.end, start), text.length)
    if (start > cursor) {
      out.push({ text: text.slice(cursor, start), start: cursor, end: start })
    }
    if (end > start) {
      out.push({ text: text.slice(start, end), start, end, matchIndexes: range.matchIndexes })
      cursor = end
    }
  }
  if (cursor < text.length) {
    out.push({ text: text.slice(cursor), start: cursor, end: text.length })
  }
  return out
}

/** Character coverage of a set of spans, and the percentage of the document. */
export function coverage(text: string, spans: Array<[number, number]>): number {
  if (!text.length) return 0
  const covered = new Set<number>()
  for (const [start, end] of spans) {
    for (let i = Math.max(0, start); i < Math.min(end, text.length); i += 1) covered.add(i)
  }
  return (covered.size / text.length) * 100
}

/** Sentence background colour for the AI heatmap (0 = human, 1 = AI). */
export function heatColor(probability: number): string {
  const p = Math.min(1, Math.max(0, probability))
  const human = [52, 211, 153]
  const uncertain = [251, 191, 36]
  const ai = [248, 113, 113]
  const [from, to, t] = p <= 0.5 ? [human, uncertain, p * 2] : [uncertain, ai, (p - 0.5) * 2]
  const mixed = from.map((c, i) => Math.round(c + (to[i] - c) * t))
  return `rgba(${mixed[0]}, ${mixed[1]}, ${mixed[2]}, 0.28)`
}

export function sentencesToRanges(sentences: SentenceScore[]): Array<{ start: number; end: number; matchIndexes: number[] }> {
  return sentences.map((s, index) => ({ start: s.start, end: s.end, matchIndexes: [index] }))
}

export function matchesToRanges(matches: PlagMatch[]): Array<{ start: number; end: number; matchIndexes: number[] }> {
  return matches.map((m, index) => ({ start: m.query_span[0], end: m.query_span[1], matchIndexes: [index] }))
}

/** Client-side exclusion filter, mirroring `plagiarism/merge.filter_matches`. */
export function filterMatches(
  matches: PlagMatch[],
  options: { quotes: boolean; bibliography: boolean; minWords: number; hideBelowPercent: number },
): PlagMatch[] {
  const perSource = new Map<string, number>()
  matches.forEach((m) => {
    perSource.set(m.source_id, (perSource.get(m.source_id) ?? 0) + (m.query_span[1] - m.query_span[0]))
  })
  const total = Math.max(1, [...perSource.values()].reduce((a, b) => a + b, 0))
  return matches.filter((m) => {
    if (m.words < options.minWords) return false
    if (options.quotes && m.in_quote) return false
    if (options.bibliography && m.in_bibliography) return false
    const share = (100 * (perSource.get(m.source_id) ?? 0)) / total
    return share >= options.hideBelowPercent
  })
}

export function sourceColor(index: number): string {
  return `var(--source-${index % 12})`
}

/** Group token flags by sentence so the inspector can show the evidence. */
export function flaggedTokens(tokens: TokenScore[], sentence: SentenceScore): TokenScore[] {
  return tokens.filter((t) => t.flagged && t.start >= sentence.start && t.end <= sentence.end)
}
