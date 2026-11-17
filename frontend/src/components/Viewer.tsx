/** Document viewer (pure DOM). Three modes: AI heatmap, plagiarism spans, and clean.
 * Clicking a sentence opens the inspector; clicking a plagiarism span opens the
 * side-by-side comparison. Document text is rendered as text nodes only — never as
 * HTML — so uploads cannot inject markup. */
import { useMemo, useState } from 'react'
import type { AIResult, PlagMatch, PlagiarismResult } from '../types'
import { buildSpans, heatColor, matchesToRanges, sourceColor, sentencesToRanges } from '../lib/spans'

type View = 'ai' | 'plagiarism' | 'clean'

export function Viewer({
  text,
  ai,
  plagiarism,
  visibleMatches,
  onSentence,
  onMatch,
}: {
  text: string
  ai: AIResult | null
  plagiarism: PlagiarismResult | null
  visibleMatches: PlagMatch[]
  onSentence: (index: number) => void
  onMatch: (match: PlagMatch) => void
}) {
  const [view, setView] = useState<View>('ai')

  const spans = useMemo(() => {
    if (view === 'ai' && ai) return buildSpans(text, sentencesToRanges(ai.sentences))
    if (view === 'plagiarism') return buildSpans(text, matchesToRanges(visibleMatches))
    return [{ text, start: 0, end: text.length }]
  }, [view, text, ai, visibleMatches])

  const sentenceFor = (start: number, end: number) =>
    ai?.sentences.findIndex((s) => s.start <= start && end <= s.end) ?? -1

  const matchFor = (index: number) => visibleMatches[index]

  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-2">
        <div role="tablist" aria-label="Viewer mode" className="flex gap-1">
          {([
            ['ai', 'AI heatmap'],
            ['plagiarism', 'Plagiarism'],
            ['clean', 'Clean'],
          ] as const).map(([key, label]) => (
            <button
              key={key}
              role="tab"
              aria-selected={view === key}
              className="btn !rounded-lg !px-2.5 !py-1 text-xs"
              style={view === key ? { backgroundColor: 'var(--surface-2)', borderColor: 'var(--accent)' } : undefined}
              onClick={() => setView(key)}
              disabled={key === 'ai' ? !ai : key === 'plagiarism' ? !plagiarism : false}
            >
              {label}
            </button>
          ))}
        </div>
        {view === 'ai' && ai && (
          <span className="muted text-xs">
            sentence heatmap · dashed underline = high machine score
            {ai.ai_fraction !== null && ` · mixed authorship: ${(ai.ai_fraction * 100).toFixed(0)}% AI`}
          </span>
        )}
      </div>

      {view === 'ai' && ai?.segments.length ? (
        <MixedStripe ai={ai} />
      ) : null}

      <div
        data-testid="document-viewer"
        className="max-h-[32rem] overflow-auto whitespace-pre-wrap rounded-card border p-3 font-sans text-sm leading-7"
        style={{ borderColor: 'var(--border)' }}
      >
        {spans.map((span, index) => {
          if (!span.matchIndexes?.length) return <span key={index}>{span.text}</span>
          if (view === 'ai') {
            const sentenceIndex = span.matchIndexes[0]
            const sentence = ai?.sentences[sentenceIndex]
            if (!sentence) return <span key={index}>{span.text}</span>
            return (
              <button
                key={index}
                className="rounded-sm px-0.5 text-left"
                style={{
                  backgroundColor: heatColor(sentence.ai_prob),
                  textDecoration: sentence.ai_prob >= 0.75 ? 'underline dashed' : 'none',
                  textUnderlineOffset: '4px',
                }}
                title={`Sentence ${sentenceIndex + 1}: ${(sentence.ai_prob * 100).toFixed(1)}% machine-like`}
                onClick={() => onSentence(sentenceFor(span.start, span.end))}
              >
                {span.text}
              </button>
            )
          }
          const match = matchFor(span.matchIndexes[0])
          if (!match) return <span key={index}>{span.text}</span>
          return (
            <button
              key={index}
              className="rounded-sm px-0.5 text-left"
              style={{
                backgroundColor: `color-mix(in srgb, ${sourceColor(match.color_index)} 32%, transparent)`,
                borderBottom: `2px solid ${sourceColor(match.color_index)}`,
              }}
              title={`${match.match_type} match with ${match.source_title} (${(match.score * 100).toFixed(0)}%)`}
              onClick={() => onMatch(match)}
            >
              {span.text}
            </button>
          )
        })}
      </div>
    </div>
  )
}

function MixedStripe({ ai }: { ai: AIResult }) {
  const total = Math.max(1, ai.segments.reduce((sum, segment) => sum + (segment.end - segment.start), 0))
  return (
    <div className="mb-2">
      <div className="flex h-2.5 w-full overflow-hidden rounded-full" aria-hidden="true">
        {ai.segments.map((segment, index) => (
          <div
            key={index}
            style={{
              width: `${((segment.end - segment.start) / total) * 100}%`,
              backgroundColor: segment.label === 'ai' ? 'var(--ai)' : 'var(--human)',
              opacity: 0.75,
            }}
          />
        ))}
      </div>
      <p className="muted mt-1 text-xs">
        Mixed-authorship segmentation (2-state HMM): {ai.segments.filter((s) => s.label === 'ai').length} machine-like
        stretch(es) of {ai.segments.length} total.
      </p>
    </div>
  )
}
