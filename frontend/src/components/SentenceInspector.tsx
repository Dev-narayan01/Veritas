/** Evidence for one sentence: component breakdown, flagged tokens and the smoothing
 * caveat. Opened from the heatmap; always available as DOM text. */
import type { AIResult } from '../types'
import { flaggedTokens } from '../lib/spans'
import { percent } from '../lib/format'

export function SentenceInspector({ ai, index, onClose }: { ai: AIResult; index: number; onClose: () => void }) {
  const sentence = ai.sentences[index]
  if (!sentence) return null
  const tokens = flaggedTokens(ai.tokens, sentence).slice(0, 24)
  return (
    <div className="card-2 p-3 text-sm" role="dialog" aria-label={`Sentence ${index + 1} evidence`}>
      <div className="mb-2 flex items-start justify-between gap-2">
        <p className="panel-title">Sentence {index + 1} · {percent(sentence.ai_prob)} machine-like</p>
        <button className="btn !border-0 !p-1 text-xs" onClick={onClose} aria-label="Close sentence evidence">
          ✕
        </button>
      </div>
      <blockquote className="mb-2 border-l-2 pl-2 italic" style={{ borderColor: 'var(--border)' }}>
        {sentence.text}
      </blockquote>
      <p className="muted mb-1 text-xs">
        Scored in a ±1 sentence context window and smoothed across three sentences, which is why
        neighbouring sentences share evidence.
      </p>
      {tokens.length > 0 && (
        <div className="mt-2">
          <p className="panel-title mb-1">Most predictable tokens</p>
          <div className="flex flex-wrap gap-1">
            {tokens.map((token, i) => (
              <span key={i} className="chip font-mono text-[11px]" title={`surprisal ${token.surprisal}`}>
                {token.token}
              </span>
            ))}
          </div>
        </div>
      )}
      <p className="muted mt-2 text-xs">
        Component scores: {ai.components.filter((c) => c.probability !== null).map((c) => `${c.name} ${percent(c.probability)}`).join(' · ') || '—'}
      </p>
    </div>
  )
}
