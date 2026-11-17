/** Results page: sticky summary bar, viewer + evidence columns, similarity report with
 * client-side filters, integrity, explanation, component chart, batch and code sections,
 * and the export actions. Every number on this page comes from the API result. */
import { lazy, Suspense, useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { Download, FileJson, FileText, Trash2 } from 'lucide-react'
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { toast } from 'sonner'
import { api, apiUrl, verdictColor, verdictLabel } from '../lib/api'
import { filterMatches, sourceColor } from '../lib/spans'
import { percent, plainPercent, relativeDate, truncate } from '../lib/format'
import type { AnalysisResult, DocumentResult, PlagMatch } from '../types'
import { useUi } from '../store'
import { useGraphicsPreference, useGraphicsTier } from '../hooks/useGraphicsTier'
import { Badge, Callout, Card, EmptyState, Skeleton } from '../components/ui'
import { Viewer } from '../components/Viewer'
import { SentenceInspector } from '../components/SentenceInspector'

const VerdictGauge = lazy(() => import('../components/scenes/VerdictGauge').then((m) => ({ default: m.VerdictGauge })))
const SimilarityGalaxy = lazy(() => import('../components/scenes/SimilarityGalaxy').then((m) => ({ default: m.SimilarityGalaxy })))

export function ResultsPage() {
  const { id = '' } = useParams()
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [activeDoc, setActiveDoc] = useState(0)
  const [sentence, setSentence] = useState<number | null>(null)
  const [selectedMatch, setSelectedMatch] = useState<PlagMatch | null>(null)
  const ui = useUi()
  const [preference] = useGraphicsPreference()
  const { tier } = useGraphicsTier(preference)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const payload = await api.analysis(id)
        if (cancelled) return
        if (payload.status === 'error') {
          setError(payload.error ?? 'The analysis failed')
        } else if (payload.status !== 'done') {
          setTimeout(load, 1500)
        } else {
          setResult(payload)
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err))
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [id])

  const document: DocumentResult | undefined = result?.documents[activeDoc]
  const visibleMatches = useMemo(() => {
    if (!document?.plagiarism) return []
    return filterMatches(document.plagiarism.matches, {
      quotes: ui.excludeQuotes,
      bibliography: ui.excludeBibliography,
      minWords: ui.minMatchWords,
      hideBelowPercent: ui.hideBelowPercent,
    })
  }, [document, ui.excludeQuotes, ui.excludeBibliography, ui.minMatchWords, ui.hideBelowPercent])

  if (error) {
    return (
      <EmptyState title="This analysis could not be shown" action={<Link className="btn" to="/">Start a new analysis</Link>}>
        {error}
      </EmptyState>
    )
  }
  if (!result) return <Skeleton lines={8} />
  if (!result.documents.length) return <EmptyState title="No documents in this analysis" />

  const ai = document?.ai ?? null
  const plagiarism = document?.plagiarism ?? null
  const componentData = (ai?.components ?? []).map((component) => ({
    name: component.name,
    value: component.probability === null ? 0 : Number((component.probability * 100).toFixed(1)),
    available: component.available,
  }))

  return (
    <div className="space-y-4">
      <div className="sticky top-14 z-30 card flex flex-wrap items-center gap-3 p-3">
        <div>
          <p className="text-sm font-medium">{result.documents.map((d) => d.filename).join(' · ')}</p>
          <p className="muted text-xs">{relativeDate(result.created_at)} · mode used: {ai?.mode_used ?? '—'}</p>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          {ai && <Badge tone={ai.verdict === 'likely_ai' ? 'bad' : ai.verdict === 'likely_human' ? 'good' : 'warn'}>{verdictLabel(ai.verdict)}</Badge>}
          {ai?.probability !== null && ai && <Badge>P(AI) {percent(ai.probability)}</Badge>}
          {plagiarism && <Badge tone="info">similarity {plainPercent(plagiarism.similarity_percent)}</Badge>}
          <a className="btn !px-2 !py-1 text-xs" href={apiUrl(`/api/analyses/${id}/report.pdf`)} target="_blank" rel="noreferrer">
            <Download size={13} aria-hidden="true" /> PDF
          </a>
          <a className="btn !px-2 !py-1 text-xs" href={apiUrl(`/api/analyses/${id}/export.json`)}>
            <FileJson size={13} aria-hidden="true" /> JSON
          </a>
          <a className="btn !px-2 !py-1 text-xs" href={apiUrl(`/api/analyses/${id}/export.csv`)}>
            <FileText size={13} aria-hidden="true" /> CSV
          </a>
          <button
            className="btn !px-2 !py-1 text-xs"
            onClick={async () => {
              try {
                await api.deleteAnalysis(id)
                toast.success('Analysis deleted')
                window.location.href = '/history'
              } catch (err) {
                toast.error(err instanceof Error ? err.message : 'Delete failed')
              }
            }}
          >
            <Trash2 size={13} aria-hidden="true" /> Delete
          </button>
        </div>
      </div>

      <Callout tone="warn" title="Disclaimer">
        {result.disclaimer || 'AI-text detection is probabilistic and must never be the sole basis for an accusation.'}
      </Callout>

      {result.documents.length > 1 && (
        <div role="tablist" aria-label="Documents" className="flex flex-wrap gap-1">
          {result.documents.map((doc, index) => (
            <button
              key={doc.id}
              role="tab"
              aria-selected={index === activeDoc}
              className="btn !rounded-lg !px-3 !py-1.5 text-xs"
              style={index === activeDoc ? { backgroundColor: 'var(--surface-2)', borderColor: 'var(--accent)' } : undefined}
              onClick={() => {
                setActiveDoc(index)
                setSentence(null)
                setSelectedMatch(null)
              }}
            >
              {truncate(doc.filename, 28)}
              {doc.ai && (
                <span className="font-mono" style={{ color: verdictColor(doc.ai.verdict) }}>
                  {doc.ai.probability === null ? '—' : percent(doc.ai.probability, 0)}
                </span>
              )}
            </button>
          ))}
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <Card title={`Document · ${document?.kind ?? ''} · ${document?.words ?? 0} words`}>
          {document && (
            <Viewer
              text={document.text}
              ai={ai}
              plagiarism={plagiarism}
              visibleMatches={visibleMatches}
              onSentence={setSentence}
              onMatch={setSelectedMatch}
            />
          )}
        </Card>

        <div className="space-y-4">
          <Card title="Verdict">
            {ai ? (
              <div className="flex flex-col items-center gap-2">
                {ai.probability !== null ? (
                  <>
                    <Suspense fallback={<Skeleton lines={3} />}>
                      <VerdictGauge
                        tier={tier}
                        probability={ai.probability}
                        tHuman={ai.thresholds.t_human ?? 0}
                        tAi={ai.thresholds.t_ai ?? 1}
                        color={verdictColor(ai.verdict)}
                      />
                    </Suspense>
                    <p className="font-mono text-2xl" data-testid="verdict-text" style={{ color: verdictColor(ai.verdict) }}>
                      {percent(ai.probability)}
                    </p>
                    <p className="text-sm">{verdictLabel(ai.verdict)}</p>
                    <p className="muted text-center text-xs">
                      95% interval {ai.ci_low === null ? '—' : percent(ai.ci_low)} – {ai.ci_high === null ? '—' : percent(ai.ci_high)}
                      <br />
                      thresholds: human ≤ {ai.thresholds.t_human === null ? '—' : percent(ai.thresholds.t_human)} · AI ≥{' '}
                      {ai.thresholds.t_ai === null ? '—' : percent(ai.thresholds.t_ai)}
                      {ai.band_collapsed && ' (band collapsed)'}
                    </p>
                  </>
                ) : (
                  <p className="muted text-sm">No probability: {ai.verdict.replace('_', ' ')}</p>
                )}
                <div className="mt-1 flex flex-wrap justify-center gap-1">
                  {ai.low_confidence && <Badge tone="warn">low confidence</Badge>}
                  <Badge>{ai.mode_used ?? 'no mode'} mode</Badge>
                  {ai.components_run.map((name) => (
                    <Badge key={name} tone="good">{name}</Badge>
                  ))}
                  {ai.components_unavailable.map((item) => (
                    <Badge key={item.component} tone="bad" title={item.reason}>{item.component} unavailable</Badge>
                  ))}
                </div>
                {ai.warnings.map((warning) => (
                  <p key={warning} className="muted text-center text-xs">{warning}</p>
                ))}
              </div>
            ) : (
              <p className="muted text-sm">AI analysis was not requested for this document.</p>
            )}
          </Card>

          {sentence !== null && ai && (
            <SentenceInspector ai={ai} index={sentence} onClose={() => setSentence(null)} />
          )}

          {plagiarism && (
            <Card
              title="Similarity report"
              actions={
                <span className="muted text-xs">
                  {plainPercent(plagiarism.similarity_percent)} raw · {plainPercent(plagiarism.similarity_percent_filtered)} after exclusions
                </span>
              }
            >
              <div className="space-y-2">
                <div className="flex flex-wrap gap-1 text-xs">
                  <Badge>lexical {plainPercent(plagiarism.lexical_percent)}</Badge>
                  <Badge>paraphrase {plainPercent(plagiarism.paraphrase_percent)}</Badge>
                  <Badge>online {plainPercent(plagiarism.online_percent)}</Badge>
                  <Badge title="Number of documents really indexed on this server">corpus {plagiarism.corpus_size}</Badge>
                </div>
                <div className="space-y-1">
                  {plagiarism.sources.length === 0 && <p className="muted text-xs">No matching sources were found.</p>}
                  {plagiarism.sources.map((source) => (
                    <div key={source.id} className="flex items-center gap-2 text-xs">
                      <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: sourceColor(source.color_index) }} aria-hidden="true" />
                      <span className="flex-1 truncate">{source.title}</span>
                      <span className="font-mono">{plainPercent(source.percent)}</span>
                      <span className="muted">{source.matches}×</span>
                    </div>
                  ))}
                </div>
                <div className="flex flex-wrap gap-3 border-t pt-2 text-xs" style={{ borderColor: 'var(--border)' }}>
                  <label className="flex items-center gap-1">
                    <input type="checkbox" checked={ui.excludeQuotes} onChange={(e) => ui.setExcludeQuotes(e.target.checked)} /> quotes
                  </label>
                  <label className="flex items-center gap-1">
                    <input type="checkbox" checked={ui.excludeBibliography} onChange={(e) => ui.setExcludeBibliography(e.target.checked)} /> bibliography
                  </label>
                  <label className="flex flex-1 items-center gap-1">
                    min words <span className="font-mono">{ui.minMatchWords}</span>
                    <input type="range" min={3} max={40} value={ui.minMatchWords} onChange={(e) => ui.setMinMatchWords(Number(e.target.value))} className="flex-1 accent-[var(--accent)]" />
                  </label>
                  <label className="flex flex-1 items-center gap-1">
                    hide &lt; <span className="font-mono">{ui.hideBelowPercent}%</span>
                    <input type="range" min={0} max={10} step={0.5} value={ui.hideBelowPercent} onChange={(e) => ui.setHideBelowPercent(Number(e.target.value))} className="flex-1 accent-[var(--accent)]" />
                  </label>
                </div>
                {selectedMatch && (
                  <div className="card-2 p-2 text-xs" role="dialog" aria-label="Match evidence">
                    <div className="mb-1 flex items-center justify-between">
                      <span className="font-medium">{selectedMatch.source_title}</span>
                      <button className="btn !border-0 !p-1" onClick={() => setSelectedMatch(null)} aria-label="Close match evidence">✕</button>
                    </div>
                    <p className="muted mb-1">
                      {selectedMatch.match_type} · score {(selectedMatch.score * 100).toFixed(0)}% · {selectedMatch.words} words
                      {selectedMatch.source_url && (
                        <>
                          {' · '}
                          <a className="underline" href={selectedMatch.source_url} target="_blank" rel="noreferrer">source</a>
                        </>
                      )}
                    </p>
                    <p className="mb-1 rounded bg-black/20 p-1.5">{truncate(selectedMatch.query_text, 260)}</p>
                    <p className="rounded bg-black/20 p-1.5">{truncate(selectedMatch.source_text, 260)}</p>
                  </div>
                )}
                {plagiarism.online_snippets_sent.length > 0 && (
                  <details className="text-xs">
                    <summary className="cursor-pointer">What was sent online ({plagiarism.online_snippets_sent.length} snippets)</summary>
                    <ul className="muted mt-1 list-disc pl-4">
                      {plagiarism.online_snippets_sent.map((snippet, index) => (
                        <li key={index}>{truncate(snippet, 120)}</li>
                      ))}
                    </ul>
                  </details>
                )}
                {plagiarism.warnings.map((warning) => (
                  <p key={warning} className="muted text-xs">• {warning}</p>
                ))}
              </div>
            </Card>
          )}

          {document && document.integrity.flags.length > 0 && (
            <Card title="Integrity flags">
              <ul className="space-y-1 text-xs">
                {document.integrity.flags.map((flag) => (
                  <li key={flag.kind}>
                    <span className="font-medium">{flag.kind.replace(/_/g, ' ')}:</span> {flag.count}
                    {flag.detail && <span className="muted"> — {flag.detail}</span>}
                    {flag.examples.length > 0 && <span className="muted font-mono"> [{flag.examples.slice(0, 4).join(', ')}]</span>}
                  </li>
                ))}
              </ul>
            </Card>
          )}

          {ai?.explanation && (
            <Card title={`Explanation · ${ai.explanation.source === 'local_llm' ? 'Local LLM' : 'Rule-based'}`}>
              <p className="text-sm leading-6">{ai.explanation.text}</p>
            </Card>
          )}

          {componentData.length > 0 && (
            <Card title="Component breakdown">
              <div className="h-40">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={componentData} layout="vertical" margin={{ left: 8, right: 16 }}>
                    <XAxis type="number" domain={[0, 100]} hide />
                    <YAxis type="category" dataKey="name" width={80} tick={{ fontSize: 10, fill: 'var(--muted)' }} />
                    <Tooltip formatter={(value) => `${value}%`} />
                    <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                      {componentData.map((entry) => (
                        <Cell key={entry.name} fill={entry.value >= 50 ? 'var(--ai)' : 'var(--human)'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="muted text-xs">
                Each bar is that component's own probability of machine authorship. The verdict uses the
                calibrated stacker over these values, not an average.
              </p>
            </Card>
          )}
        </div>
      </div>

      {result.batch && (
        <Card title="Batch comparison">
          <BatchSection result={result} tier={tier} />
        </Card>
      )}

      {result.code_matches.length > 0 && (
        <Card title="Code similarity">
          <ul className="space-y-2 text-sm">
            {result.code_matches.map((match, index) => (
              <li key={index} className="card-2 p-2">
                <p className="flex items-center justify-between">
                  <span>{match.a} ↔ {match.b}</span>
                  <span className="font-mono">{plainPercent(match.similarity * 100)}</span>
                </p>
                <div className="mt-1 h-1.5 rounded-full" style={{ backgroundColor: 'var(--surface)' }}>
                  <div className="h-full rounded-full" style={{ width: `${match.similarity * 100}%`, backgroundColor: 'var(--accent)' }} />
                </div>
                <p className="muted mt-1 text-xs">
                  matched lines A: {match.lines_a.slice(0, 12).join(', ') || '—'} · B: {match.lines_b.slice(0, 12).join(', ') || '—'}
                  {match.functions.length > 0 &&
                    ` · matching functions: ${(match.function_pairs ?? match.functions.map((name) => [name, name] as [string, string]))
                      .slice(0, 8)
                      .map(([original, copy]) => (original === copy ? original : `${original} → ${copy}`))
                      .join(', ')}`}
                </p>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {result.exit_notes.length > 0 && (
        <Card title="Notes">
          <ul className="muted space-y-1 text-xs">
            {result.exit_notes.map((note) => (
              <li key={note}>• {note}</li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  )
}

function BatchSection({ result, tier }: { result: AnalysisResult; tier: 'high' | 'low' | 'off' }) {
  const [tab, setTab] = useState<'graph' | 'matrix' | 'pairs'>('matrix')
  const batch = result.batch!
  return (
    <div>
      <div className="mb-2 flex gap-1">
        {(['graph', 'matrix', 'pairs'] as const).map((key) => (
          <button key={key} className="btn !rounded-lg !px-2.5 !py-1 text-xs capitalize" aria-pressed={tab === key} onClick={() => setTab(key)} style={tab === key ? { backgroundColor: 'var(--surface-2)' } : undefined}>
            {key}
          </button>
        ))}
      </div>
      {tab === 'graph' && (
        <Suspense fallback={<Skeleton lines={4} />}>
          <SimilarityGalaxy tier={tier} batch={batch} onSelect={() => setTab('pairs')} />
        </Suspense>
      )}
      {tab === 'matrix' && (
        <div className="overflow-auto">
          <table className="w-full text-xs">
            <caption className="sr-only">Percent of the row document found in the column document</caption>
            <thead>
              <tr>
                <th className="p-1 text-left">document</th>
                {batch.labels.map((label) => (
                  <th key={label} className="p-1 text-left">{truncate(label, 16)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {batch.matrix.map((row, i) => (
                <tr key={batch.labels[i]}>
                  <th scope="row" className="p-1 text-left font-normal">{truncate(batch.labels[i], 22)}</th>
                  {row.map((value, j) => (
                    <td key={j} className="p-1 font-mono" style={{ backgroundColor: `rgba(124,92,255,${Math.min(0.5, value / 200)})` }}>
                      {value.toFixed(1)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {tab === 'pairs' && (
        <ul className="space-y-2 text-sm">
          {batch.pairs.map((pair, index) => (
            <li key={index} className="card-2 p-2">
              <p className="flex justify-between"><span>{pair.a} ↔ {pair.b}</span><span className="font-mono">{plainPercent(pair.percent_ab)} / {plainPercent(pair.percent_ba)}</span></p>
              <ul className="muted mt-1 text-xs">
                {pair.segments.map((segment, s) => (
                  <li key={s}>{String(segment.type)} · {String(segment.words)} words</li>
                ))}
              </ul>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
