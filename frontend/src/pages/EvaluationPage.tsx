/** Evaluation page. Order matters and is enforced here: badges, then HELD-OUT (OOD)
 * results, then in-domain labelled "optimistic", then ablation, cross-generator,
 * third-party baseline (OOD only), robustness (with changed_fraction), bias (with CIs),
 * the plagiarism benchmark (labelled synthetic) and the embedding projection.
 * Every value is read from /api/metrics, which serves the JSON written by ml/ scripts. */
import { lazy, Suspense, useEffect, useState } from 'react'
import { AlertTriangle } from 'lucide-react'
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, apiUrl } from '../lib/api'
import { percent, plainPercent } from '../lib/format'
import type { MetricsResponse } from '../types'
import { useGraphicsPreference, useGraphicsTier } from '../hooks/useGraphicsTier'
import { Badge, Callout, Card, EmptyState, Skeleton } from '../components/ui'

const EmbeddingSpace = lazy(() => import('../components/scenes/EmbeddingSpace').then((m) => ({ default: m.EmbeddingSpace })))

const COMPONENT_KEYS = ['roc_auc', 'pr_auc', 'tpr_at_1pct_fpr', 'tpr_at_0.1pct_fpr', 'accuracy', 'f1', 'ece']

export function EvaluationPage() {
  const [data, setData] = useState<MetricsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [preference] = useGraphicsPreference()
  const { tier } = useGraphicsTier(preference)

  useEffect(() => {
    api.metrics().then(setData).catch((err) => setError(err instanceof Error ? err.message : String(err)))
  }, [])

  if (error) return <EmptyState title="Evaluation data unavailable">{error}</EmptyState>
  if (!data) return <Skeleton lines={8} />

  const metrics = (data.metrics ?? {}) as Record<string, any>
  const splits = (metrics.splits ?? {}) as Record<string, any>
  const ood = splits.test_ood
  const inDomain = splits.test
  const manifest = (data.manifest ?? {}) as Record<string, any>
  const smoke = Boolean(manifest.smoke) || manifest.run === 'smoke'

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <h1 className="mr-auto text-2xl font-semibold">Evaluation</h1>
        <Badge tone={data.artifact_source === 'kaggle' ? 'good' : data.artifact_source === 'cpu' ? 'warn' : 'bad'}>
          {data.artifact_source === 'kaggle'
            ? 'GPU-trained artifacts'
            : data.artifact_source === 'cpu'
              ? 'CPU-trained baseline artifacts'
              : 'no artifacts'}
        </Badge>
        <Badge tone={data.data_source === 'hc3' ? 'good' : 'warn'}>data: {data.data_source}</Badge>
        {smoke && <Badge tone="bad">SMOKE RUN — not results</Badge>}
      </div>

      {smoke && (
        <Callout tone="bad" title="Smoke run">
          These numbers come from a smoke run whose only purpose is to prove the pipeline executes.
          They must not be quoted as results.
        </Callout>
      )}
      {(metrics.notes ?? []).map((note: string) => (
        <Callout key={note} tone="warn">{note}</Callout>
      ))}

      {!ood && !inDomain && (
        <EmptyState title="No metrics have been produced yet">
          Run <code className="font-mono">python ml/run_all.py --smoke</code> locally, or the GPU pipeline on
          Kaggle, then reload this page. Nothing here is invented when the files are missing.
        </EmptyState>
      )}

      {ood && <ResultsBlock title="Held-out (out-of-domain) — the headline" split={ood} tone="good" />}
      {inDomain && <ResultsBlock title="In-domain (optimistic)" split={inDomain} tone="warn" />}

      {Object.keys(metrics.ablation ?? {}).length > 0 && (
        <Card title="Ablation — modes side by side (held-out split first)">
          <div className="overflow-auto">
            <table className="w-full text-sm">
              <thead>
                <tr>
                  {['split', 'mode', 'components', 'ROC-AUC', 'TPR @1% FPR', 'F1'].map((h) => (
                    <th key={h} className="panel-title p-2 text-left">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(metrics.ablation).map(([split, rows]: [string, any]) =>
                  Object.entries(rows).map(([mode, entry]: [string, any]) => (
                    <tr key={`${split}-${mode}`} className="border-t" style={{ borderColor: 'var(--border)' }}>
                      <td className="p-2">{split === 'test_ood' ? 'held-out' : 'in-domain'}</td>
                      <td className="p-2 capitalize">{mode}</td>
                      <td className="p-2 text-xs">{(entry.components ?? []).join(' + ')}</td>
                      <td className="p-2 font-mono">{entry.roc_auc?.toFixed?.(3) ?? '—'}</td>
                      <td className="p-2 font-mono">{entry.tpr_at_1pct_fpr?.toFixed?.(3) ?? '—'}</td>
                      <td className="p-2 font-mono">{entry.f1?.toFixed?.(3) ?? '—'}</td>
                    </tr>
                  )),
                )}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <Figures plots={data.plots} base={data.plots_base} />

      <ConfusionAndLength data={data} />

      {data.cross_generator && (
        <Card title="Cross-generator transfer (leave-one-generator-out)">
          {Object.keys((data.cross_generator as any).generators ?? {}).length === 0 ? (
            <p className="muted text-sm">{((data.cross_generator as any).notes ?? []).join(' ')}</p>
          ) : (
            <ul className="space-y-1 text-sm">
              {Object.entries((data.cross_generator as any).generators).map(([name, entry]: [string, any]) => (
                <li key={name} className="flex justify-between">
                  <span>held out {name}</span>
                  <span className="font-mono">
                    AUC {entry.roc_auc ?? '—'} · drop {entry.drop_vs_in_domain ?? '—'}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Card>
      )}

      {data.baseline && (
        <Card
          title={
            <span className="flex items-center gap-2">
              Third-party baseline <Badge tone="warn">third-party, OOD only</Badge>
            </span>
          }
        >
          {(data.baseline as any).available === false ? (
            <p className="muted text-sm">
              Not reported: {String((data.baseline as any).reason ?? 'unavailable')}. No number is shown
              rather than a guessed one.
            </p>
          ) : (
            <div className="text-sm">
              <p className="muted mb-1">{String((data.baseline as any).label ?? '')}</p>
              <p className="font-mono">
                AUC {(data.baseline as any).test_ood?.roc_auc ?? '—'} · TPR at 1% FPR{' '}
                {(data.baseline as any).test_ood?.tpr_at_1pct_fpr ?? '—'} · FPR{' '}
                {(data.baseline as any).test_ood?.['human_fpr_at_0.5'] ?? '—'}
              </p>
              <p className="muted mt-1 text-xs">{((data.baseline as any).notes ?? []).join(' ')}</p>
            </div>
          )}
        </Card>
      )}

      {data.robustness && (
        <Card title="Robustness (recall at each mode's tuned threshold)">
          <div className="overflow-auto">
            <table className="w-full text-xs">
              <thead>
                <tr>
                  {['mode', 'perturbation', 'recall before', 'recall after', 'changed', 'with integrity'].map((h) => (
                    <th key={h} className="panel-title p-1.5 text-left">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries((data.robustness as any).modes ?? {}).map(([mode, payload]: [string, any]) =>
                  Object.entries(payload.perturbations ?? {}).map(([name, entry]: [string, any]) => (
                    <tr key={`${mode}-${name}`} className="border-t" style={{ borderColor: 'var(--border)' }}>
                      <td className="p-1.5 capitalize">{mode}</td>
                      <td className="p-1.5">{name.replace(/_/g, ' ')}</td>
                      <td className="p-1.5 font-mono">{entry.recall_before}</td>
                      <td className="p-1.5 font-mono">{entry.recall_after}</td>
                      <td className="p-1.5 font-mono" style={{ color: entry.changed_fraction < 0.5 ? 'var(--ai)' : undefined }}>
                        {percent(entry.changed_fraction, 0)}
                      </td>
                      <td className="p-1.5 font-mono">{entry.recall_after_integrity_cleaning ?? '—'}</td>
                    </tr>
                  )),
                )}
              </tbody>
            </table>
          </div>
          <p className="muted mt-1 text-xs">{((data.robustness as any).notes ?? []).join(' ')}</p>
        </Card>
      )}

      {data.bias && (
        <Card title="Bias check (human false-positive rate)">
          <p className="muted mb-2 text-xs">{String((data.bias as any).proxy_definition ?? '')}</p>
          <div className="overflow-auto">
            <table className="w-full text-xs">
              <thead>
                <tr>
                  {['mode', 'normal FPR', 'proxy FPR', 'difference', '95% CI', 'flagged'].map((h) => (
                    <th key={h} className="panel-title p-1.5 text-left">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries((data.bias as any).modes ?? {}).map(([mode, entry]: [string, any]) => (
                  <tr key={mode} className="border-t" style={{ borderColor: 'var(--border)' }}>
                    <td className="p-1.5 capitalize">{mode}</td>
                    <td className="p-1.5 font-mono">{plainPercent(entry.normal_human_fpr * 100)}</td>
                    <td className="p-1.5 font-mono">{plainPercent(entry.proxy_human_fpr * 100)}</td>
                    <td className="p-1.5 font-mono">{plainPercent(entry.difference * 100)}</td>
                    <td className="p-1.5 font-mono">[{plainPercent(entry.difference_ci95[0] * 100)}, {plainPercent(entry.difference_ci95[1] * 100)}]</td>
                    <td className="p-1.5">
                      {entry.higher_false_positive_rate_on_proxy ? <Badge tone="bad">yes</Badge> : <Badge tone="good">no</Badge>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted mt-1 text-xs">{((data.bias as any).notes ?? []).join(' ')}</p>
        </Card>
      )}

      {data.plag_benchmark_results && (
        <Card title={<span className="flex items-center gap-2">Plagiarism benchmark <Badge tone="warn">synthetic</Badge></span>}>
          <p className="muted mb-1 text-xs">{String((data.plag_benchmark_results as any).caption ?? '')}</p>
          <div className="grid gap-2 sm:grid-cols-3">
            <Metric label="span F1 (overall)" value={plainPercent((data.plag_benchmark_results as any).span_f1_overall * 100)} />
            <Metric
              label="false alarms on negatives"
              value={plainPercent((data.plag_benchmark_results as any).false_alarm_rate_on_negatives * 100)}
            />
            <Metric label="cases" value={String((data.plag_benchmark_results as any).cases ?? '—')} />
          </div>
          <ul className="muted mt-2 space-y-1 text-xs">
            {Object.entries((data.plag_benchmark_results as any).span_f1_by_obfuscation ?? {}).map(([name, value]) => (
              <li key={name}>
                {name.replace(/_/g, ' ')}: F1 {String(value)}
              </li>
            ))}
            {((data.plag_benchmark_results as any).notes ?? []).map((note: string) => (
              <li key={note}>• {note}</li>
            ))}
          </ul>
        </Card>
      )}

      {data.embedding_projection && (
        <Card title="Feature space">
          <Suspense fallback={<Skeleton lines={4} />}>
            <EmbeddingSpace tier={tier} data={data.embedding_projection as never} />
          </Suspense>
        </Card>
      )}

      {!data.metrics && (
        <Callout tone="info" title="Nothing is fabricated">
          <span className="flex items-center gap-1">
            <AlertTriangle size={14} aria-hidden="true" />
            When an artifact is missing this page says so instead of showing a placeholder number.
          </span>
        </Callout>
      )}
    </div>
  )
}

const FIGURE_ORDER = [
  'roc',
  'pr',
  'calibration',
  'distributions',
  'confusion',
  'accuracy_by_length',
  'split_lengths',
]

const FIGURE_TITLES: Record<string, string> = {
  roc: 'ROC curves — every component and mode on the same axes',
  pr: 'Precision–recall — the honest view when the classes are this lopsided',
  calibration: 'Reliability diagram — before and after Platt scaling',
  distributions: 'Score distributions with the two published thresholds',
  confusion: 'Confusion matrix at the reported thresholds',
  accuracy_by_length: 'Accuracy by document length at those same thresholds',
  split_lengths: 'Document lengths per split — the length confound, exposed',
}

/** The generated figures, served from the artifact bundle.
 *
 * OOD first: the plots are ordered so the held-out variant of each chart comes before the
 * in-domain one. The images are produced by `ml/evaluate.py` in the same run as the tables, so
 * a figure can never describe a different model than the numbers above it. */
function Figures({ plots, base }: { plots: string[]; base: string | null }) {
  const [onlyOod, setOnlyOod] = useState(false)
  if (!plots || plots.length === 0 || !base) {
    return (
      <Card title="Figures">
        <p className="muted text-sm">
          No figures are in this bundle. They are written by <code>ml/evaluate.py</code> into
          <code> ml/artifacts/plots/</code>; re-run it and reload.
        </p>
      </Card>
    )
  }
  const rank = (name: string) => {
    const key = FIGURE_ORDER.findIndex((token) => name.startsWith(token))
    const ood = name.includes('_ood') ? 0 : 1
    return (key === -1 ? FIGURE_ORDER.length : key) * 2 + ood
  }
  const ordered = [...plots].sort((a, b) => rank(a) - rank(b))
  const visible = onlyOod ? ordered.filter((name) => name.includes('_ood')) : ordered
  return (
    <Card title="Figures">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <p className="muted text-xs">
          Generated by <code>ml/evaluate.py</code> from the same run as the tables above. Held-out
          figures are listed first for every chart type.
        </p>
        <label className="flex cursor-pointer items-center gap-2 text-xs">
          <input type="checkbox" checked={onlyOod} onChange={() => setOnlyOod((value) => !value)} />
          held-out only
        </label>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {visible.map((name) => {
          const token = FIGURE_ORDER.find((key) => name.startsWith(key)) ?? ''
          const title = FIGURE_TITLES[token] ?? name
          return (
            <figure key={name} className="card-2 p-2">
              <img
                src={apiUrl(`${base}/${name}`)}
                alt={`${title} (${name.includes('_ood') ? 'held-out' : 'in-domain'})`}
                loading="lazy"
                className="w-full rounded"
              />
              <figcaption className="muted mt-1 text-xs">
                {title}
                <span className="ml-1 font-mono">
                  · {name.includes('_ood') ? 'held-out' : 'in-domain'}
                </span>
              </figcaption>
            </figure>
          )
        })}
      </div>
    </Card>
  )
}

/** Confusion matrices and accuracy-by-length, straight from `metrics.json`.
 * The matrix counts *reported verdicts*, so abstentions appear as their own cells rather
 * than being folded into "correct". */
function ConfusionAndLength({ data }: { data: MetricsResponse }) {
  const splits = (((data.metrics ?? {}) as Record<string, any>).splits ?? {}) as Record<string, any>
  const rows: Array<{ split: string; mode: string; confusion: Record<string, number>; byLength: Record<string, { n: number; accuracy: number; abstained?: number }> }> = []
  for (const [splitName, split] of Object.entries(splits)) {
    for (const [mode, entry] of Object.entries((split?.modes ?? {}) as Record<string, any>)) {
      if (entry?.confusion) {
        rows.push({ split: splitName, mode, confusion: entry.confusion, byLength: entry.accuracy_by_length ?? {} })
      }
    }
  }
  if (rows.length === 0) return null
  const buckets = ['<100', '100-250', '250-500', '500+']
  return (
    <Card title="Confusion matrices and accuracy by length">
      <p className="muted mb-2 text-xs">
        Counts of the verdicts the system actually reported at the tuned thresholds.
        An abstention is reported as &ldquo;not AI&rdquo;, so it lands in the left-hand column; the
        number of abstentions is shown separately for each true class. That is why the
        false-negative cell is larger than the false-positive cell — that asymmetry is the 1%
        false-positive target working as intended.
      </p>
      <div className="grid gap-3 md:grid-cols-2">
        {rows.map((row) => {
          const c = row.confusion
          const cells: Array<[string, number, string]> = [
            ['true negative', c.human_true_negative ?? 0, 'var(--human)'],
            ['false positive', c.human_false_positive ?? 0, 'var(--ai)'],
            ['false negative', c.ai_false_negative ?? 0, 'var(--uncertain)'],
            ['true positive', c.ai_true_positive ?? 0, 'var(--human)'],
          ]
          const total = cells.reduce((sum, [, value]) => sum + value, 0) || 1
          return (
            <div key={`${row.split}-${row.mode}`} className="card-2 p-2">
              <p className="panel-title mb-1">
                {row.mode} · {row.split === 'test_ood' ? 'held-out' : 'in-domain'} · n={total}
              </p>
              <div className="grid grid-cols-2 gap-1 text-center text-xs">
                {cells.map(([label, value, color]) => (
                  <div key={label} className="rounded p-1" style={{ backgroundColor: 'var(--surface)' }}>
                    <p className="font-mono text-base" style={{ color }}>
                      {value}
                    </p>
                    <p className="muted">{label}</p>
                  </div>
                ))}
              </div>
              <p className="muted mt-1 text-[11px]">
                abstained: AI texts {c.ai_abstained ?? 0} · human texts {c.human_abstained ?? 0}
              </p>
              <table className="mt-2 w-full text-[11px]">
                <caption className="sr-only">
                  Accuracy by document length at the tuned thresholds, with abstention counts
                </caption>
                <thead>
                  <tr>
                    {buckets.map((bucket) => (
                      <th key={bucket} className="muted p-0.5 text-left font-normal">
                        {bucket}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    {buckets.map((bucket) => {
                      const entry = row.byLength[bucket]
                      return (
                        <td key={bucket} className="p-0.5 font-mono">
                          {entry ? entry.accuracy.toFixed(2) : '—'}
                          {entry && (
                            <span className="muted">
                              {' '}
                              (n={entry.n}
                              {entry.abstained !== undefined ? `, abst ${entry.abstained}` : ''})
                            </span>
                          )}
                        </td>
                      )
                    })}
                  </tr>
                </tbody>
              </table>
            </div>
          )
        })}
      </div>
      <BreakdownTable rows={rows} splits={splits} />
    </Card>
  )
}

function BreakdownTable({ rows, splits }: { rows: Array<{ split: string; mode: string }>; splits: Record<string, any> }) {
  const entries = rows
    .map((row) => ({ ...row, breakdown: splits[row.split]?.modes?.[row.mode]?.breakdown }))
    .filter((row) => row.breakdown)
  if (entries.length === 0) return null
  return (
    <details className="mt-3 text-xs">
      <summary className="cursor-pointer">Per-domain, per-generator and per-length ROC-AUC</summary>
      <div className="mt-2 space-y-2">
        {entries.map((row) => (
          <div key={`${row.split}-${row.mode}`}>
            <p className="panel-title">
              {row.mode} · {row.split === 'test_ood' ? 'held-out' : 'in-domain'}
            </p>
            {(['domain', 'generator', 'length_bucket'] as const).map((key) => {
              const block = row.breakdown?.[key]
              if (!block || Object.keys(block).length === 0) return null
              return (
                <p key={key} className="muted">
                  {key.replace('_', ' ')}:{' '}
                  {Object.entries(block)
                    .map(([name, entry]: [string, any]) =>
                      `${name} ${entry.roc_auc !== undefined ? entry.roc_auc.toFixed(3) : '—'} (n=${entry.n})`,
                    )
                    .join(' · ')}
                </p>
              )
            })}
          </div>
        ))}
      </div>
    </details>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="card-2 p-2">
      <p className="panel-title">{label}</p>
      <p className="font-mono text-base">{value}</p>
    </div>
  )
}

function ResultsBlock({ title, split, tone }: { title: string; split: any; tone: 'good' | 'warn' }) {
  const modes = Object.entries(split.modes ?? {}) as Array<[string, any]>
  const components = Object.entries(split.components ?? {}) as Array<[string, any]>
  const rows = [
    ...components.map(([name, metrics]) => ({ name, kind: 'component', metrics })),
    ...modes.map(([name, metrics]) => ({ name, kind: 'mode', metrics })),
  ]
  return (
    <Card title={<span className="flex items-center gap-2">{title} <Badge tone={tone}>{split.label}</Badge></span>}>
      {rows.length === 0 ? (
        <p className="muted text-sm">No scorer produced numbers on this split.</p>
      ) : (
        <div className="overflow-auto">
          <table className="w-full text-xs">
            <thead>
              <tr>
                <th className="panel-title p-1.5 text-left">scorer</th>
                <th className="panel-title p-1.5 text-left">type</th>
                {COMPONENT_KEYS.map((key) => (
                  <th key={key} className="panel-title p-1.5 text-left">{key.replace(/_/g, ' ')}</th>
                ))}
                <th className="panel-title p-1.5 text-left">human FPR@t_ai</th>
                <th className="panel-title p-1.5 text-left">AI miss@t_human</th>
                <th className="panel-title p-1.5 text-left">abstain</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(({ name, kind, metrics }) => (
                <tr key={`${kind}-${name}`} className="border-t" style={{ borderColor: 'var(--border)' }}>
                  <td className="p-1.5 font-medium">{name}</td>
                  <td className="p-1.5 capitalize">{kind}</td>
                  {COMPONENT_KEYS.map((key) => (
                    <td key={key} className="p-1.5 font-mono">
                      {formatMetric(metrics[key])}
                    </td>
                  ))}
                  <td className="p-1.5 font-mono">{formatMetric(metrics.human_fpr_at_t_ai)}</td>
                  <td className="p-1.5 font-mono">{formatMetric(metrics.ai_miss_at_t_human)}</td>
                  <td className="p-1.5 font-mono">{formatMetric(metrics.abstention_rate)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            {rows
              .filter((row) => row.metrics.roc_auc_ci)
              .slice(0, 2)
              .map((row) => (
                <CIChart key={row.name} label={row.name} value={row.metrics.roc_auc} ci={row.metrics.roc_auc_ci} />
              ))}
          </div>
        </div>
      )}
    </Card>
  )
}

function formatMetric(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'number') return value.toFixed(3)
  return String(value)
}

function CIChart({ label, value, ci }: { label: string; value: number; ci: [number | null, number | null] }) {
  const data = [
    { name: 'lower', value: ci[0] ?? value },
    { name: 'estimate', value },
    { name: 'upper', value: ci[1] ?? value },
  ]
  return (
    <div className="card-2 p-2">
      <p className="panel-title mb-1">{label} ROC-AUC with 95% CI</p>
      <div className="h-20">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <XAxis dataKey="name" tick={{ fontSize: 9 }} />
            <YAxis domain={[0.5, 1]} tick={{ fontSize: 9 }} />
            <Tooltip />
            <Line type="monotone" dataKey="value" stroke="var(--accent-2)" strokeWidth={2} dot />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
