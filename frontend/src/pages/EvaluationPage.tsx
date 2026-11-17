/** Evaluation page. Order matters and is enforced here: badges, then HELD-OUT (OOD)
 * results, then in-domain labelled "optimistic", then ablation, cross-generator,
 * third-party baseline (OOD only), robustness (with changed_fraction), bias (with CIs),
 * the plagiarism benchmark (labelled synthetic) and the embedding projection.
 * Every value is read from /api/metrics, which serves the JSON written by ml/ scripts. */
import { lazy, Suspense, useEffect, useState } from 'react'
import { AlertTriangle } from 'lucide-react'
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../lib/api'
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

      {data.plots.length > 0 && data.plots_base && (
        <Card title="Plots from the evaluation script">
          <div className="grid gap-3 sm:grid-cols-2">
            {data.plots.map((plot) => (
              <figure key={plot}>
                <img
                  src={`${apiBase()}${data.plots_base}/${plot}`}
                  alt={`Evaluation plot ${plot.replace(/_/g, ' ').replace('.png', '')}`}
                  className="w-full rounded-lg border"
                  style={{ borderColor: 'var(--border)' }}
                  loading="lazy"
                />
                <figcaption className="muted mt-1 text-xs">{plot.replace(/_/g, ' ').replace('.png', '')}</figcaption>
              </figure>
            ))}
          </div>
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

function apiBase(): string {
  return (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ?? ''
}
