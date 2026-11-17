/** History: searchable table of past analyses with open / export / delete. */
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { toast } from 'sonner'
import { api } from '../lib/api'
import { plainPercent, relativeDate, truncate } from '../lib/format'
import type { AnalysisSummary } from '../types'
import { Badge, Card, EmptyState, Skeleton } from '../components/ui'

export function HistoryPage() {
  const [rows, setRows] = useState<AnalysisSummary[] | null>(null)
  const [query, setQuery] = useState('')

  const refresh = () => {
    api
      .history()
      .then(setRows)
      .catch((error) => {
        setRows([])
        toast.error(error instanceof Error ? error.message : 'Could not load history')
      })
  }

  useEffect(refresh, [])

  const filtered = useMemo(() => {
    if (!rows) return []
    const needle = query.trim().toLowerCase()
    if (!needle) return rows
    return rows.filter((row) => row.filenames.join(' ').toLowerCase().includes(needle))
  }, [rows, query])

  if (rows === null) return <Skeleton lines={6} />

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1 className="text-2xl font-semibold">History</h1>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search by filename"
          aria-label="Search history"
          className="rounded-full border bg-transparent px-3 py-1.5 text-sm"
          style={{ borderColor: 'var(--border)' }}
        />
      </div>

      {filtered.length === 0 ? (
        <EmptyState title="No analyses yet" action={<Link className="btn btn-primary" to="/">Analyse a document</Link>}>
          Results appear here after an analysis completes.
        </EmptyState>
      ) : (
        <Card>
          <div className="overflow-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left">
                  {['Date', 'Documents', 'Mode', 'Verdict', 'Similarity', ''].map((heading) => (
                    <th key={heading} className="panel-title p-2">{heading}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((row) => (
                  <tr key={row.id} className="border-t" style={{ borderColor: 'var(--border)' }}>
                    <td className="p-2 whitespace-nowrap">{relativeDate(row.created_at)}</td>
                    <td className="p-2">{truncate(row.filenames.join(', '), 60)}</td>
                    <td className="p-2">{row.mode ?? '—'}</td>
                    <td className="p-2">
                      <div className="flex flex-wrap gap-1">
                        {row.verdicts.length === 0 && <span className="muted">—</span>}
                        {row.verdicts.map((verdict, index) => (
                          <Badge key={index} tone={verdict === 'likely_ai' ? 'bad' : verdict === 'likely_human' ? 'good' : 'warn'}>
                            {verdict.replace('_', ' ')}
                          </Badge>
                        ))}
                      </div>
                    </td>
                    <td className="p-2 font-mono">{row.similarity === null ? '—' : plainPercent(row.similarity)}</td>
                    <td className="p-2">
                      <div className="flex justify-end gap-1">
                        <Link className="btn !px-2 !py-1 text-xs" to={`/results/${row.id}`}>Open</Link>
                        <button
                          className="btn !px-2 !py-1 text-xs"
                          onClick={async () => {
                            try {
                              await api.deleteAnalysis(row.id)
                              toast.success('Deleted')
                              refresh()
                            } catch (error) {
                              toast.error(error instanceof Error ? error.message : 'Delete failed')
                            }
                          }}
                        >
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}
