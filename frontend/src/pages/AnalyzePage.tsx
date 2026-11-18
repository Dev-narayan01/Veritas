/** Analyze page: drag-and-drop or paste or URL, options driven by /api/health, real
 * SSE progress, and two "try a sample" buttons. Unavailable modes are disabled with the
 * server's own reason; nothing on this page guesses capability. */
import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { FileText, Link2, Trash2, Type, UploadCloud } from 'lucide-react'
import { toast } from 'sonner'
import { api, reasonForMode } from '../lib/api'
import { modeIsAvailable } from '../lib/apiBase'
import { bytes, stageLabel } from '../lib/format'
import { useUi } from '../store'
import { useGraphicsPreference, useGraphicsTier } from '../hooks/useGraphicsTier'
import { Badge, Callout, Card, Toggle } from '../components/ui'

const HeroField = lazy(() => import('../components/scenes/HeroField').then((m) => ({ default: m.HeroField })))
const Scanner = lazy(() => import('../components/scenes/Scanner').then((m) => ({ default: m.Scanner })))

type Tab = 'upload' | 'paste' | 'url'
const MODES = ['lite', 'fast', 'standard', 'deep'] as const

export function AnalyzePage() {
  const navigate = useNavigate()
  const ui = useUi()
  const [preference] = useGraphicsPreference()
  const { tier } = useGraphicsTier(preference)
  // On the Lite deployment upload and URL cannot work, so opening on a drop zone that
  // only errors when clicked is a dead end. Paste is the only real input there.
  const [tab, setTab] = useState<Tab>(api.isLite() ? 'paste' : 'upload')
  const [files, setFiles] = useState<File[]>([])
  const [text, setText] = useState('')
  const [url, setUrl] = useState('')
  const [dragging, setDragging] = useState(false)
  const [progress, setProgress] = useState<{ value: number; stage: string; message: string } | null>(null)
  const [samples, setSamples] = useState<Array<{ label: string; text: string }>>([])
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    api.samples().then(setSamples).catch(() => setSamples([]))
  }, [])

  const addFiles = useCallback((incoming: FileList | File[]) => {
    const list = Array.from(incoming)
    const accepted = list.filter((file) => file.size <= 20 * 1024 * 1024)
    if (accepted.length !== list.length) toast.error('Some files exceed the 20 MB limit and were skipped')
    setFiles((current) => [...current, ...accepted].slice(0, 50))
  }, [])

  const submit = async () => {
    if (!files.length && !text.trim() && !url.trim()) {
      toast.error('Add a file, paste some text or enter a URL first')
      return
    }
    setProgress({ value: 0.02, stage: 'queued', message: 'submitting' })
    try {
      const { id } = await api.createAnalysis({
        files,
        text: text.trim() || undefined,
        url: url.trim() || undefined,
        options: {
          mode: ui.mode,
          ai: ui.ai,
          plagiarism: ui.plagiarism,
          batch: ui.batch,
          explain: ui.explain,
          online: ui.online,
          exclusions: {
            quotes: ui.excludeQuotes,
            bibliography: ui.excludeBibliography,
            min_words: ui.minMatchWords,
            hide_below_percent: ui.hideBelowPercent,
          },
          retain: ui.retain,
        },
      })
      ui.setLastAnalysisId(id)
      api.watchProgress(
        id,
        (event) => setProgress({ value: event.progress, stage: event.stage, message: event.message }),
        () => navigate(`/results/${id}`),
        (message) => {
          setProgress(null)
          toast.error(message)
        },
      )
    } catch (error) {
      setProgress(null)
      toast.error(error instanceof Error ? error.message : 'Could not start the analysis')
    }
  }

  const lite = api.isLite()
  // Lite offers lite and, when its bundle is deployed, fast. Everything else needs the
  // full server, so it stays disabled with the server's own reason rather than a generic
  // label -- and fast must not be offered when it would only silently degrade.
  const modeReason = (mode: string) =>
    lite ? (modeIsAvailable(ui.health, mode) ? reasonForMode(ui.health, mode) : 'Needs the full analysis server') : reasonForMode(ui.health, mode)

  return (
    <div className="space-y-6">
      <section className="relative overflow-hidden rounded-card border p-6 sm:p-10" style={{ borderColor: 'var(--border)', backgroundColor: 'var(--surface)' }}>
        {tier !== 'off' && (
          <Suspense fallback={null}>
            <HeroField tier={tier} dragging={dragging} />
          </Suspense>
        )}
        <div className="relative">
          <h1 className="max-w-2xl text-3xl font-semibold leading-tight sm:text-4xl">
            Transparent AI-text and plagiarism analysis — with the evidence, not just a score.
          </h1>
          <p className="muted mt-3 max-w-2xl text-sm">
            Veritas combines calibrated statistical detectors, a sentence-level heatmap, mixed-authorship
            segmentation and a span-level similarity report. Every verdict shows the mode and components
            that actually ran, and the thresholds in force.
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            <Badge tone={ui.health?.artifact_source === 'kaggle' ? 'good' : 'warn'}>
              {ui.health?.artifact_source === 'kaggle'
                ? 'GPU-trained artifacts'
                : ui.health?.artifact_source === 'cpu'
                  ? 'CPU-trained baseline artifacts'
                  : 'No artifacts loaded'}
            </Badge>
            <Badge tone={ui.health?.data_source === 'hc3' ? 'good' : 'warn'}>
              data: {ui.health?.data_source ?? 'unknown'}
            </Badge>
            {lite ? (
              <Badge tone="warn">Lite deployment · pasted text only</Badge>
            ) : (
              <Badge>corpus: {ui.health?.corpus_size ?? 0} documents</Badge>
            )}
            {!lite && (ui.health?.corpus_size ?? 0) === 0 && <Badge tone="bad">plagiarism corpus empty</Badge>}
            {!ui.health?.ocr && <Badge tone="warn">OCR unavailable</Badge>}
          </div>
        </div>
      </section>

      {progress ? (
        <Card title="Analysing">
          <Suspense fallback={<p className="muted text-sm">Preparing…</p>}>
            <Scanner tier={tier} progress={progress.value} stage={stageLabel(progress.stage)} message={progress.message} />
          </Suspense>
        </Card>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
          <Card>
            <div role="tablist" aria-label="Input method" className="mb-3 flex gap-1">
              {([
                ['upload', 'Upload', UploadCloud],
                ['paste', 'Paste', Type],
                ['url', 'URL', Link2],
              ] as const).map(([key, label, Icon]) => (
                <button
                  key={key}
                  role="tab"
                  aria-selected={tab === key}
                  className="btn !rounded-lg !px-3 !py-1.5 text-xs"
                  style={tab === key ? { backgroundColor: 'var(--surface-2)', color: 'var(--text)' } : undefined}
                  onClick={() => setTab(key)}
                >
                  <Icon size={14} aria-hidden="true" /> {label}
                </button>
              ))}
            </div>

            {tab === 'upload' && (
              <div>
                <div
                  onDragOver={(event) => {
                    event.preventDefault()
                    setDragging(true)
                  }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={(event) => {
                    event.preventDefault()
                    setDragging(false)
                    addFiles(event.dataTransfer.files)
                  }}
                  onClick={() => {
                    if (lite) {
                      toast.error('This deployment analyses pasted text only — use the Paste tab.')
                      return
                    }
                    inputRef.current?.click()
                  }}
                  className="flex cursor-pointer flex-col items-center gap-2 rounded-card border border-dashed p-8 text-center transition-colors duration-150"
                  style={{ borderColor: dragging ? 'var(--accent)' : 'var(--border)', backgroundColor: dragging ? 'var(--surface-2)' : 'transparent' }}
                >
                  <UploadCloud aria-hidden="true" />
                  <p className="text-sm">Drop files here, or click to choose</p>
                  <p className="muted text-xs">
                    PDF, DOCX, PPTX, TXT/MD, images (OCR), code and ZIP archives · up to 50 files · 20 MB each
                  </p>
                  <input
                    ref={inputRef}
                    type="file"
                    multiple
                    className="sr-only"
                    data-testid="file-input"
                    onChange={(event) => event.target.files && addFiles(event.target.files)}
                  />
                </div>
                {files.length > 0 && (
                  <ul className="mt-3 space-y-1" aria-label="Selected files">
                    {files.map((file, index) => (
                      <li key={`${file.name}-${index}`} className="card-2 flex items-center justify-between p-2 text-sm">
                        <span className="flex items-center gap-2 truncate">
                          <FileText size={14} aria-hidden="true" />
                          <span className="truncate">{file.name}</span>
                          <span className="muted text-xs">{bytes(file.size)}</span>
                        </span>
                        <button
                          className="btn !border-0 !p-1"
                          aria-label={`Remove ${file.name}`}
                          onClick={() => setFiles((current) => current.filter((_, i) => i !== index))}
                        >
                          <Trash2 size={14} aria-hidden="true" />
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            {tab === 'paste' && (
              <label className="block">
                <span className="sr-only">Text to analyse</span>
                <textarea
                  value={text}
                  onChange={(event) => setText(event.target.value)}
                  rows={12}
                  placeholder="Paste the text you want to analyse…"
                  className="w-full rounded-card border bg-transparent p-3 font-mono text-sm"
                  style={{ borderColor: 'var(--border)' }}
                />
                <span className="muted text-xs">{text.split(/\s+/).filter(Boolean).length} words</span>
              </label>
            )}

            {tab === 'url' && (
              <label className="block">
                <span className="mb-1 block text-sm">Page URL</span>
                <input
                  value={url}
                  onChange={(event) => setUrl(event.target.value)}
                  placeholder="https://example.com/article"
                  className="w-full rounded-card border bg-transparent p-3 text-sm"
                  style={{ borderColor: 'var(--border)' }}
                />
                <span className="muted mt-1 block text-xs">Only the page's main text is fetched; private addresses are refused.</span>
              </label>
            )}

            {samples.length > 0 && (
              <div className="mt-4 flex flex-wrap gap-2">
                {samples.map((sample) => (
                  <button
                    key={sample.label}
                    className="btn text-xs"
                    onClick={() => {
                      setTab('paste')
                      setText(sample.text)
                    }}
                  >
                    {sample.label}
                  </button>
                ))}
              </div>
            )}
          </Card>

          <Card title="Analysis options">
            <div className="space-y-3">
              <div>
                <p className="mb-1 text-sm">Mode</p>
                <div className="grid grid-cols-2 gap-1">
                  {MODES.map((mode) => {
                    const reason = modeReason(mode)
                    return (
                      <button
                        key={mode}
                        className="btn !rounded-lg !px-2 !py-1.5 text-xs capitalize"
                        disabled={Boolean(reason)}
                        title={reason ?? undefined}
                        aria-pressed={ui.mode === mode}
                        style={ui.mode === mode ? { backgroundColor: 'var(--surface-2)', borderColor: 'var(--accent)' } : undefined}
                        onClick={() => ui.setMode(mode)}
                      >
                        {mode}
                      </button>
                    )
                  })}
                </div>
                {modeReason(ui.mode) && <p className="muted mt-1 text-xs">{modeReason(ui.mode)}</p>}
                <p className="muted mt-1 text-xs">
                  {ui.health?.modes.find((m) => m.mode === ui.mode)?.components.join(' + ') ?? '—'}
                </p>
                {ui.mode === 'fast' && (
                  <p className="muted mt-1 text-xs">
                    Strict band: it commits on roughly one text in eight and is about 95% accurate
                    when it does. Most texts come back uncertain, and that is deliberate.
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <Toggle checked={ui.ai} onChange={() => ui.toggle('ai')} label="AI-authorship analysis" />
                <Toggle
                  checked={ui.plagiarism && !lite}
                  disabled={lite}
                  onChange={() => ui.toggle('plagiarism')}
                  label="Plagiarism analysis"
                  description={lite ? 'runs on the full server' : undefined}
                />
                <Toggle
                  checked={ui.batch && !lite}
                  disabled={lite}
                  onChange={() => ui.toggle('batch')}
                  label="Compare documents with each other"
                  description={lite ? 'runs on the full server' : 'for two or more files'}
                />
                <Toggle checked={ui.explain} onChange={() => ui.toggle('explain')} label="Explanation" />
                <Toggle
                  checked={ui.online}
                  onChange={() => ui.toggle('online')}
                  label="Online source check (opt-in)"
                  description="Sends only 6–8 short sentences (≤200 characters each) to Wikipedia, arXiv and OpenAlex."
                />
                <Toggle
                  checked={ui.retain}
                  onChange={() => ui.toggle('retain')}
                  label="Keep the uploaded files"
                  description="When off, files and extracted text are deleted as soon as the analysis ends."
                />
              </div>

              <div className="space-y-2 border-t pt-3" style={{ borderColor: 'var(--border)' }}>
                <p className="panel-title">Exclusions</p>
                <Toggle checked={ui.excludeQuotes} onChange={ui.setExcludeQuotes} label="Exclude quotations" />
                <Toggle checked={ui.excludeBibliography} onChange={ui.setExcludeBibliography} label="Exclude bibliography" />
                <label className="block text-xs">
                  Minimum match length: <span className="font-mono">{ui.minMatchWords}</span> words
                  <input
                    type="range"
                    min={3}
                    max={40}
                    value={ui.minMatchWords}
                    onChange={(event) => ui.setMinMatchWords(Number(event.target.value))}
                    className="w-full accent-[var(--accent)]"
                  />
                </label>
              </div>

              <button className="btn btn-primary w-full !justify-center" onClick={submit} disabled={ui.apiMode === 'offline'}>
                Analyse
              </button>
              {ui.apiMode === 'offline' && <Callout tone="bad">The analysis server is unreachable.</Callout>}
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
