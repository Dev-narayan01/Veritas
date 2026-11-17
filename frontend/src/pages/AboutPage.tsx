/** About: methodology diagram (SVG), datasets, models, corpus size, what the online
 * check sends, limitations, ethics and credits. Statements are limited to what the
 * repository actually does. */
import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import type { Health } from '../types'
import { Badge, Callout, Card } from '../components/ui'

export function AboutPage() {
  const [health, setHealth] = useState<Health | null>(null)
  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null))
  }, [])

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">About Veritas</h1>
      <Card title="Methodology">
        <PipelineDiagram />
        <p className="muted mt-2 text-sm">
          Documents are parsed, integrity-checked (invisible characters, homoglyphs, hidden PDF text),
          normalised with an offset map back to the original text, and then analysed three ways: a
          calibrated AI-authorship stacker, a span-level plagiarism engine, and (on request) an
          opt-in online source check. Sentence heatmaps, token views and mixed-authorship segmentation
          come from the same cleaned text, so every highlight lines up with the file you uploaded.
        </p>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        <Card title="Detector components">
          <ul className="space-y-1 text-sm">
            {Object.entries(health?.models_loaded ?? {}).map(([name, loaded]) => (
              <li key={name} className="flex items-center justify-between">
                <span className="font-mono text-xs">{name}</span>
                <Badge tone={loaded ? 'good' : 'warn'}>{loaded ? 'loaded' : 'unavailable'}</Badge>
              </li>
            ))}
          </ul>
          <p className="muted mt-2 text-xs">
            A mode is offered only when every component it declares has loaded. Modes that cannot run
            are shown disabled with the server's own reason.
          </p>
        </Card>

        <Card title="Datasets and corpus">
          <ul className="space-y-1 text-sm">
            <li>Training data: {health?.data_source ?? 'unknown'}</li>
            <li>Plagiarism corpus on this server: {health?.corpus_size ?? 0} documents</li>
            <li>OCR: {health?.ocr ? 'available' : 'not available'}</li>
            <li>Local LLM explanations: {health?.llm ? 'available' : 'not available'}</li>
            <li>GPU: {health?.gpu ? 'available' : 'not available'}</li>
          </ul>
          <p className="muted mt-2 text-xs">
            The corpus is small compared with commercial services and is described by its real size
            everywhere it is used. Coverage is limited, and a low similarity score does not prove
            originality.
          </p>
        </Card>

        <Card title="What the online check sends">
          <p className="text-sm">
            Only when you tick “Online source check”: 6–8 of the most distinctive sentences (at most 200
            characters each) are sent as search queries to Wikipedia, arXiv and OpenAlex. Whole
            documents are never uploaded. The results page lists exactly which sentences were sent.
          </p>
        </Card>

        <Card title="Ethics and limitations">
          <ul className="muted list-disc space-y-1 pl-4 text-sm">
            <li>Detection is probabilistic; never base an accusation on it alone.</li>
            <li>Formal and non-native writing is where false positives concentrate; the bias check is reported with confidence intervals rather than hidden.</li>
            <li>The plagiarism corpus is limited and matches are reported as a share of the document, not as a verdict about intent.</li>
            <li>Robustness to paraphrase and deliberate evasion is bounded; the evaluation page reports what was measured, including perturbations that did nothing.</li>
            <li>No document text leaves the server unless the online check is enabled.</li>
          </ul>
        </Card>
      </div>

      <Callout tone="info" title="Credits">
        Built as an academic project. It uses published research methods — DetectGPT / Fast-DetectGPT,
        Binoculars, HC3, MinHash/LSH, winnowing, Sentence-BERT, FAISS — and compares against one
        third-party detector as an external baseline, clearly labelled. It does not claim parity with
        commercial services.
      </Callout>
    </div>
  )
}

function PipelineDiagram() {
  const steps = ['Ingest', 'Parse', 'Integrity', 'Normalise', 'AI detector', 'Plagiarism', 'Report']
  return (
    <svg viewBox="0 0 700 90" role="img" aria-label="Pipeline: ingest, parse, integrity, normalise, AI detector, plagiarism, report" className="w-full">
      <defs>
        <linearGradient id="pipe" x1="0" x2="1">
          <stop offset="0%" stopColor="#7C5CFF" />
          <stop offset="100%" stopColor="#22D3EE" />
        </linearGradient>
      </defs>
      <line x1="40" y1="45" x2="660" y2="45" stroke="url(#pipe)" strokeWidth="2" />
      {steps.map((step, index) => {
        const x = 40 + (index * 620) / (steps.length - 1)
        return (
          <g key={step}>
            <circle cx={x} cy={45} r={9} fill="#0E1424" stroke="url(#pipe)" strokeWidth="2" />
            <text x={x} y={78} textAnchor="middle" fontSize="11" fill="var(--muted)">
              {step}
            </text>
          </g>
        )
      })}
    </svg>
  )
}
