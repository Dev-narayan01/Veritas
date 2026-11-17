/** Types mirrored from `backend/app/schemas.py`. Keeping them here means the client
 * fails to compile when the API contract changes. */

export type Verdict = 'likely_human' | 'uncertain' | 'likely_ai' | 'too_short'
export type Mode = 'lite' | 'fast' | 'standard' | 'deep'
export type MatchType = 'exact' | 'near' | 'paraphrase' | 'online'

export interface Exclusions {
  quotes: boolean
  bibliography: boolean
  min_words: number
  hide_below_percent: number
}

export interface AnalysisOptions {
  mode: Mode | null
  ai: boolean
  plagiarism: boolean
  batch: boolean
  explain: boolean
  online: boolean
  exclusions: Exclusions
  retain: boolean
}

export interface ModeAvailability {
  mode: Mode
  available: boolean
  reason: string | null
  components: string[]
}

export interface Health {
  status: 'ok' | 'degraded' | 'loading'
  version: string
  modes: ModeAvailability[]
  models_loaded: Record<string, boolean>
  corpus_size: number
  ocr: boolean
  llm: boolean
  gpu: boolean
  artifact_source: 'kaggle' | 'cpu' | 'none'
  artifact_notes: string[]
  data_source: string
  default_mode: Mode
  disclaimer: string
  /** Present on the Vercel Lite deployment: what it cannot do, stated by the server. */
  deployment?: 'lite' | 'full'
  lite_limitations?: string[]
}

export interface ComponentScore {
  name: string
  probability: number | null
  raw: Record<string, number>
  available: boolean
  reason: string | null
}

export interface SentenceScore {
  index: number
  start: number
  end: number
  text: string
  ai_prob: number
  flagged_tokens: number[]
}

export interface Segment {
  start: number
  end: number
  label: 'human' | 'ai'
  confidence: number
}

export interface TokenScore {
  token: string
  start: number
  end: number
  surprisal: number
  rank: number | null
  flagged: boolean
}

export interface Explanation {
  source: 'rule_based' | 'local_llm'
  text: string
  factors: Array<Record<string, string | number>>
}

export interface AIResult {
  mode_requested: string
  mode_used: string | null
  components_run: string[]
  components_unavailable: Array<{ component: string; reason: string }>
  probability: number | null
  ci_low: number | null
  ci_high: number | null
  verdict: Verdict
  thresholds: { t_human: number | null; t_ai: number | null }
  band_collapsed: boolean
  word_count: number
  low_confidence: boolean
  pass_count: number
  components: ComponentScore[]
  sentences: SentenceScore[]
  tokens: TokenScore[]
  segments: Segment[]
  ai_fraction: number | null
  explanation: Explanation | null
  warnings: string[]
  latency_ms: number | null
}

export interface PlagMatch {
  query_span: [number, number]
  source_span: [number, number] | null
  query_text: string
  source_text: string
  source_id: string
  source_title: string
  source_url: string | null
  color_index: number
  match_type: MatchType
  score: number
  words: number
  in_quote: boolean
  in_bibliography: boolean
}

export interface SourceBreakdown {
  id: string
  title: string
  url: string | null
  color_index: number
  percent: number
  matches: number
}

export interface PlagiarismResult {
  similarity_percent: number
  similarity_percent_filtered: number
  lexical_percent: number
  paraphrase_percent: number
  online_percent: number
  sources: SourceBreakdown[]
  matches: PlagMatch[]
  corpus_size: number
  online_snippets_sent: string[]
  warnings: string[]
}

export interface IntegrityFlag {
  kind: string
  count: number
  detail: string
  examples: string[]
}

export interface IntegrityResult {
  flags: IntegrityFlag[]
  cleaned_chars: number
  hidden_text_removed: number
}

export interface BatchPair {
  a: string
  b: string
  percent_ab: number
  percent_ba: number
  segments: Array<Record<string, unknown>>
}

export interface BatchResult {
  matrix: number[][]
  labels: string[]
  pairs: BatchPair[]
  nodes: Array<{ id: string; label: string; words: number; verdict: Verdict }>
  edges: Array<{ a: string; b: string; weight: number }>
}

export interface CodeMatch {
  a: string
  b: string
  similarity: number
  lines_a: number[]
  lines_b: number[]
  functions: string[]
}

export interface DocumentResult {
  id: string
  filename: string
  kind: string
  words: number
  pages: number | null
  text: string
  warnings: string[]
  integrity: IntegrityResult
  ai: AIResult | null
  plagiarism: PlagiarismResult | null
}

export interface AnalysisResult {
  id: string
  created_at: string
  status: 'queued' | 'running' | 'done' | 'error'
  options: AnalysisOptions
  documents: DocumentResult[]
  batch: BatchResult | null
  code_matches: CodeMatch[]
  error: string | null
  exit_notes: string[]
  disclaimer: string
  deployment?: 'lite' | 'full'
}

export interface AnalysisSummary {
  id: string
  created_at: string
  status: string
  filenames: string[]
  mode: string | null
  verdicts: string[]
  similarity: number | null
}

export interface ProgressEvent {
  stage: string
  progress: number
  message: string
}

export interface MetricsResponse {
  artifact_source: 'kaggle' | 'cpu' | 'none'
  data_source: string
  manifest: Record<string, unknown> | null
  plots: string[]
  plots_base: string | null
  metrics: Record<string, any> | null
  robustness: Record<string, any> | null
  bias: Record<string, any> | null
  baseline: Record<string, any> | null
  cross_generator: Record<string, any> | null
  plag_benchmark_results: Record<string, any> | null
  plag_thresholds: Record<string, any> | null
  embedding_projection: Record<string, any> | null
  thresholds: Record<string, any> | null
  split_manifest: Record<string, any> | null
}
