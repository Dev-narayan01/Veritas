/** Canned API responses. They are shaped like the real contract but are obviously fake
 * values, so a screenshot from these tests can never be mistaken for real results. */
export const health = {
  status: 'ok',
  version: '1.0.0-test',
  modes: [
    { mode: 'lite', available: true, reason: null, components: ['classical'] },
    { mode: 'fast', available: false, reason: 'stacker_fast not fitted', components: ['classical', 'ppl_gpt2'] },
    { mode: 'standard', available: false, reason: 'requires torch', components: ['classical', 'ppl_gpt2', 'binoculars'] },
    { mode: 'deep', available: false, reason: 'requires torch', components: ['classical', 'fast_detectgpt'] },
  ],
  models_loaded: { classical: true, ppl_gpt2: true, binoculars: false },
  corpus_size: 30,
  ocr: false,
  llm: false,
  gpu: false,
  artifact_source: 'cpu',
  artifact_notes: ['CPU-trained baseline artifacts'],
  data_source: 'hc3',
  default_mode: 'lite',
  disclaimer: 'Probabilistic; never the sole basis for an accusation.',
}

const TEXT = 'Alpha beta gamma delta. Epsilon zeta eta theta.'

export const result = {
  id: 'test-analysis',
  status: 'done',
  created_at: new Date().toISOString(),
  options: { mode: 'lite' },
  documents: [
    {
      id: 'doc-1',
      filename: 'sample.txt',
      kind: 'txt',
      words: 8,
      parse_warnings: [],
      text: TEXT,
      integrity: { flags: [], invisible_characters: 0 },
      ai: {
        mode_used: 'lite',
        components_run: ['classical'],
        components_unavailable: [{ component: 'binoculars', reason: 'torch unavailable' }],
        probability: 0.71,
        ci_low: 0.61,
        ci_high: 0.8,
        verdict: 'uncertain',
        thresholds: { t_human: 0.25, t_ai: 0.6, band_collapsed: false },
        per_sentence: [0.7, 0.72],
        sentences: [
          { index: 0, start: 0, end: 21, text: 'Alpha beta gamma delta.', ai_prob: 0.7, flagged_tokens: [] },
          { index: 1, start: 22, end: 42, text: 'Epsilon zeta eta theta.', ai_prob: 0.72, flagged_tokens: [] },
        ],
        tokens: [{ token: 'alpha', start: 0, end: 5, surprisal: 1.1, flagged: true }],
        segments: [{ start: 0, end: 42, label: 'ai', score: 0.7 }],
        ai_fraction: 1,
        explanation: { source: 'rule_based', text: 'Low burstiness and high token predictability.' },
        low_confidence: false,
        warnings: ['test fixture'],
      },
      plagiarism: {
        similarity_percent: 12.5,
        similarity_percent_filtered: 12.5,
        lexical_percent: 12.5,
        paraphrase_percent: 0,
        online_percent: 0,
        corpus_size: 30,
        sources: [{ id: 's1', title: 'Bundled source', url: null, color_index: 0, percent: 12.5, matches: 1 }],
        matches: [
          {
            query_span: [0, 21],
            source_span: [0, 21],
            query_text: 'Alpha beta gamma delta.',
            source_text: 'Alpha beta gamma delta.',
            source_id: 's1',
            source_title: 'Bundled source',
            source_url: null,
            color_index: 0,
            match_type: 'exact',
            score: 1,
            words: 4,
            in_quote: false,
            in_bibliography: false,
          },
        ],
        excluded: [],
        online_snippets_sent: [],
        warnings: ['test fixture'],
      },
      code: null,
    },
  ],
  batch: null,
  code_matches: [],
  exit_notes: ['test fixture'],
  disclaimer: 'Probabilistic; never the sole basis for an accusation.',
}

export const metrics = {
  artifact_source: 'cpu',
  data_source: 'hc3',
  metrics: {
    notes: ['test fixture'],
    splits: {
      test_ood: {
        label: 'held-out (out of domain)',
        modes: { lite: { roc_auc: 0.888, pr_auc: 0.9, tpr_at_1pct_fpr: 0.43, f1: 0.8, ece: 0.29, roc_auc_ci: [0.86, 0.91], human_fpr_at_t_ai: 0.2, abstention_rate: 0.1 } },
        components: { classical: { roc_auc: 0.86, roc_auc_ci: [0.83, 0.89], ece: 0.2 } },
      },
      test: {
        label: 'in-domain (optimistic)',
        modes: { lite: { roc_auc: 0.976, tpr_at_1pct_fpr: 0.76, f1: 0.92, ece: 0.05, roc_auc_ci: [0.96, 0.99] } },
        components: {},
      },
    },
    ablation: {
      test_ood: { lite: { components: ['classical'], roc_auc: 0.888, tpr_at_1pct_fpr: 0.43, f1: 0.8 } },
      test: { lite: { components: ['classical'], roc_auc: 0.976, tpr_at_1pct_fpr: 0.76, f1: 0.92 } },
    },
  },
  manifest: { smoke: false, run: 'cpu' },
  robustness: {
    notes: ['test fixture'],
    modes: { lite: { perturbations: { synonym_replace: { recall_before: 0.9, recall_after: 0.7, changed_fraction: 0.4, recall_after_integrity_cleaning: 0.71 } } } },
  },
  bias: {
    proxy_definition: 'ESL proxy: short simple sentences.',
    notes: ['test fixture'],
    modes: { lite: { normal_human_fpr: 0.004, proxy_human_fpr: 0.004, difference: 0, difference_ci95: [-0.012, 0.012], higher_false_positive_rate_on_proxy: false } },
  },
  cross_generator: { generators: {}, notes: ['test fixture'] },
  baseline: { available: false, reason: 'test fixture' },
  plag_thresholds: null,
  plag_benchmark_results: {
    caption: 'synthetic benchmark',
    span_f1_overall: 0.941,
    false_alarm_rate_on_negatives: 0,
    cases: 83,
    span_f1_by_obfuscation: { verbatim: 0.96, light_edit: 0.97 },
    notes: ['test fixture'],
  },
  corpus_stats: { documents: 30 },
  embedding_projection: null,
  plots: [],
  plots_base: '/api/metrics/plots',
  thresholds: null,
  manifest_notes: [],
}

/** Minimal structural type for the Playwright `page` so these fixtures also compile
 * under the jsdom test build, which does not load Playwright's types. */
interface RouteStub {
  route: (
    url: string,
    handler: (route: {
      request(): { method(): string }
      fulfill(options: { json?: unknown; body?: string; status?: number }): unknown
    }) => unknown,
  ) => unknown
}

export async function stubApi(page: RouteStub) {
  await page.route('**/api/health', (route) => route.fulfill({ json: health }))
  await page.route('**/api/samples', (route) =>
    route.fulfill({ json: [{ label: 'Human sample', text: 'Alpha beta gamma delta.' }, { label: 'AI sample', text: 'Epsilon zeta eta theta.' }] }),
  )
  await page.route('**/api/metrics', (route) => route.fulfill({ json: metrics }))
  await page.route('**/api/analyses', (route) => {
    if (route.request().method() === 'POST') return route.fulfill({ json: { id: 'test-analysis' } })
    return route.fulfill({ json: [{ id: 'test-analysis', created_at: new Date().toISOString(), filenames: ['sample.txt'], mode: 'lite', verdicts: ['uncertain'], similarity: 12.5 }] })
  })
  await page.route('**/api/analyses/test-analysis', (route) => route.fulfill({ json: result }))
}
