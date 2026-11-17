import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { Viewer } from '../../src/components/Viewer'
import type { AIResult } from '../../src/types'

const ai = {
  mode_used: 'lite',
  components_run: ['classical'],
  components_unavailable: [],
  probability: 0.8,
  ci_low: 0.7,
  ci_high: 0.9,
  verdict: 'likely_ai',
  thresholds: { t_human: 0.25, t_ai: 0.6, band_collapsed: false },
  per_sentence: [0.9, 0.2],
  sentences: [
    { index: 0, start: 0, end: 17, text: 'Alpha beta gamma.', ai_prob: 0.9, flagged_tokens: [] },
    { index: 1, start: 18, end: 28, text: 'Delta ipsum.', ai_prob: 0.2, flagged_tokens: [] },
  ],
  tokens: [],
  segments: [],
  ai_fraction: 0.5,
  explanation: null,
  low_confidence: false,
  warnings: [],
} as unknown as AIResult

describe('Viewer', () => {
  it('renders the document text and can switch to the clean view', () => {
    render(
      <Viewer text="Alpha beta gamma. Delta ipsum." ai={ai} plagiarism={null} visibleMatches={[]} onSentence={vi.fn()} onMatch={vi.fn()} />,
    )
    const viewer = screen.getByTestId('document-viewer')
    expect(viewer.textContent?.replace(/\s+/g, ' ')).toContain('Alpha beta gamma.')
  })

  it('opens the sentence when a heatmap span is clicked', async () => {
    const onSentence = vi.fn()
    render(
      <Viewer text="Alpha beta gamma. Delta ipsum." ai={ai} plagiarism={null} visibleMatches={[]} onSentence={onSentence} onMatch={vi.fn()} />,
    )
    const button = await screen.findByTitle(/Sentence 1/)
    button.click()
    expect(onSentence).toHaveBeenCalledWith(0)
  })

  it('disables the plagiarism tab when no plagiarism result exists', () => {
    render(
      <Viewer text="Alpha beta gamma." ai={ai} plagiarism={null} visibleMatches={[]} onSentence={vi.fn()} onMatch={vi.fn()} />,
    )
    expect(screen.getByRole('tab', { name: 'Plagiarism' })).toBeDisabled()
  })
})
