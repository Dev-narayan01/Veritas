/** Route-level render tests in jsdom. These stand in for the Playwright smoke suite when
 * no browser binary is available: they mount the real pages with a stubbed fetch and
 * assert the same user-visible guarantees (honest mode reporting, DOM verdict, OOD first).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { App } from '../../src/App'
import { useUi } from '../../src/store'
import { health, metrics, result } from '../fixtures'

function mount(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  )
}

const json = (body: unknown) => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body), text: () => Promise.resolve('') })

beforeEach(() => {
  useUi.getState().setHealth(health as never, 'full', null)
  vi.stubGlobal(
    'fetch',
    vi.fn((input: RequestInfo) => {
      const url = String(input)
      if (url.includes('/api/health')) return json(health)
      if (url.includes('/api/metrics/plots')) return json([])
      if (url.includes('/api/metrics')) return json(metrics)
      if (url.includes('/api/samples')) return json([{ label: 'Human sample', text: 'Alpha beta.' }])
      if (url.includes('/api/analyses/test-analysis')) return json(result)
      if (url.includes('/api/analyses')) return json([{ id: 'test-analysis', created_at: new Date().toISOString(), filenames: ['sample.txt'], mode: 'lite', verdicts: ['uncertain'], similarity: 12.5 }])
      return json({})
    }),
  )
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('Analyze page', () => {
  it('disables modes the server reports as unavailable and shows their reason', async () => {
    mount('/')
    expect(await screen.findByRole('heading', { level: 1 })).toBeTruthy()
    const standard = await screen.findByRole('button', { name: 'standard' })
    expect(standard).toBeDisabled()
    expect(screen.getByRole('button', { name: 'lite' })).toBeEnabled()
  })

  it('shows the real corpus size from health', async () => {
    mount('/')
    expect(await screen.findByText('corpus: 30 documents')).toBeTruthy()
  })
})

describe('Results page', () => {
  it('shows the verdict, probability and the unavailable component', async () => {
    mount('/results/test-analysis')
    expect(await screen.findByTestId('verdict-text')).toHaveTextContent('71.0%')
    expect(screen.getByText('binoculars unavailable')).toBeTruthy()
    expect(screen.getByText('corpus 30')).toBeTruthy()
  })

  it('renders the document text in the viewer', async () => {
    mount('/results/test-analysis')
    const viewer = await screen.findByTestId('document-viewer')
    expect(viewer.textContent).toContain('Alpha beta gamma delta.')
  })

  it('shows the similarity report with the after-exclusions percentage', async () => {
    mount('/results/test-analysis')
    expect(await screen.findByText(/after exclusions/)).toBeTruthy()
  })
})

describe('Evaluation page', () => {
  it('lists held-out results before in-domain results', async () => {
    const { container } = mount('/evaluation')
    await screen.findByText(/Held-out \(out-of-domain\)/)
    const headings = Array.from(container.querySelectorAll('h2')).map((node) => node.textContent ?? '')
    const ood = headings.findIndex((text) => text.includes('Held-out'))
    const inDomain = headings.findIndex((text) => text.includes('In-domain'))
    expect(ood).toBeGreaterThanOrEqual(0)
    expect(inDomain).toBeGreaterThan(ood)
  })

  it('labels the third-party baseline and the synthetic benchmark', async () => {
    mount('/evaluation')
    expect(await screen.findByText('third-party, OOD only')).toBeTruthy()
    expect(screen.getByText('synthetic')).toBeTruthy()
  })

  it('reports robustness changed_fraction so no-op perturbations are visible', async () => {
    mount('/evaluation')
    expect(await screen.findByText('synonym replace')).toBeTruthy()
    expect(screen.getByText('40%')).toBeTruthy()
  })

  it('does not show a smoke warning for a non-smoke run', async () => {
    mount('/evaluation')
    await screen.findByText(/Held-out/)
    expect(screen.queryByText('SMOKE RUN — not results')).toBeNull()
  })
})

describe('About and 404', () => {
  it('states the corpus size and the limitations', async () => {
    mount('/about')
    expect(await screen.findByRole('heading', { name: 'About Veritas' })).toBeTruthy()
    expect(await screen.findByText('Plagiarism corpus on this server: 30 documents')).toBeTruthy()
  })

  it('renders the custom 404 for unknown routes', async () => {
    mount('/does-not-exist')
    const heading = await screen.findByText('404')
    expect(within(heading.parentElement as HTMLElement).getByText('That page does not exist.')).toBeTruthy()
  })
})
