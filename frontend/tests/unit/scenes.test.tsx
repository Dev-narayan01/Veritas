/** Scene layout maths and fallback rendering, tested without a canvas. */
import { describe, expect, it } from 'vitest'
import { render } from '@testing-library/react'
import { EmbeddingSpace } from '../../src/components/scenes/EmbeddingSpace'
import { SimilarityGalaxy, computeLayout } from '../../src/components/scenes/SimilarityGalaxy'
import { VerdictGauge } from '../../src/components/scenes/VerdictGauge'
import type { BatchResult } from '../../src/types'

const batch = {
  matrix: [
    [100, 42],
    [42, 100],
  ],
  labels: ['a.txt', 'b.txt'],
  pairs: [],
  nodes: [
    { id: 'n1', label: 'a.txt', words: 400, verdict: 'likely_human' },
    { id: 'n2', label: 'b.txt', words: 900, verdict: 'likely_ai' },
    { id: 'n3', label: 'c.txt', words: 50, verdict: 'too_short' },
  ],
  edges: [
    { a: 'a.txt', b: 'b.txt', weight: 42 },
    { a: 'a.txt', b: 'c.txt', weight: 2 },
  ],
} as unknown as BatchResult

describe('computeLayout', () => {
  it('returns one positioned node per document', () => {
    const { nodes } = computeLayout(batch)
    expect(nodes).toHaveLength(3)
    for (const node of nodes) {
      expect(node.position).toHaveLength(3)
      expect(node.position.every((value) => Number.isFinite(value))).toBe(true)
    }
  })

  it('hides weak edges instead of drawing every pair', () => {
    const { links } = computeLayout(batch)
    expect(links).toHaveLength(1)
    expect(links[0].weight).toBe(42)
  })

  it('is deterministic for the same input', () => {
    expect(computeLayout(batch).nodes.map((node) => node.position)).toEqual(
      computeLayout(batch).nodes.map((node) => node.position),
    )
  })

  it('keeps the verdict so the colour can be derived', () => {
    expect(computeLayout(batch).nodes.map((node) => node.verdict)).toEqual([
      'likely_human',
      'likely_ai',
      'too_short',
    ])
  })
})

describe('similarity graph edges', () => {
  it('labels code and text edges differently so a structural match is not read as copied prose', () => {
    const batch: BatchResult = {
      matrix: [
        [100, 100],
        [100, 100],
      ],
      nodes: [
        { id: 'a', label: 'a.py', words: 70, verdict: 'too_short' },
        { id: 'b', label: 'b.py', words: 70, verdict: 'too_short' },
      ],
      labels: ['a.py', 'b.py'],
      pairs: [],
      edges: [{ a: 'a.py', b: 'b.py', weight: 100, kind: 'code' as const }],
    }
    const { links } = computeLayout(batch)
    expect(links).toHaveLength(1)
    expect(links[0].kind).toBe('code')
    expect(links[0].weight).toBe(100)
  })

  it('defaults an unlabelled edge to text', () => {
    const batch: BatchResult = {
      matrix: [
        [100, 42],
        [42, 100],
      ],
      nodes: [
        { id: 'a', label: 'a.txt', words: 200, verdict: 'likely_human' },
        { id: 'b', label: 'b.txt', words: 200, verdict: 'likely_human' },
      ],
      labels: ['a.txt', 'b.txt'],
      pairs: [],
      edges: [{ a: 'a.txt', b: 'b.txt', weight: 42 }],
    }
    expect(computeLayout(batch).links[0].kind).toBe('text')
  })
})

// Render-level checks for the "works identically with WebGL disabled" requirement. The
// Playwright spec asserts this in a real browser; these run here, where no browser binary
// exists, so the fallback contract is at least covered by CI.
describe('scene fallbacks without WebGL', () => {
  it('renders the 2D fallback and no canvas when the tier is off', () => {
    // The shape the artifact really uses: a flat list of [x, y, z] triples plus a parallel
    // label array. A test with a friendlier shape would hide a render crash.
    const projection = {
      method: 'pca',
      caption: 'projection of model features; for visual intuition only',
      points: [
        [0.1, 0.2, 0.3],
        [-0.4, 0.5, -0.1],
      ],
      labels: [1, 0],
      misclassified: [0, 0],
      scores: [0.91, 0.12],
    }
    const { container } = render(
      <EmbeddingSpace tier="off" data={projection as never} />,
    )
    expect(container.querySelector('[data-scene-fallback="true"]')).not.toBeNull()
    expect(container.querySelector('canvas')).toBeNull()
  })

  it('renders the textual galaxy list when the tier is off', () => {
    const batch: BatchResult = {
      matrix: [
        [100, 42],
        [42, 100],
      ],
      labels: ['a.txt', 'b.txt'],
      nodes: [
        { id: 'a', label: 'a.txt', words: 200, verdict: 'likely_human' },
        { id: 'b', label: 'b.txt', words: 220, verdict: 'likely_ai' },
      ],
      pairs: [],
      edges: [{ a: 'a.txt', b: 'b.txt', weight: 42, kind: 'text' }],
    }
    const { container } = render(<SimilarityGalaxy tier="off" batch={batch} onSelect={() => {}} />)
    expect(container.querySelector('canvas')).toBeNull()
    expect(container.textContent).toContain('a.txt')
    expect(container.textContent).toContain('42% text with b.txt')
  })

  it('renders the SVG gauge when the tier is off', () => {
    const { container } = render(
      <VerdictGauge tier="off" probability={0.9} tHuman={0.25} tAi={0.95} color="#F87171" />,
    )
    expect(container.querySelector('canvas')).toBeNull()
    expect(container.querySelector('svg')).not.toBeNull()
  })
})
