/** Scene layout maths, tested without a canvas. */
import { describe, expect, it } from 'vitest'
import { computeLayout } from '../../src/components/scenes/SimilarityGalaxy'
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
