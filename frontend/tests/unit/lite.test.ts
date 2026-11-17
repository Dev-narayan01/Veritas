/** The Vercel Lite path: the client must not silently pretend it did more than it can. */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, clearCachedResult, getApiState, initApi, readCachedResult, setApiState } from '../../src/lib/api'
import { health as fullHealth } from '../fixtures'

const liteHealth = {
  ...fullHealth,
  status: 'ok',
  deployment: 'lite',
  corpus_size: 0,
  models_loaded: { classical: true, ppl_gpt2: false },
  modes: [
    { mode: 'lite', available: true, reason: null, components: ['classical'] },
    { mode: 'fast', available: false, reason: 'Lite deployment', components: ['classical', 'ppl_gpt2'] },
    { mode: 'standard', available: false, reason: 'Lite deployment', components: [] },
    { mode: 'deep', available: false, reason: 'Lite deployment', components: [] },
  ],
  lite_limitations: ['no plagiarism', 'no uploads'],
}

const liteResult = {
  id: 'lite',
  created_at: '2026-11-17T10:00:00Z',
  status: 'done',
  documents: [],
  exit_notes: ['no plagiarism'],
  disclaimer: 'probabilistic',
}

const options = {
  mode: 'lite' as const,
  ai: true,
  plagiarism: false,
  batch: false,
  explain: true,
  online: false,
  exclusions: { quotes: true, bibliography: true, min_words: 8, hide_below_percent: 0 },
  retain: false,
}

beforeEach(async () => {
  sessionStorage.clear()
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo) => {
    const url = String(input)
    if (url.includes('/api/health')) return { ok: true, status: 200, json: async () => liteHealth, text: async () => '' }
    if (url.includes('/api/analyze')) return { ok: true, status: 200, json: async () => liteResult, text: async () => '' }
    return { ok: false, status: 404, json: async () => ({}), text: async () => 'not found' }
  }))
  const state = await initApi()
  setApiState(state)
})

afterEach(() => {
  vi.unstubAllGlobals()
  clearCachedResult('lite-latest')
})

describe('Lite deployment', () => {
  it('is detected as lite rather than full', () => {
    expect(getApiState().mode).toBe('lite')
    expect(api.isLite()).toBe(true)
  })

  it('refuses uploads with an explanation instead of analysing nothing', async () => {
    const file = new File(['hello'], 'essay.txt', { type: 'text/plain' })
    await expect(api.createAnalysis({ files: [file], options })).rejects.toThrow(/pasted text/i)
  })

  it('refuses URLs for the same reason', async () => {
    await expect(api.createAnalysis({ files: [], url: 'https://example.com', options })).rejects.toThrow(/pasted text/i)
  })

  it('posts pasted text to /api/analyze and caches the result', async () => {
    const { id } = await api.createAnalysis({ files: [], text: 'Some essay text.', options })
    expect(id).toBe('lite-latest')
    expect(readCachedResult('lite-latest')?.status).toBe('done')
    const call = (fetch as unknown as { mock: { calls: unknown[][] } }).mock.calls.find((args) => String(args[0]).includes('/api/analyze'))
    expect(call).toBeTruthy()
    const body = JSON.parse(String((call?.[1] as RequestInit).body))
    expect(body).toEqual({ text: 'Some essay text.', mode: 'lite' })
  })

  it('serves the cached result back and keeps no other history', async () => {
    await api.createAnalysis({ files: [], text: 'Some essay text.', options })
    expect((await api.analysis('lite-latest')).status).toBe('done')
    await expect(api.analysis('some-other-id')).rejects.toThrow(/no history/i)
    const history = await api.history()
    expect(history).toHaveLength(1)
    expect(history[0].similarity).toBeNull()
  })

  it('reports progress instantly because the function answers in one request', async () => {
    const onDone = vi.fn()
    const onEvent = vi.fn()
    await new Promise<void>((resolve) => {
      api.watchProgress('lite-latest', onEvent, () => {
        onDone()
        resolve()
      }, () => resolve())
    })
    expect(onEvent).toHaveBeenCalledWith(expect.objectContaining({ stage: 'done', progress: 1 }))
    expect(onDone).toHaveBeenCalled()
  })
})
