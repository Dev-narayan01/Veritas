import { afterEach, describe, expect, it, vi } from 'vitest'
import { resolveApiBase } from '../../src/lib/apiBase'
import type { Health } from '../../src/types'

const health = {
  status: 'ok',
  version: '1.0.0',
  modes: [{ mode: 'lite', available: true, reason: null, components: ['classical'] }],
} as unknown as Health

afterEach(() => {
  vi.unstubAllEnvs()
})

describe('resolveApiBase', () => {
  it('uses the same-origin API when the full server is not configured', async () => {
    vi.stubEnv('VITE_API_BASE_URL', '')
    const state = await resolveApiBase({ probe: async (base) => (base === '' ? health : Promise.reject(new Error('nope'))) })
    expect(state.base).toBe('')
    expect(state.mode).toBe('lite')
    expect(state.health?.status).toBe('ok')
  })

  it('stays usable and honest when nothing answers', async () => {
    vi.stubEnv('VITE_API_BASE_URL', '')
    const state = await resolveApiBase({ probe: async () => Promise.reject(new Error('offline')) })
    expect(state.mode).toBe('offline')
    expect(state.health).toBeNull()
    expect(state.message).toBeTruthy()
  })

  it('prefers the configured Space and reports wake-up progress', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'https://example.invalid/')
    const messages: string[] = []
    let calls = 0
    const state = await resolveApiBase({
      onWaiting: (message) => messages.push(message),
      probe: async (base) => {
        calls += 1
        if (calls < 2) throw new Error('starting')
        return base === 'https://example.invalid' ? health : Promise.reject(new Error('wrong base'))
      },
    })
    expect(state.base).toBe('https://example.invalid')
    expect(state.mode).toBe('full')
    expect(messages.length).toBeGreaterThan(0)
  }, 20_000)
})
