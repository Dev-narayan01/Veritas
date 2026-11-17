/** Zustand store for cross-page UI state (analysis options, last result id, health). */
import { create } from 'zustand'
import type { Health, Mode } from './types'

export interface UiState {
  health: Health | null
  apiMode: 'full' | 'lite' | 'offline'
  apiMessage: string | null
  mode: Mode
  ai: boolean
  plagiarism: boolean
  batch: boolean
  explain: boolean
  online: boolean
  retain: boolean
  excludeQuotes: boolean
  excludeBibliography: boolean
  minMatchWords: number
  hideBelowPercent: number
  lastAnalysisId: string | null
  setHealth: (health: Health | null, mode: 'full' | 'lite' | 'offline', message: string | null) => void
  setMode: (mode: Mode) => void
  toggle: (key: 'ai' | 'plagiarism' | 'batch' | 'explain' | 'online' | 'retain') => void
  setExcludeQuotes: (value: boolean) => void
  setExcludeBibliography: (value: boolean) => void
  setMinMatchWords: (value: number) => void
  setHideBelowPercent: (value: number) => void
  setLastAnalysisId: (id: string | null) => void
}

export const useUi = create<UiState>((set) => ({
  health: null,
  apiMode: 'offline',
  apiMessage: null,
  mode: 'standard',
  ai: true,
  plagiarism: true,
  batch: false,
  explain: true,
  online: false,
  retain: true,
  excludeQuotes: true,
  excludeBibliography: true,
  minMatchWords: 8,
  hideBelowPercent: 0.5,
  lastAnalysisId: null,
  setHealth: (health, mode, message) =>
    set({
      health,
      apiMode: mode,
      apiMessage: message,
      mode: health?.default_mode ?? 'standard',
    }),
  setMode: (mode) => set({ mode }),
  toggle: (key) => set((state) => ({ [key]: !state[key] }) as Partial<UiState>),
  setExcludeQuotes: (excludeQuotes) => set({ excludeQuotes }),
  setExcludeBibliography: (excludeBibliography) => set({ excludeBibliography }),
  setMinMatchWords: (minMatchWords) => set({ minMatchWords }),
  setHideBelowPercent: (hideBelowPercent) => set({ hideBelowPercent }),
  setLastAnalysisId: (lastAnalysisId) => set({ lastAnalysisId }),
}))
