/** Dark is the default; the toggle persists a light preference on <html>. */
import { useEffect, useState } from 'react'

const KEY = 'veritas.theme'

export function useTheme(): ['dark' | 'light', () => void] {
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    try {
      return (localStorage.getItem(KEY) as 'dark' | 'light') ?? 'dark'
    } catch {
      return 'dark'
    }
  })

  useEffect(() => {
    const root = document.documentElement
    root.classList.toggle('light', theme === 'light')
    root.classList.toggle('dark', theme === 'dark')
    try {
      localStorage.setItem(KEY, theme)
    } catch {
      /* ignore */
    }
  }, [theme])

  return [theme, () => setTheme((value) => (value === 'dark' ? 'light' : 'dark'))]
}
