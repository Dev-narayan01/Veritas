import type { Config } from 'tailwindcss'

/**
 * Design tokens from the specification. Colours are duplicated in CSS variables
 * (styles/tokens.css) so the 3D scenes can read the same palette.
 */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#070A12',
        surface: '#0E1424',
        'surface-2': '#141C31',
        border: '#1F2A44',
        ink: '#E8ECF6',
        muted: '#8B96AD',
        accent: '#7C5CFF',
        cyan: '#22D3EE',
        human: '#34D399',
        uncertain: '#FBBF24',
        ai: '#F87171',
      },
      fontFamily: {
        sans: ['InterVariable', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      borderRadius: { card: '16px' },
      transitionDuration: { 150: '150ms', 250: '250ms' },
    },
  },
  plugins: [],
} satisfies Config
