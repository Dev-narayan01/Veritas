/** App shell: wordmark, navigation, Effects and theme controls, disclaimer footer.
 * The layout is fully usable with 3D disabled; nothing here depends on WebGL. */
import { Link, NavLink, useLocation } from 'react-router-dom'
import { Activity, BarChart3, History, Info, Moon, ScanLine, Sun } from 'lucide-react'
import { motion } from 'framer-motion'
import { useUi } from '../store'
import { useTheme } from '../hooks/useTheme'
import { useGraphicsPreference, useGraphicsTier } from '../hooks/useGraphicsTier'
import type { GraphicsPreference } from '../lib/graphics'

const NAV = [
  { to: '/', label: 'Analyze', icon: ScanLine },
  { to: '/history', label: 'History', icon: History },
  { to: '/evaluation', label: 'Evaluation', icon: BarChart3 },
  { to: '/about', label: 'About', icon: Info },
]

export function Layout({ children }: { children: React.ReactNode }) {
  const [theme, toggleTheme] = useTheme()
  const [preference, setPreference] = useGraphicsPreference()
  const { tier } = useGraphicsTier(preference)
  const apiMode = useUi((s) => s.apiMode)
  const apiMessage = useUi((s) => s.apiMessage)
  const location = useLocation()

  return (
    <div className="min-h-screen flex flex-col">
      <header className="sticky top-0 z-40 border-b" style={{ borderColor: 'var(--border)', backgroundColor: 'color-mix(in srgb, var(--bg) 88%, transparent)', backdropFilter: 'blur(10px)' }}>
        <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3">
          <Link to="/" className="flex items-center gap-2" aria-label="Veritas home">
            <span
              aria-hidden="true"
              className="grid h-8 w-8 place-items-center rounded-xl text-sm font-bold text-black"
              style={{ backgroundImage: 'linear-gradient(135deg, var(--accent), var(--accent-2))' }}
            >
              V
            </span>
            <span className="text-lg font-semibold tracking-tight">Veritas</span>
          </Link>

          <nav className="ml-2 flex items-center gap-1" aria-label="Main">
            {NAV.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                className="flex items-center gap-1.5 rounded-full px-3 py-1.5 text-sm transition-colors duration-150"
                style={({ isActive }) => ({
                  color: isActive ? 'var(--text)' : 'var(--muted)',
                  backgroundColor: isActive ? 'var(--surface-2)' : 'transparent',
                })}
              >
                <Icon size={15} aria-hidden="true" />
                <span className="hidden sm:inline">{label}</span>
              </NavLink>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-2">
            {apiMode !== 'full' && (
              <span className="chip" style={{ color: 'var(--uncertain)' }} title={apiMessage ?? undefined}>
                <Activity size={12} aria-hidden="true" />
                {apiMode === 'lite' ? 'Lite mode' : 'Server offline'}
              </span>
            )}
            <label className="chip gap-1">
              <span className="muted hidden md:inline">Effects</span>
              <select
                aria-label="Graphics effects"
                className="bg-transparent text-xs outline-none"
                value={preference}
                onChange={(event) => setPreference(event.target.value as GraphicsPreference)}
              >
                <option value="auto">Auto</option>
                <option value="high">High</option>
                <option value="low">Low</option>
                <option value="off">Off</option>
              </select>
              <span className="muted hidden md:inline">({tier})</span>
            </label>
            <button className="btn !px-3 !py-1.5" onClick={toggleTheme} aria-label="Toggle colour theme">
              {theme === 'dark' ? <Sun size={15} aria-hidden="true" /> : <Moon size={15} aria-hidden="true" />}
            </button>
          </div>
        </div>
      </header>

      {apiMessage && (
        <div
          role="status"
          className="border-b px-4 py-2 text-center text-xs"
          style={{ borderColor: 'var(--border)', color: 'var(--uncertain)', backgroundColor: 'var(--surface)' }}
        >
          {apiMessage}
        </div>
      )}

      <motion.main
        key={location.pathname}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.18 }}
        className="mx-auto w-full max-w-7xl flex-1 px-4 py-6"
      >
        {children}
      </motion.main>

      <footer className="border-t px-4 py-5 text-center text-xs" style={{ borderColor: 'var(--border)', color: 'var(--muted)' }}>
        <p className="mx-auto max-w-3xl">
          AI-text detection is probabilistic and must never be the sole basis for an accusation.
          Veritas reports only numbers produced by evaluation scripts in its own repository.
        </p>
        <p className="mt-1">
          <Link to="/about" className="underline">
            Methodology, datasets and limitations
          </Link>
        </p>
      </footer>
    </div>
  )
}
