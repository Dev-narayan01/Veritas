/** Small presentational primitives shared by the pages. */
import type { ReactNode } from 'react'
import { AlertTriangle, CheckCircle2, Info } from 'lucide-react'

export function Card({ children, className = '', title, actions }: { children: ReactNode; className?: string; title?: ReactNode; actions?: ReactNode }) {
  return (
    <section className={`card p-4 ${className}`}>
      {(title || actions) && (
        <div className="mb-3 flex items-center justify-between gap-2">
          {title ? <h2 className="panel-title flex items-center gap-2">{title}</h2> : <span />}
          {actions}
        </div>
      )}
      {children}
    </section>
  )
}

export function Badge({ children, tone = 'neutral', title }: { children: ReactNode; tone?: 'neutral' | 'good' | 'warn' | 'bad' | 'info'; title?: string }) {
  const colors: Record<string, string> = {
    neutral: 'var(--muted)',
    good: 'var(--human)',
    warn: 'var(--uncertain)',
    bad: 'var(--ai)',
    info: 'var(--accent-2)',
  }
  return (
    <span className="chip" style={{ color: colors[tone], borderColor: colors[tone] }} title={title}>
      {children}
    </span>
  )
}

export function Callout({ children, tone = 'info', title }: { children: ReactNode; tone?: 'info' | 'warn' | 'bad' | 'good'; title?: string }) {
  const map = {
    info: { color: 'var(--accent-2)', Icon: Info },
    warn: { color: 'var(--uncertain)', Icon: AlertTriangle },
    bad: { color: 'var(--ai)', Icon: AlertTriangle },
    good: { color: 'var(--human)', Icon: CheckCircle2 },
  } as const
  const { color, Icon } = map[tone]
  return (
    <div className="card-2 flex gap-2 p-3 text-sm" style={{ borderColor: color }}>
      <Icon size={16} aria-hidden="true" style={{ color, flex: 'none' }} />
      <div>
        {title && <p className="text-xs font-semibold uppercase tracking-wide" style={{ color }}>{title}</p>}
        <div className="muted">{children}</div>
      </div>
    </div>
  )
}

export function Skeleton({ lines = 3 }: { lines?: number }) {
  return (
    <div className="space-y-2" aria-busy="true" aria-label="Loading">
      {Array.from({ length: lines }).map((_, index) => (
        <div key={index} className="skeleton h-4" style={{ width: `${90 - index * 12}%` }} />
      ))}
    </div>
  )
}

export function EmptyState({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="card-2 flex flex-col items-center gap-2 p-8 text-center">
      <p className="font-medium">{title}</p>
      {children && <p className="muted max-w-md text-sm">{children}</p>}
      {action}
    </div>
  )
}

export function Toggle({ checked, onChange, label, description, disabled }: { checked: boolean; onChange: (value: boolean) => void; label: string; description?: string; disabled?: boolean }) {
  return (
    <label className={`flex items-start gap-2 ${disabled ? 'opacity-50' : 'cursor-pointer'}`}>
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
        className="mt-1 h-4 w-4 accent-[var(--accent)]"
      />
      <span>
        <span className="text-sm">{label}</span>
        {description && <span className="muted block text-xs">{description}</span>}
      </span>
    </label>
  )
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="card-2 p-3">
      <p className="panel-title">{label}</p>
      <p className="mt-1 font-mono text-lg">{value}</p>
      {hint && <p className="muted text-xs">{hint}</p>}
    </div>
  )
}
