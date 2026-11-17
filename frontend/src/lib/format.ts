/** Small formatting helpers used across pages. */

export function percent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return `${(value * 100).toFixed(digits)}%`
}

export function plainPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return `${value.toFixed(digits)}%`
}

export function bytes(size: number): string {
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(0)} kB`
  return `${(size / 1024 / 1024).toFixed(1)} MB`
}

export function relativeDate(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function truncate(text: string, max = 160): string {
  if (text.length <= max) return text
  return `${text.slice(0, max - 1)}…`
}

export function stageLabel(stage: string): string {
  const labels: Record<string, string> = {
    queued: 'Queued',
    parse: 'Reading documents',
    integrity: 'Integrity checks',
    ai: 'AI-authorship analysis',
    plagiarism: 'Plagiarism analysis',
    batch: 'Comparing documents',
    report: 'Assembling the report',
    done: 'Done',
    error: 'Failed',
    running: 'Working',
  }
  return labels[stage] ?? stage
}
