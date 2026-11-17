/** Shared React Three Fiber wrapper.
 *
 * Responsibilities (spec 13.3): tier-based dpr and antialias, a `frameloop` that
 * pauses when the tab is hidden or the canvas is off-screen, `aria-hidden` with a
 * text alternative sibling, an error boundary and a `webglcontextlost` handler that
 * swaps in the 2D fallback. Only one shell may be mounted at a time; callers own that
 * rule by rendering a single scene per page.
 */
import { Suspense, useEffect, useRef, useState, type ReactNode } from 'react'
import { Canvas } from '@react-three/fiber'
import { PerformanceMonitor } from '@react-three/drei'
import { BUDGETS, type GraphicsTier } from '../../lib/graphics'

interface Props {
  tier: GraphicsTier
  fallback: ReactNode
  description: string
  children: ReactNode
  onDowngrade?: () => void
  className?: string
  camera?: { position: [number, number, number]; fov?: number }
}

export function CanvasShell({ tier, fallback, description, children, onDowngrade, className, camera }: Props) {
  const [failed, setFailed] = useState(false)
  const [visible, setVisible] = useState(true)
  const [hidden, setHidden] = useState(false)
  const hostRef = useRef<HTMLDivElement>(null)
  const budget = BUDGETS[tier]

  useEffect(() => {
    const onVisibility = () => setHidden(document.hidden)
    document.addEventListener('visibilitychange', onVisibility)
    onVisibility()
    return () => document.removeEventListener('visibilitychange', onVisibility)
  }, [])

  useEffect(() => {
    const node = hostRef.current
    if (!node || typeof IntersectionObserver === 'undefined') return
    const observer = new IntersectionObserver((entries) => setVisible(entries[0]?.isIntersecting ?? true), { threshold: 0.05 })
    observer.observe(node)
    return () => observer.disconnect()
  }, [])

  if (tier === 'off' || failed) {
    return (
      <div className={className} data-scene-fallback="true">
        {fallback}
      </div>
    )
  }

  return (
    <div ref={hostRef} className={className}>
      <Canvas
        aria-hidden="true"
        frameloop={visible && !hidden ? 'always' : 'demand'}
        dpr={budget.dpr}
        gl={{ antialias: tier === 'high', powerPreference: 'high-performance', alpha: true }}
        camera={camera ?? { position: [0, 0, 6], fov: 50 }}
        onCreated={({ gl }) => {
          gl.domElement.addEventListener('webglcontextlost', (event) => {
            event.preventDefault()
            setFailed(true)
          })
        }}
        onError={() => setFailed(true)}
      >
        <PerformanceMonitor onDecline={() => onDowngrade?.()}>
          <Suspense fallback={null}>{children}</Suspense>
        </PerformanceMonitor>
      </Canvas>
      <p className="sr-only">{description}</p>
    </div>
  )
}

/** Text alternative that carries the same information as the canvas. */
export function SceneText({ children }: { children: ReactNode }) {
  return <div className="muted mt-2 text-xs">{children}</div>
}
