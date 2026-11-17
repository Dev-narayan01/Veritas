/** Analysis progress: a document plane with a procedurally drawn text-line texture and
 * a beam whose vertical position is bound to the *real* progress value from the SSE
 * stream. The stage name and percentage are DOM text outside the canvas. */
import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import type { GraphicsTier } from '../../lib/graphics'
import { CanvasShell } from './CanvasShell'

function lineTexture(): THREE.CanvasTexture | null {
  if (typeof document === 'undefined') return null
  const canvas = document.createElement('canvas')
  canvas.width = 512
  canvas.height = 682
  const ctx = canvas.getContext('2d')
  if (!ctx) return null
  ctx.fillStyle = '#0E1424'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  ctx.fillStyle = '#2A3552'
  let y = 60
  let seed = 7
  while (y < canvas.height - 40) {
    seed = (seed * 9301 + 49297) % 233280
    const width = 180 + (seed / 233280) * 280
    ctx.fillRect(48, y, width, 8)
    y += 22
  }
  const texture = new THREE.CanvasTexture(canvas)
  texture.needsUpdate = true
  return texture
}

function Sheet({ progress }: { progress: number }) {
  const texture = useMemo(lineTexture, [])
  const beam = useRef<THREE.Mesh>(null)
  const sheet = useRef<THREE.Mesh>(null)

  useFrame(({ clock }) => {
    if (beam.current) {
      const height = 3.4
      beam.current.position.y = height / 2 - progress * height
      beam.current.scale.x = 1 + Math.sin(clock.getElapsedTime() * 3) * 0.02
    }
    if (sheet.current) sheet.current.rotation.y = Math.sin(clock.getElapsedTime() * 0.4) * 0.08
  })

  return (
    <group>
      <mesh ref={sheet}>
        <planeGeometry args={[2.4, 3.4]} />
        <meshBasicMaterial map={texture ?? undefined} color={texture ? '#ffffff' : '#141C31'} />
      </mesh>
      <mesh ref={beam} position={[0, 1.4, 0.02]}>
        <planeGeometry args={[2.6, 0.14]} />
        <meshBasicMaterial color="#22D3EE" transparent opacity={0.85} />
      </mesh>
    </group>
  )
}

export function Scanner({ tier, progress, stage, message }: { tier: GraphicsTier; progress: number; stage: string; message: string }) {
  const clamped = Math.max(0, Math.min(1, progress))
  return (
    <div className="flex flex-col items-center gap-3">
      <CanvasShell
        tier={tier}
        className="h-56 w-full max-w-sm"
        description={`Analysis progress: ${Math.round(clamped * 100)}% at stage ${stage}.`}
        fallback={<ProgressBar progress={clamped} />}
        camera={{ position: [0, 0, 4.6], fov: 45 }}
      >
        <ambientLight intensity={1} />
        <Sheet progress={clamped} />
      </CanvasShell>
      <div className="text-center">
        <p className="font-mono text-2xl">{Math.round(clamped * 100)}%</p>
        <p className="muted text-sm">{message || stage}</p>
      </div>
      <ProgressBar progress={clamped} srOnly />
    </div>
  )
}

function ProgressBar({ progress, srOnly = false }: { progress: number; srOnly?: boolean }) {
  return (
    <div
      role="progressbar"
      aria-valuenow={Math.round(progress * 100)}
      aria-valuemin={0}
      aria-valuemax={100}
      className={`h-1.5 w-full max-w-sm overflow-hidden rounded-full ${srOnly ? 'sr-only' : ''}`}
      style={{ backgroundColor: 'var(--surface-2)' }}
    >
      <div
        className="h-full rounded-full transition-[width] duration-250"
        style={{ width: `${progress * 100}%`, backgroundImage: 'linear-gradient(90deg, var(--accent), var(--accent-2))' }}
      />
    </div>
  )
}
