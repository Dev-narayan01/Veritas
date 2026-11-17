/** A 270° torus arc whose *length* animates to the probability, coloured by the verdict
 * palette, with faint threshold markers. The number and verdict stay in the DOM and the
 * SVG fallback has the identical layout, so the gauge is decorative by construction. */
import { useMemo, useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import type { GraphicsTier } from '../../lib/graphics'
import { CanvasShell } from './CanvasShell'

const SWEEP = (270 / 360) * Math.PI * 2
const START = Math.PI / 2 + (Math.PI * 2 - SWEEP) / 2
const SEGMENTS = 90

function Arc({ probability, color }: { probability: number; color: string }) {
  const group = useRef<THREE.Group>(null)
  const [display, setDisplay] = useState(0)
  const target = useRef(probability)
  target.current = probability

  useFrame((_, delta) => {
    const next = display + (target.current - display) * Math.min(1, delta * 5)
    if (Math.abs(next - display) > 0.0005) setDisplay(next)
  })

  const visibleSegments = Math.max(0, Math.round(SEGMENTS * Math.min(1, Math.max(0, display))))
  const segments = useMemo(() => {
    return Array.from({ length: SEGMENTS }, (_, index) => START + (SWEEP * index) / SEGMENTS)
  }, [])

  return (
    <group ref={group}>
      {segments.map((angle, index) => {
        const radius = 1
        return (
          <mesh key={index} position={[Math.cos(angle) * radius, Math.sin(angle) * radius, 0]} rotation={[0, 0, angle - Math.PI / 2]}>
            <boxGeometry args={[0.2, 0.07, 0.07]} />
            <meshBasicMaterial color={index < visibleSegments ? color : '#141C31'} transparent opacity={index < visibleSegments ? 0.95 : 0.5} />
          </mesh>
        )
      })}
    </group>
  )
}

function ThresholdMarker({ angle, label }: { angle: number; label: string }) {
  const position = useMemo(
    () => [Math.cos(angle) * 1.28, Math.sin(angle) * 1.28, 0] as [number, number, number],
    [angle],
  )
  return (
    <mesh position={position} name={label}>
      <sphereGeometry args={[0.045, 8, 8]} />
      <meshBasicMaterial color="#E8ECF6" transparent opacity={0.7} />
    </mesh>
  )
}

export function VerdictGauge({
  tier,
  probability,
  tHuman,
  tAi,
  color,
}: {
  tier: GraphicsTier
  probability: number
  tHuman: number
  tAi: number
  color: string
}) {
  const clamped = Math.min(1, Math.max(0, probability))
  return (
    <CanvasShell
      tier={tier}
      className="h-40 w-40"
      description={`Probability of machine authorship ${(clamped * 100).toFixed(1)} percent, human threshold ${(tHuman * 100).toFixed(0)} percent, AI threshold ${(tAi * 100).toFixed(0)} percent.`}
      fallback={<GaugeSvg probability={clamped} tHuman={tHuman} tAi={tAi} color={color} />}
      camera={{ position: [0, 0, 3.1], fov: 45 }}
    >
      <ambientLight intensity={1} />
      <mesh>
        <torusGeometry args={[1, 0.11, 12, 96, SWEEP]} />
        <meshBasicMaterial color="#1F2A44" />
      </mesh>
      <Arc probability={clamped} color={color} />
      <ThresholdMarker angle={START + SWEEP * tHuman} label="human threshold" />
      <ThresholdMarker angle={START + SWEEP * tAi} label="AI threshold" />
    </CanvasShell>
  )
}

export function GaugeSvg({ probability, tHuman, tAi, color }: { probability: number; tHuman: number; tAi: number; color: string }) {
  const radius = 56
  const circumference = 2 * Math.PI * radius
  const sweep = 0.75
  const dash = circumference * sweep
  return (
    <svg
      viewBox="0 0 160 160"
      role="img"
      aria-label={`Probability of machine authorship ${(probability * 100).toFixed(1)} percent`}
      className="h-40 w-40"
    >
      <g transform="rotate(135 80 80)">
        <circle cx="80" cy="80" r={radius} fill="none" stroke="var(--border)" strokeWidth="13" strokeDasharray={`${dash} ${circumference}`} strokeLinecap="round" />
        <circle cx="80" cy="80" r={radius} fill="none" stroke={color} strokeWidth="13" strokeDasharray={`${dash * probability} ${circumference}`} strokeLinecap="round" />
        <circle cx="80" cy="80" r={radius + 9} fill="none" stroke="var(--muted)" strokeWidth="2" strokeDasharray={`2 ${circumference}`} transform={`rotate(${tHuman * 270} 80 80)`} />
        <circle cx="80" cy="80" r={radius + 9} fill="none" stroke="var(--text)" strokeWidth="2" strokeDasharray={`2 ${circumference}`} transform={`rotate(${tAi * 270} 80 80)`} />
      </g>
    </svg>
  )
}
