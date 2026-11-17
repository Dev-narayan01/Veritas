/** Hero background: instanced "document sheets" on slow orbits plus a particle field,
 * with pointer parallax and a swirl when a file is dragged over the drop zone. The
 * canvas is pointer-events-none so it can never block the form. */
import { useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { BUDGETS, type GraphicsTier } from '../../lib/graphics'
import { CanvasShell } from './CanvasShell'

function Sheets({ count, excited }: { count: number; excited: boolean }) {
  const mesh = useRef<THREE.InstancedMesh>(null)
  const dummy = useMemo(() => new THREE.Object3D(), [])
  const seeds = useMemo(
    () =>
      Array.from({ length: count }, (_, index) => ({
        radius: 2.2 + (index % 7) * 0.45,
        speed: 0.05 + (index % 5) * 0.015,
        offset: (index / count) * Math.PI * 2,
        tilt: (index % 3) * 0.4,
        y: ((index % 9) - 4) * 0.55,
      })),
    [count],
  )

  useFrame(({ clock }) => {
    if (!mesh.current) return
    const t = clock.getElapsedTime()
    seeds.forEach((seed, index) => {
      const radius = seed.radius * (excited ? 0.82 : 1)
      const angle = seed.offset + t * seed.speed * (excited ? 4 : 1)
      dummy.position.set(Math.cos(angle) * radius, seed.y + Math.sin(t * 0.4 + seed.offset) * 0.12, Math.sin(angle) * radius)
      dummy.rotation.set(seed.tilt, angle, seed.tilt * 0.5)
      const scale = 0.55 + (index % 4) * 0.08
      dummy.scale.set(scale * 0.75, scale, 0.02)
      dummy.updateMatrix()
      mesh.current!.setMatrixAt(index, dummy.matrix)
    })
    mesh.current.instanceMatrix.needsUpdate = true
  })

  if (count === 0) return null
  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, count]}>
      <planeGeometry args={[1, 1.3]} />
      <meshStandardMaterial color="#7C5CFF" emissive="#22D3EE" emissiveIntensity={excited ? 0.9 : 0.35} roughness={0.4} metalness={0.2} />
    </instancedMesh>
  )
}

function Particles({ count }: { count: number }) {
  const points = useRef<THREE.Points>(null)
  const positions = useMemo(() => {
    const array = new Float32Array(count * 3)
    for (let i = 0; i < count; i += 1) {
      array[i * 3] = (Math.random() - 0.5) * 16
      array[i * 3 + 1] = (Math.random() - 0.5) * 10
      array[i * 3 + 2] = (Math.random() - 0.5) * 10 - 2
    }
    return array
  }, [count])

  useFrame(({ clock }) => {
    if (points.current) points.current.rotation.y = clock.getElapsedTime() * 0.02
  })

  if (count === 0) return null
  return (
    <points ref={points}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.035} color="#22D3EE" transparent opacity={0.7} sizeAttenuation />
    </points>
  )
}

function Parallax() {
  const { camera, pointer } = useThree()
  useFrame(() => {
    camera.position.x += (pointer.x * 0.6 - camera.position.x) * 0.04
    camera.position.y += (pointer.y * 0.35 - camera.position.y) * 0.04
    camera.lookAt(0, 0, 0)
  })
  return null
}

export function HeroField({ tier, dragging }: { tier: GraphicsTier; dragging: boolean }) {
  const budget = BUDGETS[tier]
  return (
    <CanvasShell
      tier={tier}
      className="pointer-events-none absolute inset-0"
      description="Decorative animated background of floating document sheets; it carries no data."
      fallback={
        <div
          aria-hidden="true"
          className="absolute inset-0"
          style={{ backgroundImage: 'radial-gradient(60% 60% at 50% 20%, rgba(124,92,255,0.22), transparent), radial-gradient(40% 50% at 80% 70%, rgba(34,211,238,0.16), transparent)' }}
        />
      }
    >
      <ambientLight intensity={0.7} />
      <hemisphereLight args={['#7C5CFF', '#070A12', 0.6]} />
      <directionalLight position={[3, 5, 4]} intensity={1.1} />
      <Sheets count={budget.sheets} excited={dragging} />
      <Particles count={budget.particles} />
      <Parallax />
    </CanvasShell>
  )
}
