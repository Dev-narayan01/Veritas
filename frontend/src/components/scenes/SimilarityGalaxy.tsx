/** Batch similarity graph: nodes are documents (radius ∝ √words, colour by verdict),
 * edges are similarities (width/opacity ∝ weight, hidden below 5%). Positions are
 * computed once with a fixed number of d3-force-3d ticks, so the layout is static and
 * deterministic; OrbitControls adds damping. A keyboard-accessible list mirrors it. */
import { useMemo, useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { forceCenter, forceLink, forceManyBody, forceSimulation } from 'd3-force-3d'
import * as THREE from 'three'
import type { GraphicsTier } from '../../lib/graphics'
import type { BatchResult } from '../../types'
import { CanvasShell } from './CanvasShell'

const VERDICT_HEX: Record<string, string> = {
  likely_human: '#34D399',
  uncertain: '#FBBF24',
  likely_ai: '#F87171',
  too_short: '#8B96AD',
}

interface PositionedNode {
  id: string
  label: string
  words: number
  verdict: string
  position: [number, number, number]
}

interface GraphLink {
  source: number
  target: number
  weight: number
}

export function computeLayout(batch: BatchResult): { nodes: PositionedNode[]; links: GraphLink[] } {
  const raw = batch.nodes.map((node, index) => ({ ...node, index }))
  const links = batch.edges
    .map((edge) => ({
      source: batch.nodes.findIndex((n) => n.label === edge.a),
      target: batch.nodes.findIndex((n) => n.label === edge.b),
      weight: edge.weight,
    }))
    .filter((edge) => edge.source >= 0 && edge.target >= 0 && edge.weight >= 5)

  const simulation = forceSimulation(raw as never, 3)
    .force('charge', (forceManyBody as unknown as () => { strength(value: number): unknown })().strength(-40))
    .force('center', forceCenter(0, 0, 0))
    .force('link', forceLink(links as never).id((_node: unknown, index: number) => index).distance(2.4).strength(0.4))
    .stop()
  for (let tick = 0; tick < 120; tick += 1) simulation.tick()

  const nodes = raw.map((node) => ({
    id: node.id,
    label: node.label,
    words: node.words,
    verdict: node.verdict,
    position: [
      Number((node as unknown as { x: number }).x ?? 0),
      Number((node as unknown as { y: number }).y ?? 0),
      Number((node as unknown as { z: number }).z ?? 0),
    ] as [number, number, number],
  }))
  return { nodes, links }
}

function Galaxy({ nodes, links, onSelect }: { nodes: PositionedNode[]; links: GraphLink[]; onSelect: (index: number) => void }) {
  const group = useRef<THREE.Group>(null)
  useFrame((_, delta) => {
    if (group.current) group.current.rotation.y += delta * 0.05
  })
  return (
    <group ref={group}>
      {nodes.map((node, index) => (
        <mesh
          key={node.id}
          position={node.position}
          onClick={() => onSelect(index)}
          scale={0.1 + Math.sqrt(Math.max(1, node.words)) * 0.004}
        >
          <sphereGeometry args={[1, 18, 18]} />
          <meshStandardMaterial
            color={VERDICT_HEX[node.verdict] ?? '#8B96AD'}
            emissive={VERDICT_HEX[node.verdict] ?? '#8B96AD'}
            emissiveIntensity={0.35}
          />
        </mesh>
      ))}
      {links.map((edge, index) => {
        const from = nodes[edge.source]?.position ?? [0, 0, 0]
        const to = nodes[edge.target]?.position ?? [0, 0, 0]
        return (
          <line key={index}>
            <bufferGeometry>
              <bufferAttribute attach="attributes-position" args={[new Float32Array([...from, ...to]), 3]} />
            </bufferGeometry>
            <lineBasicMaterial color="#7C5CFF" transparent opacity={Math.min(0.9, edge.weight / 60)} />
          </line>
        )
      })}
    </group>
  )
}

export function SimilarityGalaxy({ tier, batch, onSelect }: { tier: GraphicsTier; batch: BatchResult; onSelect: (index: number) => void }) {
  const { nodes, links } = useMemo(() => computeLayout(batch), [batch])
  return (
    <CanvasShell
      tier={tier}
      className="h-80 w-full"
      description={`Similarity graph of ${nodes.length} documents: ${nodes.map((n) => n.label).join(', ')}.`}
      fallback={<GalaxyList batch={batch} onSelect={onSelect} />}
      camera={{ position: [0, 1.5, 7], fov: 50 }}
    >
      <ambientLight intensity={0.9} />
      <directionalLight position={[3, 5, 4]} intensity={1} />
      <OrbitControls enablePan={false} enableDamping dampingFactor={0.08} />
      <Galaxy nodes={nodes} links={links} onSelect={onSelect} />
    </CanvasShell>
  )
}

/** Keyboard-accessible mirror of the graph: the same information, no canvas. */
export function GalaxyList({ batch, onSelect }: { batch: BatchResult; onSelect: (index: number) => void }) {
  const [active, setActive] = useState<number | null>(null)
  return (
    <ul className="space-y-1 text-sm" aria-label="Documents in the similarity graph">
      {batch.nodes.map((node, index) => (
        <li key={node.id}>
          <button
            className="btn w-full !justify-between !rounded-lg"
            onClick={() => onSelect(index)}
            onMouseEnter={() => setActive(index)}
            onMouseLeave={() => setActive(null)}
          >
            <span>{node.label}</span>
            <span className="muted font-mono text-xs">
              {node.words} words · {node.verdict.replace('_', ' ')}
              {active === index ? ' · selected' : ''}
            </span>
          </button>
        </li>
      ))}
    </ul>
  )
}
