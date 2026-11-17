/** Batch similarity graph: nodes are documents (radius ∝ √words, colour by verdict),
 * edges are similarities (width/opacity ∝ weight, hidden below 5%). Positions come from a
 * fixed number of d3-force-3d ticks, so the layout is deterministic and testable.
 *
 * The scene never rotates itself: nodes are clickable meshes and a rotating group would
 * slide the hitboxes away from the visible spheres. Idle motion is the camera's
 * (OrbitControls auto-rotate), which keeps pointer rays consistent; interaction stops it.
 * A keyboard-accessible list mirrors the graph and is the fallback when WebGL is off.
 */
import { useMemo, useState } from 'react'
import { OrbitControls } from '@react-three/drei'
import { forceCenter, forceLink, forceManyBody, forceSimulation } from 'd3-force-3d'
import type { GraphicsTier } from '../../lib/graphics'
import type { BatchResult } from '../../types'
import { CanvasShell } from './CanvasShell'

const VERDICT_HEX: Record<string, string> = {
  likely_human: '#34D399',
  uncertain: '#FBBF24',
  likely_ai: '#F87171',
  too_short: '#8B96AD',
}
const MIN_EDGE_PERCENT = 5
const TICKS = 120

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

/** Run a fixed number of force ticks and return stable positions. */
export function computeLayout(batch: BatchResult): { nodes: PositionedNode[]; links: GraphLink[] } {
  const raw = batch.nodes.map((node, index) => ({ ...node, index }))
  const links: GraphLink[] = batch.edges
    .map((edge) => ({
      source: batch.nodes.findIndex((node) => node.label === edge.a),
      target: batch.nodes.findIndex((node) => node.label === edge.b),
      weight: edge.weight,
    }))
    .filter((link) => link.source >= 0 && link.target >= 0 && link.weight >= MIN_EDGE_PERCENT)

  const simulation = forceSimulation(raw as never, 3)
    .force('charge', (forceManyBody as unknown as () => { strength(value: number): unknown })().strength(-40))
    .force('center', forceCenter(0, 0, 0))
    .force('link', forceLink(links as never).id((_node: unknown, index: number) => index).distance(2.4).strength(0.4))
    .stop()
  for (let tick = 0; tick < TICKS; tick += 1) simulation.tick()

  const nodes = raw.map((node) => {
    const positioned = node as unknown as { x?: number; y?: number; z?: number }
    return {
      id: node.id,
      label: node.label,
      words: node.words,
      verdict: node.verdict,
      position: [Number(positioned.x ?? 0), Number(positioned.y ?? 0), Number(positioned.z ?? 0)] as [
        number,
        number,
        number,
      ],
    }
  })
  return { nodes, links }
}

function Galaxy({
  nodes,
  links,
  onSelect,
}: {
  nodes: PositionedNode[]
  links: GraphLink[]
  onSelect: (index: number) => void
}) {
  return (
    <group>
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
      {links.map((link, index) => {
        const from = nodes[link.source]?.position ?? [0, 0, 0]
        const to = nodes[link.target]?.position ?? [0, 0, 0]
        return (
          <line key={index}>
            <bufferGeometry>
              <bufferAttribute attach="attributes-position" args={[new Float32Array([...from, ...to]), 3]} />
            </bufferGeometry>
            <lineBasicMaterial color="#7C5CFF" transparent opacity={Math.min(0.9, link.weight / 60)} />
          </line>
        )
      })}
    </group>
  )
}

export function SimilarityGalaxy({
  tier,
  batch,
  onSelect,
}: {
  tier: GraphicsTier
  batch: BatchResult
  onSelect: (index: number) => void
}) {
  const { nodes, links } = useMemo(() => computeLayout(batch), [batch])
  const [spinning, setSpinning] = useState(true)
  return (
    <CanvasShell
      tier={tier}
      className="h-80 w-full"
      description={`Similarity graph of ${nodes.length} documents: ${nodes.map((node) => node.label).join(', ')}.`}
      fallback={<GalaxyList batch={batch} onSelect={onSelect} />}
      camera={{ position: [0, 1.5, 7], fov: 50 }}
    >
      <ambientLight intensity={0.9} />
      <directionalLight position={[3, 5, 4]} intensity={1} />
      <OrbitControls
        enablePan={false}
        enableDamping
        dampingFactor={0.08}
        autoRotate={spinning}
        autoRotateSpeed={0.35}
        onStart={() => setSpinning(false)}
      />
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
