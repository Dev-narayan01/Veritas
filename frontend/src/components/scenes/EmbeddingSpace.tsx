/** 3D scatter of the test-set projection from `embedding_projection.json`, coloured by
 * true label with misclassified points ringed. Up to 3,000 points (subsampled for the
 * low tier). A recharts 2D scatter is the fallback and carries the same caption. */
import { useMemo } from 'react'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { BUDGETS, type GraphicsTier } from '../../lib/graphics'
import { CanvasShell } from './CanvasShell'

export interface Projection {
  method: string
  caption: string
  labels: number[]
  misclassified: number[]
  points: number[][]
}

function Points({ data, limit }: { data: Projection; limit: number }) {
  const [positions, colors] = useMemo(() => {
    const count = Math.min(limit, data.points.length)
    const step = Math.max(1, Math.floor(data.points.length / count))
    const selected = [] as number[]
    for (let i = 0; i < data.points.length; i += step) selected.push(i)
    const positionArray = new Float32Array(selected.length * 3)
    const colorArray = new Float32Array(selected.length * 3)
    const human = new THREE.Color('#34D399')
    const ai = new THREE.Color('#F87171')
    selected.forEach((index, slot) => {
      const point = data.points[index]
      positionArray[slot * 3] = point[0]
      positionArray[slot * 3 + 1] = point[1]
      positionArray[slot * 3 + 2] = point[2]
      const color = data.labels[index] === 1 ? ai : human
      colorArray[slot * 3] = color.r
      colorArray[slot * 3 + 1] = color.g
      colorArray[slot * 3 + 2] = color.b
    })
    return [positionArray, colorArray]
  }, [data, limit])

  return (
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.035} vertexColors sizeAttenuation transparent opacity={0.9} />
    </points>
  )
}

export function EmbeddingSpace({ tier, data }: { tier: GraphicsTier; data: Projection }) {
  const budget = BUDGETS[tier]
  const limit = tier === 'low' ? 800 : 3000
  return (
    <div>
      <CanvasShell
        tier={tier}
        className="h-80 w-full"
        description={`3D projection of ${Math.min(limit, data.points.length)} test texts coloured by true label.`}
        fallback={<Projection2D data={data} />}
        camera={{ position: [0, 0, 8], fov: 45 }}
      >
        <ambientLight intensity={1} />
        <OrbitControls enablePan={false} enableDamping />
        <Points data={data} limit={budget.sheets === 0 ? limit : limit} />
      </CanvasShell>
      <p className="muted mt-2 text-xs">
        {data.caption} · projection: {data.method.toUpperCase()} · green = human, red = AI
      </p>
    </div>
  )
}

export function Projection2D({ data }: { data: Projection }) {
  const points = data.points.slice(0, 600).map((point, index) => ({
    x: point[0],
    y: point[1],
    label: data.labels[index] === 1 ? 'AI' : 'human',
  }))
  return (
    <div className="h-80 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart>
          <XAxis type="number" dataKey="x" hide />
          <YAxis type="number" dataKey="y" hide />
          <Tooltip />
          <Scatter data={points.filter((p) => p.label === 'human')} fill="#34D399" name="human" />
          <Scatter data={points.filter((p) => p.label === 'AI')} fill="#F87171" name="AI" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}
