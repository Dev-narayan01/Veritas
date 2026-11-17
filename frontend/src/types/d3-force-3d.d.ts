/** Minimal typings for d3-force-3d (the package ships no declarations). */
declare module 'd3-force-3d' {
  export interface SimNode {
    index?: number
    x?: number
    y?: number
    z?: number
    vx?: number
    vy?: number
    vz?: number
  }
  export interface Simulation {
    force(name: string, force?: unknown): Simulation
    tick(iterations?: number): Simulation
    stop(): Simulation
    nodes(nodes?: unknown[]): Simulation
  }
  export interface LinkForce {
    id(fn: (node: unknown, index: number) => number): LinkForce
    distance(value: number): LinkForce
    strength(value: number): LinkForce
  }
  export function forceSimulation(nodes?: unknown[], numDimensions?: number): Simulation
  export function forceManyBody(): unknown
  export function forceCenter(x?: number, y?: number, z?: number): unknown
  export function forceLink(links?: unknown[]): LinkForce
}
