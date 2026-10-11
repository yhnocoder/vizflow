import type { Node } from '../autograd';
import { BOX_HEIGHT, type Block, DISC_HEIGHT, type Vec3 } from './layout';
import { capFor } from './scales';

export interface SceneModule {
  id?: string;
  fixed?: boolean;
  title: string;
  subtitle?: string;
  nodes: Node[];
  side?: 'left' | 'right';
}

export const MODULE_GAP = 1.6;
export const MODULE_STEP = 2.2;
export const MODULE_PAD = 0.4;
export const BRACE_WIDTH = 0.42;
export const BRACE_RADIUS = 0.45;
export const BRACE_MIN_RADIUS = 0.05;
export const BRACE_OPACITY = 0.55;
export const BRACE_LABEL_GAP = 0.6;
export const BRACE_MIN_SLOPE = 0.2;
const BRACE_SEGMENTS = 14;

export interface BlockSpan {
  node: Node;
  bottom: number;
  top: number;
  x: number;
  z: number;
}

export interface Brace {
  module: SceneModule;
  side: 'left' | 'right';
  nodes: Node[];
  column: number;
  x: number;
  y0: number;
  y1: number;
  z: number;
  cx: number;
  cz: number;
}

export interface SceneBounds {
  xMin: number;
  xMax: number;
}

export function blockSpan(block: Block): BlockSpan {
  const base = Math.min(...block.elements.map((e) => e.y));
  const high = Math.max(...block.elements.map((e) => e.y));
  const rise = block.shape === 'disc' ? DISC_HEIGHT : block.shape === 'box' ? BOX_HEIGHT : capFor(block.shape, block.node.shape.length);
  const xs = block.elements.map((e) => e.x);
  const zs = block.elements.map((e) => e.z);
  return {
    node: block.node,
    bottom: block.plate === undefined ? base : Math.min(base, block.plate.y),
    top: high + rise,
    x: (Math.min(...xs) + Math.max(...xs)) / 2,
    z: (Math.min(...zs) + Math.max(...zs)) / 2,
  };
}

export function checkModules(modules: readonly SceneModule[], placed: readonly Node[]): void {
  for (const module of modules) {
    if (!module.nodes.some((n) => placed.includes(n))) throw new Error(`modules: '${module.title}' has no placed node`);
  }
}

function contains(outer: SceneModule, inner: SceneModule): boolean {
  const set = new Set(outer.nodes);
  const innerSet = new Set(inner.nodes);
  return innerSet.size < set.size && [...innerSet].every((n) => set.has(n));
}

export function layoutBraces(modules: readonly SceneModule[], spans: readonly BlockSpan[], bounds: SceneBounds): Brace[] {
  const braces = modules.map((module): Brace => {
    const own = spans.filter((s) => module.nodes.includes(s.node));
    return {
      module,
      side: module.side ?? 'left',
      nodes: own.map((s) => s.node),
      column: 0,
      x: 0,
      y0: Math.min(...own.map((s) => s.bottom)) - MODULE_PAD,
      y1: Math.max(...own.map((s) => s.top)) + MODULE_PAD,
      z: (Math.min(...own.map((s) => s.z)) + Math.max(...own.map((s) => s.z))) / 2,
      cx: (Math.min(...own.map((s) => s.x)) + Math.max(...own.map((s) => s.x))) / 2,
      cz: (Math.min(...own.map((s) => s.z)) + Math.max(...own.map((s) => s.z))) / 2,
    };
  });
  const order = braces.map((_, i) => i).sort((a, b) => new Set(modules[a].nodes).size - new Set(modules[b].nodes).size || a - b);
  const assigned: Brace[] = [];
  for (const i of order) {
    const brace = braces[i];
    const sameSide = assigned.filter((other) => other.side === brace.side);
    let column = 0;
    for (const other of sameSide) if (contains(brace.module, other.module)) column = Math.max(column, other.column + 1);
    const taken = (c: number) => sameSide.some((other) => other.column === c && other.y0 < brace.y1 && brace.y0 < other.y1);
    while (taken(column)) column++;
    brace.column = column;
    brace.x = brace.side === 'left' ? bounds.xMin - MODULE_GAP - MODULE_STEP * column : bounds.xMax + MODULE_GAP + MODULE_STEP * column;
    assigned.push(brace);
  }
  return braces;
}

function outward(brace: Brace): number {
  return brace.side === 'left' ? -1 : 1;
}

export function braceOffset(brace: Brace, view: Vec3): number {
  const [bx, by, bz] = view;
  return (bx * by * (brace.x - brace.cx) + bz * by * (brace.z - brace.cz)) / Math.max(BRACE_MIN_SLOPE, 1 - by * by);
}

export function braceTip(brace: Brace, offset = 0): Vec3 {
  return [brace.x + BRACE_LABEL_GAP * outward(brace), (brace.y0 + brace.y1) / 2 + offset, brace.z];
}

export function bracePoints(brace: Brace, offset = 0): number[] {
  const w = BRACE_WIDTH * outward(brace);
  const y0 = brace.y0 + offset;
  const y1 = brace.y1 + offset;
  const ym = (y0 + y1) / 2;
  const r = Math.min(BRACE_RADIUS, Math.max(BRACE_MIN_RADIUS, (y1 - y0) / 4));
  const points: number[] = [];
  const segment = (p0: [number, number], c1: [number, number], c2: [number, number], p1: [number, number]) => {
    for (let s = 0; s <= BRACE_SEGMENTS; s++) {
      const u = s / BRACE_SEGMENTS;
      const v = 1 - u;
      const at = (k: 0 | 1) => v * v * v * p0[k] + 3 * v * v * u * c1[k] + 3 * v * u * u * c2[k] + u * u * u * p1[k];
      points.push(brace.x + at(0), at(1), brace.z);
    }
  };
  segment([0, y0], [w / 2, y0], [w / 2, y0], [w / 2, y0 + r]);
  segment([w / 2, ym - r], [w / 2, ym], [w / 2, ym], [w, ym]);
  segment([w, ym], [w / 2, ym], [w / 2, ym], [w / 2, ym + r]);
  segment([w / 2, y1 - r], [w / 2, y1], [w / 2, y1], [0, y1]);
  return points;
}
