import type { Node } from '../autograd';
import { BOX_HEIGHT, type Block, DISC_HEIGHT, type Vec3 } from './layout';
import { HEIGHT_CAP } from './scales';

export interface SceneModule {
  title: string;
  subtitle?: string;
  nodes: Node[];
  side?: 'left' | 'right';
}

export const MODULE_GAP = 1.6;
export const MODULE_STEP = 2.2;
export const MODULE_PAD = 0.4;
export const BRACE_RADIUS = 0.35;
export const BRACE_ACTIVE_OPACITY = 0.55;
const ARC_SEGMENTS = 6;

export interface BlockSpan {
  node: Node;
  bottom: number;
  top: number;
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
}

export interface SceneBounds {
  xMin: number;
  xMax: number;
}

export function blockSpan(block: Block): BlockSpan {
  const base = Math.min(...block.elements.map((e) => e.y));
  const high = Math.max(...block.elements.map((e) => e.y));
  const rise = block.shape === 'disc' ? DISC_HEIGHT : block.shape === 'box' ? BOX_HEIGHT : HEIGHT_CAP;
  const zs = block.elements.map((e) => e.z);
  return { node: block.node, bottom: block.plate === undefined ? base : Math.min(base, block.plate.y), top: high + rise, z: (Math.min(...zs) + Math.max(...zs)) / 2 };
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

function braceRadius(brace: Brace): number {
  return Math.min(BRACE_RADIUS, (brace.y1 - brace.y0) / 4);
}

export function braceTip(brace: Brace): Vec3 {
  const toward = brace.side === 'left' ? 1 : -1;
  return [brace.x - toward * braceRadius(brace), (brace.y0 + brace.y1) / 2, brace.z];
}

export function bracePoints(brace: Brace): number[] {
  const toward = brace.side === 'left' ? 1 : -1;
  const r = braceRadius(brace);
  const middle = (brace.y0 + brace.y1) / 2;
  const points: number[] = [];
  const arc = (cu: number, cy: number, from: number, to: number) => {
    for (let s = 0; s <= ARC_SEGMENTS; s++) {
      const angle = from + ((to - from) * s) / ARC_SEGMENTS;
      points.push(brace.x + toward * (cu + r * Math.cos(angle)), cy + r * Math.sin(angle), brace.z);
    }
  };
  const quarter = Math.PI / 2;
  arc(r, brace.y1 - r, quarter, 2 * quarter);
  arc(-r, middle + r, 0, -quarter);
  arc(-r, middle - r, quarter, 0);
  arc(r, brace.y0 + r, 2 * quarter, 3 * quarter);
  return points;
}
