import type { Node } from '../autograd';
import { BOX_HEIGHT, type Block, DISC_HEIGHT, type Vec3 } from './layout';
import { capFor } from './scales';

export interface SceneModule {
  id?: string;
  title: string;
  subtitle?: string;
  nodes: Node[];
  side?: 'left' | 'right' | 'outer';
  depth?: number;
  column?: number;
  chosen?: boolean;
}

export const MODULE_GAP = 1.6;
export const MODULE_STEP = 2.2;
export const MODULE_PAD = 0.4;
export const BRACE_WIDTH = 0.42;
export const BRACE_RADIUS = 0.45;
export const BRACE_MIN_RADIUS = 0.05;
export const BRACE_OPACITY = 0.55;
export const BRACE_LABEL_GAP = 0.6;
export const BRACE_MIN_ZOOM = 0.1;
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
  thick?: boolean;
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
      side: module.side === 'right' ? 'right' : 'left',
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

export const BRACE_THICK_OFFSET = 0.016;

export function levelBraces(
  modules: readonly SceneModule[],
  ranges: ReadonlyMap<string, [number, number]>,
  bounds: SceneBounds & { z: number; cx: number; cz: number },
  gaps: { left: number; right: number; outer: number },
): Brace[] {
  type Placed = { module: SceneModule; side: 'left' | 'right' | 'outer'; column: number; range: [number, number] };
  const placed: Placed[] = [];
  const overlaps = (a: [number, number], b: [number, number]) => a[0] < b[1] && b[0] < a[1];
  const crosses = (module: SceneModule, range: [number, number], other: Placed) => overlaps(range, other.range) && !contains(module, other.module) && !contains(other.module, module);
  const columnOn = (module: SceneModule, side: Placed['side'], range: [number, number], from: number) => {
    const inner = placed.filter((other) => other.side === side && overlaps(range, other.range) && contains(module, other.module));
    let column = Math.max(from, ...inner.map((other) => other.column + 1));
    while (placed.some((other) => other.side === side && other.column === column && crosses(module, range, other))) column++;
    return column;
  };
  const sized = modules.flatMap((module, i) => {
    const range = module.id === undefined ? undefined : ranges.get(module.id);
    return range === undefined ? [] : [{ module, i, range }];
  });
  const order = (side: string) => sized.filter((p) => (p.module.side ?? 'left') === side).sort((p, q) => (side === 'left' ? 1 : -1) * (p.range[1] - p.range[0] - (q.range[1] - q.range[0])) || p.i - q.i);
  for (const side of ['left', 'right', 'outer'] as const) {
    for (const { module, range } of order(side)) {
      const crowded = side !== 'left' && placed.some((other) => other.side === side && crosses(module, range, other));
      const leftFree = !placed.some((other) => other.side === 'left' && crosses(module, range, other));
      const lane = side !== 'left' && crowded && leftFree && module.chosen !== true ? 'left' : side;
      placed.push({ module, side: lane, column: columnOn(module, lane, range, lane === 'left' ? (module.column ?? 0) : 0), range });
    }
  }
  const widest = Math.max(0, ...placed.filter((p) => p.side === 'right').map((p) => p.column));
  const out = new Map<SceneModule, Brace>();
  for (const { module, side, column, range } of placed) {
    const x = side === 'left' ? bounds.xMin - gaps.left - MODULE_STEP * column : side === 'right' ? bounds.xMax + gaps.right + MODULE_STEP * column : bounds.xMax + gaps.right + gaps.outer + MODULE_STEP * (widest + column);
    out.set(module, { module, side: side === 'left' ? 'left' : 'right', thick: side === 'outer', nodes: [], column: 0, x, y0: range[0], y1: range[1], z: bounds.z, cx: bounds.cx, cz: bounds.cz });
  }
  return modules.flatMap((module) => out.get(module) ?? []);
}

function outward(brace: Brace): number {
  return brace.side === 'left' ? -1 : 1;
}

export function braceOffset(brace: Brace, view: Vec3): number {
  const [bx, by, bz] = view;
  return (bx * by * (brace.x - brace.cx) + bz * by * (brace.z - brace.cz)) / Math.max(BRACE_MIN_SLOPE, 1 - by * by);
}

export function braceTip(brace: Brace, offset = 0, scale = 1): Vec3 {
  return [brace.x + BRACE_LABEL_GAP * scale * outward(brace), (brace.y0 + brace.y1) / 2 + offset, brace.z];
}

export function braceScale(zoom: number): number {
  return 1 / Math.min(1, Math.max(BRACE_MIN_ZOOM, zoom));
}

export function bracePoints(brace: Brace, offset = 0, scale = 1): number[] {
  const w = BRACE_WIDTH * scale * outward(brace);
  const y0 = brace.y0 + offset;
  const y1 = brace.y1 + offset;
  const ym = (y0 + y1) / 2;
  const r = Math.min(BRACE_RADIUS * scale, Math.max(BRACE_MIN_RADIUS, (y1 - y0) / 4));
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
