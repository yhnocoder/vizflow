import type { Node } from '../autograd';
import { type CacheDecl, type GenerationState, type ModuleTile, type TileState, tileLeaves, tileStates } from '../steps';
import { offsetOf } from '../tensor';
import type { Block, Element, Plate, ResolvedLayout, TextLabel, Vec3 } from './layout';
import { LEVEL_FRONT, LEVEL_ROW, PLATE_MARGIN, ROW_NAME_LIFT, ROW_NAME_SIDE, TOP_ROOM, rowBands } from './levels';
import { capFor } from './scales';

export const COMPACT_TILE_BACK = 0.5;
export const COMPACT_CACHE_GAP = 1.6;
export const COMPACT_BOARD_GAP = 1.2;
export const COMPACT_LEVEL_GAP = 0.6;
export const COMPACT_SECTION_GAP = 1.4;
export const COMPACT_TITLE_ROOM = 0.7;
export const COMPACT_TITLE_BACK = 0.45;
export const BLOCK_TEXT = '块';

export interface CompactPool {
  blockSize: number;
  table: readonly number[];
}

export interface CompactMetrics {
  tileWidth: number;
  tileDepth: number;
  up: Vec3;
  side: Vec3;
  tokens: readonly string[];
  horizon(node: Node, offset: number): number | undefined;
}

export interface CompactTile {
  tile: ModuleTile | undefined;
  state: TileState;
  y: number;
  x: [number, number];
  z: [number, number];
}

export interface CompactBoard {
  decl: CacheDecl;
  leaf: number;
  block: Block;
  x: [number, number];
  z: [number, number];
  y: number;
}

export interface CompactPlan {
  layout: ResolvedLayout;
  tiles: CompactTile[];
  loops: { tile: ModuleTile; y: [number, number] }[];
  boards: CompactBoard[];
  shared: CompactBoard | undefined;
}

type Shape = [number, number][];

interface Draft {
  decl: CacheDecl;
  leaf: number;
  source: Block;
  full: [number, number];
  positional: boolean;
}

function dot(a: Vec3, b: readonly number[]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function shapeOf(metrics: CompactMetrics, x: [number, number], z: [number, number], y: [number, number]): Shape {
  return x.flatMap((xx) => z.flatMap((zz) => y.map((yy): [number, number] => [dot(metrics.side, [xx, yy, zz]), dot(metrics.up, [xx, yy, zz])])));
}

function envelope(shape: Shape, s: number, pick: (a: number, b: number) => number): number | undefined {
  let found: number | undefined;
  for (const [s0, u0] of shape) for (const [s1, u1] of shape) {
    if (s < Math.min(s0, s1) || s > Math.max(s0, s1)) continue;
    const u = s1 === s0 ? Math.max(u0, u1) : u0 + ((u1 - u0) * (s - s0)) / (s1 - s0);
    found = found === undefined ? u : pick(found, u);
  }
  return found;
}

function clearance(below: Shape, above: Shape): number {
  let need = -Infinity;
  for (const [s] of [...below, ...above]) {
    const top = envelope(below, s, Math.max);
    const bottom = envelope(above, s, Math.min);
    if (top !== undefined && bottom !== undefined) need = Math.max(need, top - bottom);
  }
  return need;
}

export function compactBlocks(layout: ResolvedLayout, tiles: readonly ModuleTile[], generation: GenerationState, pool: CompactPool | undefined, metrics: CompactMetrics): CompactPlan {
  const compact = generation.compact!;
  const leaves = tileLeaves(tiles);
  const states: TileState[] = [...tileStates(leaves, compact), compact.sample ? 'lit' : 'idle'];
  const t = generation.position;
  const T = metrics.tokens.length;
  const sourceOf = layout.levels?.sourceOf;
  const blockOf = (node: Node) => layout.blocks.find((b) => b.node === (sourceOf?.get(node) ?? node) && b.copyOf === undefined);
  const horizonOf = (block: Block, e: Element) => metrics.horizon(block.node, offsetOf(block.node.shape, e.index));
  const bs = pool?.blockSize ?? T;
  const groups = Math.ceil(T / bs);
  const bands = rowBands(Array.from({ length: groups }, () => bs), 0, true);
  const bandZ = (row: number) => bands.start[Math.floor(row / bs)] + (row % bs) * LEVEL_ROW;
  const drafts: Draft[] = leaves.flatMap((leaf, i) =>
    leaf.caches.flatMap((decl) => {
      const entry = generation.caches.find((e) => e.decl === decl)!;
      const nodes = entry.kind === 'state' ? (states[i] === 'idle' ? entry.previous : entry.nodes).slice(0, 1) : decl.nodes;
      return nodes.flatMap((node) => {
        const source = blockOf(node);
        if (source === undefined) return [];
        const positional = entry.kind !== 'state' && source.elements.every((e) => horizonOf(source, e) === e.row);
        const zs = positional && pool !== undefined ? Array.from({ length: groups * bs }, (_, r) => bandZ(r)) : source.elements.map((e) => e.z);
        const pad = source.footprint[1] / 2 + PLATE_MARGIN / 2;
        return [{ decl, leaf: i, source, positional, full: [Math.min(...zs) - pad, Math.max(...zs) + pad] as [number, number] }];
      });
    }),
  );
  const width = (d: Draft) => {
    const xs = d.source.elements.map((e) => e.x);
    return Math.max(...xs) - Math.min(...xs) + d.source.footprint[0] + PLATE_MARGIN;
  };
  const xStart = new Map<Draft, number>();
  leaves.forEach((_, i) => {
    let x = metrics.tileWidth + COMPACT_CACHE_GAP;
    for (const d of drafts.filter((d) => d.leaf === i)) {
      xStart.set(d, x);
      x += width(d) + COMPACT_BOARD_GAP;
    }
  });
  const tileZ: [number, number] = [LEVEL_FRONT - COMPACT_TILE_BACK, LEVEL_FRONT - COMPACT_TILE_BACK + metrics.tileDepth];
  const tileShape = shapeOf(metrics, [0, metrics.tileWidth], tileZ, [0, 0]);
  const shapesAt = (i: number): Shape[] => {
    if (i === leaves.length) return [tileShape];
    const shapes = [tileShape];
    for (const d of drafts.filter((d) => d.leaf === i)) {
      const cap = capFor(d.source.shape, d.source.node.shape.length);
      const x0 = xStart.get(d)!;
      shapes.push(shapeOf(metrics, [x0, x0 + width(d)], [d.full[0] - COMPACT_TITLE_BACK - COMPACT_TITLE_ROOM, d.full[1]], [-cap, cap]));
    }
    return shapes;
  };
  const parentOf = (i: number) => tiles.find((tile) => tile.children.includes(leaves[i]));
  const ys: number[] = [0];
  const upY = Math.max(1e-6, metrics.up[1]);
  for (let i = 1; i <= leaves.length; i++) {
    const gap = i < leaves.length && parentOf(i) !== undefined && parentOf(i) === parentOf(i - 1) ? COMPACT_LEVEL_GAP : COMPACT_SECTION_GAP;
    let need = 0;
    for (const a of shapesAt(i - 1)) for (const b of shapesAt(i)) need = Math.max(need, (clearance(a, b) + gap) / upY);
    ys.push(ys[i - 1] + need);
  }
  const compactTiles: CompactTile[] = ys.map((y, i) => ({ tile: leaves[i], state: states[i], y, x: [0, metrics.tileWidth], z: tileZ }));
  const moved = new Map<Block, Block>();
  const boards: CompactBoard[] = [];
  let shared: CompactBoard | undefined;
  const bandedLeaf = pool === undefined ? undefined : drafts.find((d) => d.positional)?.leaf;
  const labelled = [...drafts].reverse().find((d) => d.positional && d.leaf === bandedLeaf);
  for (const d of drafts) {
    const block = d.source;
    const y = ys[d.leaf];
    const xs = block.elements.map((e) => e.x);
    const dx = xStart.get(d)! + block.footprint[0] / 2 + PLATE_MARGIN / 2 - Math.min(...xs);
    const banded = d.positional && pool !== undefined;
    const elements = block.elements.map((e) => ({ ...e, x: e.x + dx, y, z: banded ? bandZ(e.row) : e.z }));
    const included = elements.filter((_, k) => {
      const h = horizonOf(block, block.elements[k]);
      return h === undefined || !d.positional || h <= t;
    });
    const columns = block.chunks === undefined ? [elements] : block.chunks.map((chunk) => elements.filter((_, k) => Math.abs(block.elements[k].x - chunk.x) <= chunk.w / 2));
    const zRanges: [number, number][] = banded
      ? Array.from({ length: Math.floor(t / bs) + 1 }, (_, g) => [bands.start[g], bands.start[g] + (bs - 1) * LEVEL_ROW])
      : [[Math.min(...included.map((e) => e.z)), Math.max(...included.map((e) => e.z))]];
    const [fw, fd] = block.footprint;
    const plates: Plate[] = columns.flatMap((members) => {
      const mx = members.map((e) => e.x);
      return zRanges.map(([z0, z1]) => ({ x: (Math.min(...mx) + Math.max(...mx)) / 2, y, z: (z0 + z1) / 2, w: Math.max(...mx) - Math.min(...mx) + fw + PLATE_MARGIN, d: z1 - z0 + fd + PLATE_MARGIN }));
    });
    const x0 = Math.min(...plates.map((p) => p.x - p.w / 2));
    const x1 = Math.max(...plates.map((p) => p.x + p.w / 2));
    const z0 = Math.min(...plates.map((p) => p.z - p.d / 2));
    const z1 = Math.max(...plates.map((p) => p.z + p.d / 2));
    const plate: Plate = { x: (x0 + x1) / 2, y, z: (z0 + z1) / 2, w: x1 - x0, d: z1 - z0 };
    const board: CompactBoard = { decl: d.decl, leaf: d.leaf, block, x: [x0, x1], z: [z0, z1], y };
    const rowLabels: TextLabel[] = [];
    if (d === labelled) {
      shared = board;
      for (let g = 0; g * bs <= t; g++) rowLabels.push({ text: `${BLOCK_TEXT} ${pool!.table[g] ?? ''}`, col: g, x: x1 + ROW_NAME_SIDE, y: y + ROW_NAME_LIFT, z: bands.start[g] + ((bs - 1) * LEVEL_ROW) / 2, anchor: 'right' });
    }
    const dy = y - block.elements[0].y;
    moved.set(block, {
      ...block,
      hidden: false,
      quiet: true,
      elements,
      plate,
      chunks: plates.length > 1 ? plates : undefined,
      slabs: undefined,
      rowGroups: banded ? Array.from({ length: block.rows }, (_, r) => Math.floor(r / bs)) : undefined,
      title: { x: (x0 + x1) / 2, y, z: z0 - COMPACT_TITLE_BACK, anchor: 'above' },
      classLabels: block.classLabels.map((label) => ({ ...label, x: label.x + dx, y: label.y + dy })),
      rowLabels,
      axis: block.axis === undefined ? undefined : { x: block.axis.x + dx, z: block.axis.z },
      frameHalf: undefined,
      frameRows: [],
      extent: [],
    });
    boards.push(board);
  }
  const blocks = layout.blocks.map((block) => moved.get(block) ?? { ...block, hidden: true });
  const loops: CompactPlan['loops'] = [];
  let leaf = 0;
  for (const tile of tiles) {
    const count = Math.max(1, tile.children.length);
    if (tile.children.length > 0) loops.push({ tile, y: [ys[leaf], ys[leaf + count - 1]] });
    leaf += count;
  }
  const levels = layout.levels === undefined ? undefined : { ...layout.levels, blocks, boards: [], boxes: [], ranges: new Map<string, [number, number]>(), io: [], chips: [], lo: 0, top: ys[ys.length - 1] + TOP_ROOM };
  for (const board of boards) board.block = moved.get(board.block)!;
  return { layout: { blocks, boards: [], levels }, tiles: compactTiles, loops, boards, shared };
}
