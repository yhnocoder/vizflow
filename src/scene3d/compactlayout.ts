import type { Node } from '../autograd';
import { type CacheDecl, type GenerationState, type ModuleTile, type TileState, tileLeaves, tileStates } from '../steps';
import { offsetOf } from '../tensor';
import type { Block, Element, Plate, ResolvedLayout, TextLabel, Vec3 } from './layout';
import { COLUMN_NAME_PLATE_BACK, COLUMN_NAME_PLATE_LIFT, LEVEL_FRONT, LEVEL_ROW, PLATE_MARGIN, ROW_NAME_LIFT, ROW_NAME_SIDE, TOP_ROOM, rowBands } from './levels';
import { capFor } from './scales';

export const COMPACT_MIN_ZOOM = 0.6;
export const COMPACT_MAX_ZOOM = 1.4;
export const COMPACT_CACHE_GAP_PX = 14;
export const COMPACT_BOARD_GAP = 1.2;
export const COMPACT_LEVEL_GAP = 0.6;
export const COMPACT_SECTION_GAP = 1.4;
export const COMPACT_TITLE_ROOM = 0.7;
export const COMPACT_TITLE_BACK = 0.45;
export const COMPACT_NOTE_GAP_PX = 6;
export const COMPACT_ROW_FILL = 0.8;
export const COMPACT_COLUMN_GAP_PX = 6;
export const BLOCK_TEXT = '块';
export const SHARED_TABLE_TEXT = '每一层按同一张块表存放';

export interface CompactPool {
  blockSize: number;
  table: readonly number[];
}

export type CompactText = 'row' | 'note';

export interface CompactMetrics {
  up: Vec3;
  side: Vec3;
  tokens: readonly string[];
  horizon(node: Node, offset: number): number | undefined;
  perUnit: number;
  room: number;
  tile: [number, number];
  measure(text: string, kind: CompactText): [number, number];
}

export interface CompactTile {
  tile: ModuleTile | undefined;
  state: TileState;
  y: number;
  center: Vec3;
  size: [number, number];
}

export interface CompactNote {
  text: string;
  pos: Vec3;
}

export interface CompactBoard {
  decl: CacheDecl;
  leaf: number;
  block: Block;
  x: [number, number];
  z: [number, number];
  y: number;
  note: CompactNote | undefined;
}

export interface CompactPlan {
  layout: ResolvedLayout;
  tiles: CompactTile[];
  loops: { tile: ModuleTile; y: [number, number] }[];
  boards: CompactBoard[];
  extent: Vec3[];
  zoom: number;
  scale: number;
}

type Shape = [number, number][];

interface RowText {
  text: string;
  z: number;
}

interface ColumnText {
  text: string;
  col: number;
  x: number;
}

interface Draft {
  decl: CacheDecl;
  leaf: number;
  source: Block;
  banded: boolean;
  elements: Element[];
  plates: Plate[];
  width: number;
  z: [number, number];
  cap: number;
  rows: RowText[];
  columns: ColumnText[];
  blocks: RowText[];
  note: string | undefined;
}

interface Placed {
  draft: Draft;
  x: number;
  tokens: number;
  right: number;
  rows: RowText[];
  columns: ColumnText[];
}

function dot(a: Vec3, b: readonly number[]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function shapeOf(metrics: CompactMetrics, x: [number, number], z: [number, number], y: [number, number]): Shape {
  return x.flatMap((xx) => z.flatMap((zz) => y.map((yy): [number, number] => [dot(metrics.side, [xx, yy, zz]), dot(metrics.up, [xx, yy, zz])])));
}

function rect(s: [number, number], u: [number, number]): Shape {
  return [
    [s[0], u[0]],
    [s[1], u[0]],
    [s[0], u[1]],
    [s[1], u[1]],
  ];
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

function spaced<T>(items: readonly T[], at: (item: T) => number, step: number): T[] {
  const kept: T[] = [];
  for (const item of [...items].sort((a, b) => at(b) - at(a))) {
    if (kept.length === 0 || at(kept[kept.length - 1]) - at(item) >= step) kept.push(item);
  }
  return kept;
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
  const sharedLeaf = pool === undefined ? undefined : leaves.findIndex((leaf) => leaf.caches.length > 0);
  const drafts: Draft[] = leaves.flatMap((leaf, i) =>
    leaf.caches.flatMap((decl) => {
      const entry = generation.caches.find((e) => e.decl === decl)!;
      const nodes = entry.kind === 'state' ? (states[i] === 'idle' ? entry.previous : entry.nodes).slice(0, 1) : decl.nodes;
      return nodes.flatMap((node): Draft[] => {
        const block = blockOf(node);
        if (block === undefined) return [];
        const positional = entry.kind !== 'state' && block.elements.every((e) => horizonOf(block, e) === e.row);
        const banded = positional && pool !== undefined;
        const xs = block.elements.map((e) => e.x);
        const dx = block.footprint[0] / 2 + PLATE_MARGIN / 2 - Math.min(...xs);
        const elements = block.elements.map((e) => ({ ...e, x: e.x + dx, y: 0, z: banded ? bandZ(e.row) : e.z }));
        const first = entry.read.length === 0 ? t : Math.min(t, ...entry.read);
        const shown = elements.filter((_, k) => {
          const h = horizonOf(block, block.elements[k]);
          return h === undefined || !positional || h <= t;
        });
        const columns = block.chunks === undefined ? [elements] : block.chunks.map((chunk) => elements.filter((_, k) => Math.abs(block.elements[k].x - chunk.x) <= chunk.w / 2));
        const zRanges: [number, number][] = banded
          ? Array.from({ length: Math.floor(t / bs) + 1 }, (_, g) => [bands.start[g], bands.start[g] + (bs - 1) * LEVEL_ROW])
          : [[Math.min(...shown.map((e) => e.z)), Math.max(...shown.map((e) => e.z))]];
        const [fw, fd] = block.footprint;
        const plates: Plate[] = columns.flatMap((members) => {
          const mx = members.map((e) => e.x);
          return zRanges.map(([z0, z1]) => ({ x: (Math.min(...mx) + Math.max(...mx)) / 2, y: 0, z: (z0 + z1) / 2, w: Math.max(...mx) - Math.min(...mx) + fw + PLATE_MARGIN, d: z1 - z0 + fd + PLATE_MARGIN }));
        });
        const width = Math.max(...plates.map((p) => p.x + p.w / 2));
        const z: [number, number] = [Math.min(...plates.map((p) => p.z - p.d / 2)), Math.max(...plates.map((p) => p.z + p.d / 2))];
        const horizons = block.elements.map((e) => horizonOf(block, e));
        const constant = (key: (e: Element) => number) => {
          const seen = new Map<number, number | undefined>();
          return block.elements.every((e, k) => {
            if (!seen.has(key(e))) seen.set(key(e), horizons[k]);
            return seen.get(key(e)) === horizons[k];
          });
        };
        const byRow = entry.kind !== 'state' && constant((e) => e.row);
        const byColumn = entry.kind !== 'state' && !byRow && constant((e) => e.col);
        const rows = new Map<number, RowText>();
        const cols = new Map<number, ColumnText>();
        elements.forEach((e, k) => {
          const h = horizons[k];
          if (h === undefined || h > t || h < first) return;
          if (byRow && !rows.has(e.row)) rows.set(e.row, { text: metrics.tokens[h] ?? '', z: e.z });
          if (byColumn && !cols.has(e.col)) cols.set(e.col, { text: metrics.tokens[h] ?? '', col: e.col, x: e.x });
        });
        return [{ decl, leaf: i, source: block, banded, elements, plates, width, z, cap: capFor(block.shape, block.node.shape.length), rows: [...rows.values()], columns: [...cols.values()], blocks: [], note: entry.kind === 'state' ? `固定 ${entry.values} 个数` : undefined }];
      });
    }),
  );
  drafts.forEach((d, k) => {
    const earlier = drafts.slice(0, k).filter((e) => e.leaf === d.leaf).flatMap((e) => e.rows.map((row) => `${row.text}@${row.z}`));
    if (d.rows.every((row) => earlier.includes(`${row.text}@${row.z}`))) d.rows = [];
  });
  const shared = sharedLeaf === undefined ? undefined : [...drafts].reverse().find((d) => d.leaf === sharedLeaf && d.banded);
  if (shared !== undefined) {
    for (let g = 0; g * bs <= t; g++) shared.blocks.push({ text: `${BLOCK_TEXT} ${pool!.table[g] ?? ''}`, z: bands.start[g] + ((bs - 1) * LEVEL_ROW) / 2 });
    shared.note = SHARED_TABLE_TEXT;
  }
  const widest = (texts: readonly RowText[]) => Math.max(0, ...texts.map((row) => metrics.measure(row.text, 'row')[0]));
  const rowHeight = metrics.measure('|', 'row')[1];
  const place = (scale: number): Placed[] => {
    const placed: Placed[] = [];
    leaves.forEach((_, i) => {
      let x = 0;
      for (const draft of drafts.filter((d) => d.leaf === i)) {
        const step = (COMPACT_ROW_FILL * rowHeight) / scale / Math.max(1e-6, Math.abs(metrics.up[2]));
        const rows = spaced(draft.rows, (row) => row.z, step);
        const columnStep = (Math.max(0, ...draft.columns.map((column) => metrics.measure(column.text, 'row')[0])) + COMPACT_COLUMN_GAP_PX) / scale / Math.max(1e-6, Math.abs(metrics.side[0]));
        const columns = spaced(draft.columns, (column) => column.x, columnStep);
        const tokens = rows.length === 0 ? 0 : ROW_NAME_SIDE + widest(rows) / scale;
        const right = tokens + (draft.blocks.length === 0 ? 0 : ROW_NAME_SIDE + widest(draft.blocks) / scale);
        placed.push({ draft, x, tokens, right, rows, columns });
        x += draft.width + right + COMPACT_BOARD_GAP;
      }
    });
    return placed;
  };
  const columnRoom = (d: Draft, scale: number) => (d.columns.length === 0 ? 0 : rowHeight / scale / Math.max(1e-6, Math.abs(metrics.up[2])));
  const columnZ = (d: Draft) => Math.min(...d.elements.map((e) => e.z)) - COLUMN_NAME_PLATE_BACK;
  const titleZ = (d: Draft, scale: number) => (d.columns.length === 0 ? d.z[0] - COMPACT_TITLE_BACK : columnZ(d) - columnRoom(d, scale));
  const boxOf = (p: Placed, scale: number): Shape => shapeOf(metrics, [p.x, p.x + p.draft.width + p.right], [titleZ(p.draft, scale) - COMPACT_TITLE_ROOM, p.draft.z[1]], [-p.draft.cap, p.draft.cap]);
  const bare = place(Infinity);
  const span = Math.max(
    0,
    ...leaves.map((_, i) => {
      const ss = bare.filter((p) => p.draft.leaf === i).flatMap((p) => boxOf(p, Infinity).map(([s]) => s));
      return ss.length === 0 ? 0 : Math.max(...ss) - Math.min(...ss);
    }),
  );
  const labelPx = Math.max(0, ...leaves.map((_, i) => drafts.filter((d) => d.leaf === i).reduce((sum, d) => sum + widest(d.rows) + widest(d.blocks), 0)));
  const fit = span === 0 ? COMPACT_MAX_ZOOM : (metrics.room - metrics.tile[0] - COMPACT_CACHE_GAP_PX - labelPx) / (span * metrics.perUnit);
  const zoom = Math.min(COMPACT_MAX_ZOOM, Math.max(COMPACT_MIN_ZOOM, fit));
  const scale = zoom * metrics.perUnit;
  const placed = place(scale);
  const noteRect = (p: Placed, box: Shape): Shape | undefined => {
    if (p.draft.note === undefined) return undefined;
    const [w, h] = metrics.measure(p.draft.note, 'note');
    const s = dot(metrics.side, [p.x + p.draft.width / 2, 0, (p.draft.z[0] + p.draft.z[1]) / 2]);
    const top = Math.min(...box.map(([, u]) => u)) - COMPACT_NOTE_GAP_PX / scale;
    return rect([s - w / (2 * scale), s + w / (2 * scale)], [top - h / scale, top]);
  };
  const boardShapes = placed.map((p) => {
    const box = boxOf(p, scale);
    const note = noteRect(p, box);
    return { p, box, note };
  });
  const allS = boardShapes.flatMap(({ box }) => box.map(([s]) => s));
  const tileSize: [number, number] = [metrics.tile[0] / scale, metrics.tile[1] / scale];
  const sRight = (allS.length === 0 ? 0 : Math.min(...allS)) - COMPACT_CACHE_GAP_PX / scale;
  const sCenter = sRight - tileSize[0] / 2;
  const anchor = placed[0];
  const ref: Vec3 = anchor === undefined ? [0, 0, LEVEL_FRONT] : [anchor.x + anchor.draft.width / 2, 0, (anchor.draft.z[0] + anchor.draft.z[1]) / 2];
  const uRef = dot(metrics.up, ref);
  const tileShape = rect([sCenter - tileSize[0] / 2, sCenter + tileSize[0] / 2], [uRef - tileSize[1] / 2, uRef + tileSize[1] / 2]);
  const shapesAt = (i: number): Shape[] => [tileShape, ...boardShapes.filter(({ p }) => p.draft.leaf === i).flatMap(({ box, note }) => (note === undefined ? [box] : [box, note]))];
  const parentOf = (i: number) => tiles.find((tile) => tile.children.includes(leaves[i]));
  const ys: number[] = [0];
  const upY = Math.max(1e-6, metrics.up[1]);
  for (let i = 1; i <= leaves.length; i++) {
    const gap = i < leaves.length && parentOf(i) !== undefined && parentOf(i) === parentOf(i - 1) ? COMPACT_LEVEL_GAP : COMPACT_SECTION_GAP;
    let need = 0;
    for (const a of shapesAt(i - 1)) for (const b of shapesAt(i)) need = Math.max(need, (clearance(a, b) + gap) / upY);
    ys.push(ys[i - 1] + need);
  }
  const sOf = (y: number) => dot(metrics.side, [ref[0], y, ref[2]]);
  const at = (s: number, u: number, y: number): Vec3 => {
    const v = u + metrics.up[1] * y;
    return [metrics.side[0] * s + metrics.up[0] * v, metrics.side[1] * s + metrics.up[1] * v, metrics.side[2] * s + metrics.up[2] * v];
  };
  const compactTiles: CompactTile[] = ys.map((y, i) => {
    const ds = sCenter - sOf(y);
    return { tile: leaves[i], state: states[i], y, center: [ref[0] + metrics.side[0] * ds, y, ref[2] + metrics.side[2] * ds], size: tileSize };
  });
  const moved = new Map<Block, Block>();
  const boards: CompactBoard[] = [];
  const extent: Vec3[] = [];
  for (const { p, note } of boardShapes) {
    const { draft } = p;
    const block = draft.source;
    const y = ys[draft.leaf];
    const shift = (e: { x: number; y: number }) => ({ x: e.x + p.x, y: e.y + y });
    const elements = draft.elements.map((e) => ({ ...e, ...shift(e) }));
    const plates = draft.plates.map((plate) => ({ ...plate, ...shift(plate) }));
    const x0 = p.x;
    const x1 = p.x + draft.width;
    const [z0, z1] = draft.z;
    const rowLabels: TextLabel[] = [
      ...p.rows.map((row, k): TextLabel => ({ text: row.text, col: k, x: x1 + ROW_NAME_SIDE, y: y + ROW_NAME_LIFT, z: row.z, anchor: 'right' })),
      ...draft.blocks.map((row, g): TextLabel => ({ text: row.text, col: g, x: x1 + p.tokens + ROW_NAME_SIDE, y: y + ROW_NAME_LIFT, z: row.z, anchor: 'right' })),
    ];
    for (const row of [...p.rows, ...draft.blocks]) extent.push([x1 + p.right, y, row.z]);
    let placedNote: CompactNote | undefined;
    if (note !== undefined) {
      const s = (note[0][0] + note[1][0]) / 2;
      placedNote = { text: draft.note!, pos: at(s, note[2][1], y) };
      extent.push(...note.map(([s, u]) => at(s, u, y)));
    }
    const dx = elements[0].x - block.elements[0].x;
    const dy = y - block.elements[0].y;
    moved.set(block, {
      ...block,
      hidden: false,
      quiet: true,
      elements,
      plate: { x: (x0 + x1) / 2, y, z: (z0 + z1) / 2, w: x1 - x0, d: z1 - z0 },
      chunks: plates.length > 1 ? plates : undefined,
      slabs: undefined,
      rowGroups: draft.banded ? Array.from({ length: block.rows }, (_, r) => Math.floor(r / bs)) : undefined,
      title: { x: (x0 + x1) / 2, y, z: titleZ(draft, scale), anchor: 'above' },
      classLabels:
        draft.columns.length === 0
          ? block.classLabels.map((label) => ({ ...label, x: label.x + dx, y: label.y + dy }))
          : p.columns.map((column): TextLabel => ({ text: column.text, col: column.col, x: column.x + p.x, y: y + COLUMN_NAME_PLATE_LIFT, z: columnZ(draft), anchor: 'above' })),
      rowLabels,
      axis: block.axis === undefined ? undefined : { x: block.axis.x + dx, z: block.axis.z },
      frameHalf: undefined,
      frameRows: [],
      extent: [],
    });
    boards.push({ decl: draft.decl, leaf: draft.leaf, block, x: [x0, x1], z: [z0, z1], y, note: placedNote });
  }
  const blocks = layout.blocks.map((block) => moved.get(block) ?? { ...block, hidden: true });
  const loops: CompactPlan['loops'] = [];
  let leaf = 0;
  for (const tile of tiles) {
    const count = Math.max(1, tile.children.length);
    if (tile.children.length > 0) loops.push({ tile, y: [ys[leaf], ys[leaf + count - 1]] });
    leaf += count;
  }
  const halfY = tileSize[1] / 2 / upY;
  const levels = layout.levels === undefined ? undefined : { ...layout.levels, blocks, boards: [], boxes: [], ranges: new Map<string, [number, number]>(), io: [], chips: [], lo: ys[0] - halfY, top: ys[ys.length - 1] + halfY + TOP_ROOM };
  for (const board of boards) board.block = moved.get(board.block)!;
  return { layout: { blocks, boards: [], levels }, tiles: compactTiles, loops, boards, extent, zoom, scale };
}
