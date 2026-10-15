import type { Node } from '../autograd';
import { explainOf } from '../explain';
import { isLeafLike } from '../steps';
import { forEachIndex } from '../tensor';
import { plates, rowCol, rowsCols, sourceIndex } from './layout';
import type { LayoutOptions } from './schemes';

export const STREAM_X = -7.2;
export const STREAM_SPACING = 0.95;
export const BOARD_GAP = 1.45;
export const COLUMN_BREAK = 0.6;
export const CHIP_ROOM = 3.5;
export const LABEL_CHAR = 0.3;
export const SCRIPT_SCALE = 0.6;
export const BLOCK_HALF = 0.5;
export const SPACING_MAX = 1.45;
export const SPACING_MIN = 0.74;
export const SPACING_SPAN = 11.4;
export const DISC_SIDE = 1.6;
export const DISC_NEAR_SIDE = 0.62;
export const DISC_GROUP_SIDE = 0.67;
export const LEVEL_ROW = 1.0;
export const STACK_GAP = 0.9;
export const CHIP_LABEL_SIDE = 0.6;

export const TITLE = {
  left: [-0.55, 0.3] as const,
  board: [-0.55, -0.45] as const,
  below: [-0.55, 0.9] as const,
  right: [0.6, 0.35] as const,
  vector: [0.55, 0.35] as const,
  node: [0.05, 0.28, -0.2] as const,
};

export type Kind = 'plate' | 'tile' | 'vector' | 'disc' | 'box';
export type TitleKind = 'left' | 'below' | 'right' | 'none';
export type Side = 'left' | 'right';

export interface Track {
  id: number;
  x0: number;
  spacing: number;
  stream: boolean;
  breaks: Map<number, number>;
  members: Entry[];
  anchor: Track | undefined;
  side: Side;
  bound: Entry | undefined;
}

export interface Entry {
  key: string;
  node: Node;
  kind: Kind;
  level: Level;
  rows: number;
  cols: number;
  track: Track | undefined;
  host: Entry | undefined;
  offsets: number[];
  xs: number[] | undefined;
  title: TitleKind;
  quiet: boolean;
  table: Node | undefined;
  transposed: boolean;
  zShift: number;
}

export interface Level {
  entries: Entry[];
  module: string | null;
  y: number;
}

export interface BoardRef {
  op: Node;
  a: Node;
  b: Node;
  linear: boolean;
  batched: boolean;
}

export interface ColumnGraph {
  levels: Level[];
  entries: Entry[];
  options: LayoutOptions;
  sides: { node: Node; side: 'with' | 'left' | 'right'; of: Node }[];
  merges: { left: Node[]; right: Node[] }[];
  boards: ReadonlyMap<Node, BoardRef>;
  tiles: ReadonlyMap<Node, Entry>;
  kernelTiles: ReadonlyMap<Node, Entry>;
  stacked: ReadonlyMap<Node, Node>;
  copies: ReadonlyMap<Entry, Entry>;
  tables: ReadonlyMap<Node, Node>;
  slabbed: ReadonlyMap<Node, Node>;
  drawnOn: ReadonlyMap<Node, Node[]>;
  computed: ReadonlySet<Node>;
  entryOf: ReadonlyMap<Node, Entry>;
  sourceOf: ReadonlyMap<Node, Node>;
  source(n: Node): Node;
  consumers(n: Node): Node[];
  prepared(n: Node): boolean;
  labelled(n: Node): boolean;
}

type Span = [number, number];

interface Piece {
  entry: Entry;
  index: number;
  lo: number;
  hi: number;
  offset?: number;
  count?: number;
}

const TEX_WRAPPERS = /\\(?:mathbf|mathrm|mathit|mathcal|boldsymbol|operatorname|text|textrm|tilde|hat|bar|overline|vec|widehat|widetilde)\s*(?=\{)/g;
const TEX_SPACING = /\\(?:left|right|,|;|!|:|quad|qquad|thinspace)/g;
const TEX_FUNCTIONS = /\\(cos|sin|tan|exp|log|ln|max|min|mean|sum|softmax|sigmoid|silu|tanh|sqrt|det|arg|lim|norm|cat|diag|tr)(?![a-zA-Z])/g;

export function texWidth(tex: string): number {
  const s = tex.replace(TEX_SPACING, '').replace(TEX_WRAPPERS, '').replace(TEX_FUNCTIONS, '$1');
  let count = 0;
  const stack: number[] = [];
  let scale = 1;
  let script = false;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === '{') {
      stack.push(scale);
      if (script) scale *= SCRIPT_SCALE;
      script = false;
      continue;
    }
    if (c === '}') {
      scale = stack.pop() ?? 1;
      continue;
    }
    if (c === '^' || c === '_') {
      script = true;
      continue;
    }
    if (c === ' ') continue;
    const own = script ? scale * SCRIPT_SCALE : scale;
    script = false;
    if (c === '\\') {
      const m = /^\\[a-zA-Z]+/.exec(s.slice(i));
      if (m !== null) {
        i += m[0].length - 1;
        count += own;
        continue;
      }
      i++;
      count += own;
      continue;
    }
    count += own;
  }
  return count * LABEL_CHAR;
}

export function laneSpacing(cols: number): number {
  return Math.min(SPACING_MAX, Math.max(SPACING_MIN, SPACING_SPAN / cols));
}

function consecutive(start: number, n: number): number[] {
  return Array.from({ length: n }, (_, i) => start + i);
}

export function position(track: Track, i: number): number {
  let breaks = 0;
  for (const [b, width] of track.breaks) if (b <= i) breaks += width;
  return track.x0 + i * track.spacing + breaks;
}

function slabXs(e: Entry): number[] {
  const x0 = position(e.track!, e.offsets[0]);
  return e.offsets.map((_, i) => x0 + i * e.track!.spacing);
}

function boardStart(op: Node): number {
  const head = (op.attrs as { index?: number }).index;
  return head === undefined ? 0 : head * op.inputs[0].shape[2];
}

function rank(e: Entry): number {
  if (e.kind === 'tile' && e.table !== undefined) return 1;
  if (e.kind === 'vector') return 5;
  if (isLeafLike(e.node)) return 4;
  return 2;
}

function overlaps(a: Span, b: Span, gap: number): boolean {
  return a[0] < b[1] + gap - 1e-9 && b[0] < a[1] + gap - 1e-9;
}

export class Columns {
  readonly tracks: Track[] = [];
  readonly stream: Track;
  private readonly done = new Set<Entry>();
  private readonly attached: Entry[] = [];
  private readonly nudges = new Map<Entry, number[]>();
  private readonly widened = new Map<string, Map<number, number>>();
  private readonly gap: number;

  constructor(private readonly g: ColumnGraph) {
    this.gap = g.options.gap === 'labels' ? 0 : g.options.gap;
    this.stream = this.track(STREAM_SPACING, undefined, 'right');
    this.stream.stream = true;
  }

  run(): void {
    this.assignMembers();
    this.relativeXs();
    this.assignTitles();
    this.widenBreaks();
    this.pack();
    this.absoluteXs();
  }

  private track(spacing: number, anchor: Track | undefined, side: Side, bound?: Entry): Track {
    const track: Track = { id: this.tracks.length, x0: 0, spacing, stream: false, breaks: new Map(), members: [], anchor, side, bound };
    this.tracks.push(track);
    return track;
  }

  private levelIndex(e: Entry): number {
    return this.g.levels.indexOf(e.level);
  }

  private levelKey(e: Entry): string {
    return `${this.levelIndex(e)}:${e.track!.id}`;
  }

  private widening(e: Entry, o: number): number {
    let total = 0;
    for (const [at, width] of this.widened.get(this.levelKey(e)) ?? []) if (at <= o) total += width;
    return total;
  }

  private order(e: Entry): number {
    const fed = this.g.computed.has(e.node) && e.node.inputs.some((i) => this.g.computed.has(this.g.source(i)) || this.g.sourceOf.has(i));
    return rank(e) + (this.g.computed.has(e.node) && !fed && e.kind === 'plate' ? 1 : 0);
  }

  private vectorHost(e: Entry): Entry | undefined {
    const n = e.node;
    const user = this.g.consumers(n)[0];
    const placed = this.g.sides.find((side) => side.node === n);
    if (placed !== undefined) return this.g.entryOf.get(placed.of);
    if (user === undefined) return undefined;
    const target = this.g.entryOf.get(user);
    return target?.level === e.level ? target : this.g.entryOf.get(user.inputs.find((i) => this.g.source(i) !== n) ?? user);
  }

  private leafHost(e: Entry): Entry | undefined {
    const n = e.node;
    const placed = this.g.sides.find((side) => side.node === n);
    if (placed !== undefined) return this.g.entryOf.get(placed.of);
    const user = this.g.consumers(n)[0];
    if (user === undefined) return undefined;
    const main = user.inputs.find((i) => this.g.source(i) !== n);
    return main === undefined ? this.g.entryOf.get(user) : this.g.entryOf.get(this.g.source(main));
  }

  private entryBehind(n: Node): Entry | undefined {
    const seen = new Set<Node>();
    for (let at: Node | undefined = n; at !== undefined && !seen.has(at); at = at.inputs[0]) {
      seen.add(at);
      const found = this.g.entryOf.get(at) ?? this.g.entryOf.get(this.g.source(at));
      if (found !== undefined) return found;
      if (isLeafLike(at)) return undefined;
    }
    return undefined;
  }

  private hostTrack(e: Entry | undefined): Track | undefined {
    const seen = new Set<Entry>();
    for (let at = e; at !== undefined && !seen.has(at); at = at.host ?? this.g.entryOf.get(this.g.source(at.node.inputs[0] ?? at.node))) {
      seen.add(at);
      if (at.track !== undefined) return at.track;
    }
    return undefined;
  }

  private drawnA(board: BoardRef): Entry {
    return this.g.entryOf.get(board.a)!;
  }

  private join(e: Entry, track: Track, offsets: number[]): void {
    e.track = track;
    e.offsets = offsets;
    track.members.push(e);
    this.followers(e);
  }

  private attach(e: Entry, host: Entry): void {
    e.host = host;
    this.attached.push(e);
    this.followers(e);
  }

  private followers(e: Entry): void {
    const tile = this.g.kernelTiles.get(e.node);
    if (tile !== undefined && e.track !== undefined) this.join(tile, e.track, e.offsets);
  }

  private free(level: Level, track: Track, offsets: number[]): boolean {
    for (const o of level.entries) {
      if (o.track !== track || o.kind === 'tile' || o.kind === 'vector' || this.g.copies.has(o)) continue;
      if (o.offsets.some((k) => offsets.includes(k))) return false;
    }
    return true;
  }

  private open(e: Entry, offsets: number[], anchor: Track | undefined, spacing = Math.min(STREAM_SPACING, laneSpacing(e.cols))): void {
    const n = e.node;
    let at = anchor;
    let side: Side = this.g.options.prepare === 'side' && this.g.prepared(n) && !isLeafLike(n) ? 'left' : 'right';
    const placed = this.g.sides.find((s) => s.node === n && s.side !== 'with');
    const target = placed === undefined ? undefined : this.hostTrack(this.g.entryOf.get(placed.of));
    if (target !== undefined) {
      at = target;
      side = placed!.side as Side;
    }
    const merge = this.g.merges.find((m) => m.right.includes(n));
    if (merge !== undefined && placed === undefined) {
      const left = e.level.entries.find((o) => merge.left.includes(o.node) && o.track !== undefined);
      if (left !== undefined) at = left.track;
    }
    if (this.stream.members.length === 0 && at === undefined) {
      this.join(e, this.stream, offsets);
      return;
    }
    this.join(e, this.track(spacing, at ?? this.stream, side), offsets);
  }

  private assignMembers(): void {
    for (const level of this.g.levels) {
      const ordered = [...level.entries].sort((p, q) => this.order(p) - this.order(q));
      for (const e of ordered) this.place(e);
    }
    for (const [e, base] of this.g.copies) if (base.track !== undefined) this.join(e, base.track, base.offsets);
  }

  private place(e: Entry): void {
    if (this.done.has(e) || this.g.copies.has(e)) return;
    this.done.add(e);
    const n = e.node;
    const level = e.level;
    if (e.kind === 'tile' && e.table === undefined) return;
    if (e.table !== undefined) {
      this.open(e, consecutive(0, e.cols), undefined);
      return;
    }
    const table = this.g.tables.get(n);
    if (table !== undefined) {
      const tile = this.g.entryOf.get(table)!;
      this.place(tile);
      this.join(e, tile.track!, tile.offsets);
      return;
    }
    const board = this.g.boards.get(n);
    if (board !== undefined) {
      this.placeBoardGroup(board, level);
      return;
    }
    if (e.kind === 'vector') {
      const host = this.vectorHost(e);
      if (host !== undefined) this.place(host);
      const placed = this.g.sides.some((s) => s.node === n && s.side !== 'with');
      if (host === undefined || placed) this.open(e, consecutive(0, e.cols), this.hostTrack(host));
      else if (host.track !== undefined) this.join(e, host.track, host.offsets.length >= e.cols ? host.offsets.slice(0, e.cols) : consecutive(host.offsets[0] ?? 0, e.cols));
      else this.attach(e, host);
      return;
    }
    if (isLeafLike(n)) {
      const host = this.leafHost(e);
      if (host !== undefined) this.place(host);
      this.open(e, consecutive(0, e.cols), this.hostTrack(host));
      return;
    }
    const inputs = n.inputs.map((i) => this.entryBehind(i)).filter((i): i is Entry => i !== undefined);
    const main = inputs[0];
    if (main !== undefined) this.place(main);
    if (main === undefined || (main.kind === 'vector' && main.track === undefined && main.host === undefined)) {
      this.open(e, consecutive(0, e.cols), undefined);
      return;
    }
    if (e.kind === 'disc' || e.kind === 'box' || (e.cols === 1 && main.track === undefined)) {
      this.attach(e, main);
      return;
    }
    if (main.track === undefined) {
      this.open(e, consecutive(0, e.cols), this.hostTrack(main));
      return;
    }
    const traced = this.tracedOffsets(e, main.track);
    const joined = inputs.flatMap((i) => i.offsets);
    const cat = n.kind === 'op' && n.op === 'cat' && inputs.every((i) => i.track === main.track) && new Set(joined).size === joined.length;
    const offsets = traced ?? (cat ? joined : this.inherited(e, main));
    if (this.free(level, main.track, offsets)) this.join(e, main.track, offsets);
    else this.open(e, consecutive(0, e.cols), main.track);
  }

  private inherited(e: Entry, main: Entry): number[] {
    const n = e.node;
    const start = n.kind === 'op' && n.op === 'narrow' ? ((n.attrs as { start: number }).start ?? 0) : n.kind === 'op' && n.op === 'select' && n.inputs[0].shape.length === 3 && (n.attrs as { dim: number }).dim === 1 ? (n.attrs as { index: number }).index * n.inputs[0].shape[2] : 0;
    const stride = n.kind === 'op' && n.op === 'narrow' ? ((n.attrs as { step?: number }).step ?? 1) : 1;
    return main.offsets.length >= start + (e.cols - 1) * stride + 1 ? Array.from({ length: e.cols }, (_, i) => main.offsets[start + i * stride]) : consecutive(main.offsets[0] ?? 0, e.cols);
  }

  private tracedOffsets(e: Entry, track: Track): number[] | undefined {
    const n = e.node;
    const def = explainOf(n.op as string);
    if (def?.pattern !== 'elementwise' || def.selected === undefined || e.cols !== rowsCols(n)[1]) return undefined;
    const ctx = { node: n, inputs: n.inputs.map((i) => i.value), out: n.value, attrs: n.attrs };
    const offsets: (number | undefined)[] = Array.from({ length: e.cols }, () => undefined);
    let traced = true;
    forEachIndex(n.shape, (index) => {
      const [row, col] = rowCol(n, index);
      if (!traced || row !== 0) return;
      const picks = def.selected!(ctx, index);
      const input = picks.length === 1 ? n.inputs[picks[0].arg] : undefined;
      const from = input === undefined ? undefined : this.g.entryOf.get(input);
      const at = from === undefined || from.track !== track || from.cols !== rowsCols(from.node)[1] ? undefined : sourceIndex(input!, from.node, picks[0].index);
      if (at === undefined) traced = false;
      else offsets[col] = from!.offsets[rowCol(from!.node, at)[1]];
    });
    if (!traced || offsets.some((o) => o === undefined) || new Set(offsets).size !== offsets.length) return undefined;
    return offsets as number[];
  }

  private placeBoardGroup(board: BoardRef, level: Level): void {
    const first = this.g.stacked.get(board.op);
    if (first !== undefined) {
      const below = this.g.entryOf.get(first)!;
      this.place(below);
      const out = this.g.entryOf.get(board.op)!;
      this.join(out, below.track!, below.offsets);
      out.zShift = below.zShift + below.rows * LEVEL_ROW + STACK_GAP;
      return;
    }
    const a = this.drawnA(board);
    this.place(a);
    const out = this.g.entryOf.get(board.op)!;
    const tile = this.g.tiles.get(board.op)!;
    const same = a.level === level;
    const aTrack = a.track ?? this.hostTrack(a) ?? this.stream;
    if (board.batched && !same) {
      const track = this.track(aTrack.spacing, aTrack, 'right');
      this.join(out, track, consecutive(0, out.cols));
      this.join(tile, track, consecutive(0, tile.cols));
      this.done.add(out);
      this.done.add(tile);
      return;
    }
    const group = board.batched
      ? [board]
      : [...this.g.boards.values()].filter((b) => !b.batched && !this.g.stacked.has(b.op) && this.g.entryOf.get(b.op)?.level === level && !this.g.copies.has(this.g.entryOf.get(b.op)!) && this.drawnA(b) === a).sort((p, q) => p.op.id - q.op.id);
    if (same) {
      const total = group.reduce((sum, b) => sum + this.g.entryOf.get(b.op)!.cols, 0);
      const track = this.track(board.batched ? aTrack.spacing : laneSpacing(total), aTrack, 'right', a);
      let offset = 0;
      group.forEach((b, i) => {
        const o = this.g.entryOf.get(b.op)!;
        const t = this.g.tiles.get(b.op)!;
        if (i > 0) track.breaks.set(offset, COLUMN_BREAK);
        this.join(o, track, consecutive(offset, o.cols));
        this.join(t, track, consecutive(offset, t.cols));
        this.done.add(o);
        this.done.add(t);
        offset += o.cols;
      });
      return;
    }
    for (const b of group) {
      const o = this.g.entryOf.get(b.op)!;
      const t = this.g.tiles.get(b.op)!;
      this.done.add(o);
      this.done.add(t);
      const start = a.track === undefined ? 0 : (a.offsets[boardStart(b.op)] ?? a.offsets[0] ?? 0);
      const offsets = consecutive(start, o.cols);
      if (a.track !== undefined && this.free(level, a.track, offsets)) {
        this.join(o, a.track, offsets);
        this.join(t, a.track, offsets);
      } else {
        const track = this.track(aTrack.spacing, aTrack, 'right');
        this.join(o, track, consecutive(0, o.cols));
        this.join(t, track, consecutive(0, t.cols));
      }
    }
    if (out.track === undefined) {
      const track = this.track(aTrack.spacing, aTrack, 'right');
      this.join(out, track, consecutive(0, out.cols));
      this.join(tile, track, consecutive(0, tile.cols));
    }
  }

  private relativeXs(): void {
    for (const track of this.tracks) track.x0 = 0;
    for (const e of this.g.entries) e.xs = undefined;
    const seen = new Map<Entry, Set<number>>();
    const levelCuts = new Map<string, Set<number>>();
    for (const e of this.g.entries) {
      if (e.track === undefined) continue;
      const cuts = levelCuts.get(this.levelKey(e)) ?? new Set<number>();
      for (const c of this.laneCuts(e, seen)) cuts.add(c);
      levelCuts.set(this.levelKey(e), cuts);
    }
    for (const e of this.g.entries) {
      if (e.xs !== undefined || e.track === undefined) continue;
      if (this.g.slabbed.has(e.node)) {
        e.xs = slabXs(e);
        continue;
      }
      const cuts = [...(levelCuts.get(this.levelKey(e)) ?? [])];
      const aligned = e.node.inputs.map((i) => this.g.entryOf.get(i) ?? this.g.entryOf.get(this.g.source(i))).filter((f) => f !== undefined && f !== e && f.track === e.track && f.xs !== undefined && !this.g.slabbed.has(f.node));
      const base = (o: number) => position(e.track!, o) + cuts.filter((c) => c <= o).length * COLUMN_BREAK + this.widening(e, o);
      const inherited = new Map<number, number>();
      for (const o of e.offsets) {
        const f = aligned.find((a) => a!.offsets.includes(o));
        if (f === undefined) continue;
        const own = this.laneCuts(f, seen);
        const extra = [...this.laneCuts(e, seen)].filter((c) => c > Math.min(...e.offsets) && c <= o && !own.has(c)).length;
        inherited.set(o, f.xs![f.offsets.indexOf(o)] + extra * COLUMN_BREAK + (f.level === e.level ? 0 : this.widening(e, o)));
      }
      const covered = [...inherited.keys()].sort((p, q) => p - q);
      e.xs = e.offsets.map((o) => {
        const known = inherited.get(o);
        if (known !== undefined) return known;
        const near = covered.filter((c) => c < o).pop() ?? covered[0];
        return base(o) + (near === undefined ? 0 : inherited.get(near)! - base(near));
      });
    }
    for (const [op, tile] of this.g.kernelTiles) tile.xs = this.g.entryOf.get(op)?.xs ?? tile.xs;
    const resolving = new Set<Entry>();
    const resolve = (e: Entry): number[] => {
      if (e.xs !== undefined) return e.xs;
      if (resolving.has(e)) return [0];
      resolving.add(e);
      const host = e.host ?? this.g.entryOf.get(e.node.inputs[0]) ?? e;
      if (host !== e) resolve(host);
      const nudge = this.nudges.get(e) ?? [];
      e.xs = this.sideXs(e, host).map((x, i) => x + (nudge[i] ?? 0));
      return e.xs;
    };
    for (const e of this.attached) resolve(e);
    for (const e of this.g.entries) if (e.xs === undefined && !this.g.copies.has(e)) resolve(e);
    for (const [e, base] of this.g.copies) e.xs = base.xs;
  }

  private ownTrack(e: Entry): Track | undefined {
    return e.track ?? this.hostTrack(e.host);
  }

  private pieces(e: Entry): Piece[] {
    const xs = e.xs ?? [0];
    const out: Piece[] = [];
    if (e.track === undefined) xs.forEach((x, index) => out.push({ entry: e, index, lo: x - BLOCK_HALF, hi: x + BLOCK_HALF }));
    else {
      const cols = e.offsets.map((o, i) => ({ o, x: xs[i] })).sort((p, q) => p.o - q.o);
      for (let i = 0; i < cols.length; i++) {
        const last = out[out.length - 1];
        if (last !== undefined && last.offset! + last.count! === cols[i].o) {
          last.hi = cols[i].x + BLOCK_HALF;
          last.count!++;
        } else out.push({ entry: e, index: out.length, lo: cols[i].x - BLOCK_HALF, hi: cols[i].x + BLOCK_HALF, offset: cols[i].o, count: 1 });
      }
    }
    const width = texWidth(e.node.tex);
    const first = out[0];
    const last = out[out.length - 1];
    if (e.title === 'left') first.lo = Math.min(first.lo, first.lo + BLOCK_HALF + TITLE.left[0] - width);
    if (e.title === 'right' && (e.kind === 'disc' || e.kind === 'box')) first.hi = Math.max(first.hi, first.lo + BLOCK_HALF + TITLE.node[0] + width);
    else if (e.title === 'right') last.hi = Math.max(last.hi, last.hi - BLOCK_HALF + (e.kind === 'vector' ? TITLE.vector[0] : TITLE.right[0]) + width);
    if (e.title === 'below') {
      const mid = (first.lo + last.hi) / 2;
      first.lo = Math.min(first.lo, mid - width / 2);
      last.hi = Math.max(last.hi, mid + width / 2);
    }
    if (e.kind === 'plate' && this.g.labelled(e.node)) last.hi = Math.max(last.hi, last.hi - BLOCK_HALF + CHIP_LABEL_SIDE + CHIP_ROOM);
    return out;
  }

  private extent(e: Entry): Span {
    const pieces = this.pieces(e);
    return [Math.min(...pieces.map((p) => p.lo)), Math.max(...pieces.map((p) => p.hi))];
  }

  private crowded(): boolean {
    const rides = (e: Entry) => e.kind === 'tile' || e.kind === 'vector' || this.g.copies.has(e) || this.g.stacked.has(e.node);
    for (const track of this.tracks) {
      const byLevel = new Map<Level, Entry[]>();
      for (const e of this.g.entries) {
        if (this.ownTrack(e) !== track || rides(e)) continue;
        byLevel.set(e.level, [...(byLevel.get(e.level) ?? []), e]);
      }
      for (const entries of byLevel.values()) {
        const sorted = entries.flatMap((e) => this.pieces(e)).sort((p, q) => p.lo - q.lo || p.entry.node.id - q.entry.node.id);
        for (let i = 1; i < sorted.length; i++) {
          const q = sorted[i];
          const need = Math.max(...sorted.slice(0, i).filter((p) => p.entry !== q.entry).map((p) => p.hi)) - q.lo;
          if (need <= 1e-9) continue;
          if (q.offset !== undefined) {
            const key = this.levelKey(q.entry);
            const widths = this.widened.get(key) ?? new Map<number, number>();
            widths.set(q.offset, (widths.get(q.offset) ?? 0) + need);
            this.widened.set(key, widths);
          }
          else {
            const nudge = this.nudges.get(q.entry) ?? [];
            nudge[q.index] = (nudge[q.index] ?? 0) + need;
            this.nudges.set(q.entry, nudge);
          }
          return true;
        }
      }
    }
    return false;
  }

  private widenBreaks(): void {
    for (let round = 0; round < this.g.entries.length && this.crowded(); round++) this.relativeXs();
  }

  private occupancy(track: Track): { levels: Map<number, Span>; columns: Span } {
    const members = new Map<number, Span>();
    const add = (l: number, span: Span) => {
      const known = members.get(l);
      members.set(l, known === undefined ? span : [Math.min(known[0], span[0]), Math.max(known[1], span[1])]);
    };
    const xs: number[] = [];
    for (const e of this.g.entries) {
      if (this.ownTrack(e) !== track || this.g.copies.has(e)) continue;
      add(this.levelIndex(e), this.extent(e));
      if (e.track === track) xs.push(...(e.xs ?? []));
    }
    const columns: Span = xs.length === 0 ? [0, 0] : [Math.min(...xs) - BLOCK_HALF, Math.max(...xs) + BLOCK_HALF];
    if (this.g.options.span === 'members') return { levels: members, columns };
    const indices = [...members.keys()];
    const levels = new Map<number, Span>();
    for (let l = Math.min(...indices); l <= Math.max(...indices); l++) {
      const own = members.get(l);
      levels.set(l, own === undefined ? columns : [Math.min(own[0], columns[0]), Math.max(own[1], columns[1])]);
    }
    return { levels, columns };
  }

  private pack(): void {
    const occupancy = new Map(this.tracks.map((t) => [t, this.occupancy(t)]));
    const placed = new Set<Track>();
    const relOf = (t: Track): number => (t.bound === undefined || t.anchor === undefined ? 0 : Math.max(...(t.bound.xs ?? [0])) + BOARD_GAP + relOf(t.anchor));
    const unit = (root: Track): Track[] => [root, ...this.tracks.filter((t) => t.bound !== undefined && t.anchor === root).flatMap(unit)];
    const conflict = (members: Track[], x0: number): { lo: number; hi: number; at: number; rel: number } | undefined => {
      for (const t of members) {
        const rel = relOf(t) - relOf(members[0]);
        for (const [l, span] of occupancy.get(t)!.levels) {
          for (const other of placed) {
            const theirs = occupancy.get(other)!.levels.get(l);
            if (theirs === undefined) continue;
            const mine: Span = [span[0] + x0 + rel, span[1] + x0 + rel];
            const abs: Span = [theirs[0] + other.x0, theirs[1] + other.x0];
            if (overlaps(mine, abs, this.gap)) return { lo: abs[0], hi: abs[1], at: span[members[0].side === 'left' ? 1 : 0], rel };
          }
        }
      }
      return undefined;
    };
    const settle = (root: Track, x0: number) => {
      for (const t of unit(root)) {
        t.x0 = x0 + relOf(t) - relOf(root);
        placed.add(t);
      }
    };
    this.stream.x0 = STREAM_X;
    settle(this.stream, STREAM_X);
    const pending = this.tracks.filter((t) => t !== this.stream && t.bound === undefined).sort((p, q) => Math.min(...occupancy.get(p)!.levels.keys()) - Math.min(...occupancy.get(q)!.levels.keys()) || p.id - q.id);
    while (pending.length > 0) {
      const index = pending.findIndex((t) => t.anchor === undefined || placed.has(t.anchor));
      const root = pending.splice(Math.max(0, index), 1)[0];
      const anchor = root.anchor !== undefined && placed.has(root.anchor) ? root.anchor : this.stream;
      const members = unit(root);
      const columns = occupancy.get(anchor)!.columns;
      if (root.side === 'right') {
        let x0 = anchor.x0 + columns[0] - occupancy.get(root)!.columns[0];
        for (let hit = conflict(members, x0); hit !== undefined; hit = conflict(members, x0)) x0 = hit.hi + this.gap - hit.at - hit.rel;
        settle(root, x0);
      } else {
        let x0 = anchor.x0 + columns[1] - occupancy.get(root)!.columns[1];
        for (let hit = conflict(members, x0); hit !== undefined; hit = conflict(members, x0)) x0 = hit.lo - this.gap - hit.at - hit.rel;
        settle(root, x0);
      }
    }
  }

  private absoluteXs(): void {
    for (const e of this.g.entries) {
      if (this.g.copies.has(e)) continue;
      const track = this.ownTrack(e);
      if (track !== undefined && e.xs !== undefined) e.xs = e.xs.map((x) => x + track.x0);
    }
    for (const [e, base] of this.g.copies) e.xs = base.xs;
  }

  private sideXs(e: Entry, main: Entry): number[] {
    const n = e.node;
    const mainXs = main.xs ?? (main.track === undefined ? [0] : main.offsets.map((o) => position(main.track!, o)));
    if (n.kind === 'op' && n.op === 'select' && main.node === n.inputs[0] && main.node.shape.length === 2 && e.cols === 1 && main.cols === mainXs.length) {
      const attrs = n.attrs as { dim: number; index: number };
      if (attrs.dim === 1 || attrs.dim === -1) return [mainXs[(attrs.index + main.cols) % main.cols]];
    }
    if (main.kind === 'disc' || main.kind === 'box' || (main.cols === 1 && main !== e)) {
      if (e.kind === 'box') return [mainXs[0]];
      return mainXs.slice(0, e.cols);
    }
    if (e.cols > main.cols && main.track !== undefined) return consecutive(main.offsets[0] ?? 0, e.cols).map((o) => position(main.track!, o) + DISC_GROUP_SIDE);
    if (e.cols > 1) {
      const size = main.cols / e.cols;
      return Array.from({ length: e.cols }, (_, g) => mainXs[Math.min(mainXs.length - 1, Math.ceil((g + 1) * size) - 1)] + DISC_GROUP_SIDE);
    }
    const group = main.track === undefined ? 0 : this.groupSize(main.track);
    const side = group > 0 && main.cols < group ? DISC_NEAR_SIDE : DISC_SIDE;
    return [Math.max(...mainXs) + side];
  }

  private groupSize(track: Track): number {
    let size = 0;
    for (const m of track.members) {
      const own = this.entryGroup(m);
      if (own > 0 && (size === 0 || own < size)) size = own;
    }
    return size;
  }

  private entryGroup(m: Entry): number {
    let size = 0;
    for (const node of [m.node, ...(this.g.drawnOn.get(m.node) ?? [])]) {
      const { count, width } = plates(node);
      if (node.shape.length >= 3 && count > 1 && width > 1 && (size === 0 || width < size)) size = width;
      for (const u of this.g.computed) {
        if (u.kind !== 'op' || !u.inputs.includes(node)) continue;
        const attrs = u.attrs as { size?: number; length?: number; dim?: number };
        const s = typeof attrs.size === 'number' ? attrs.size : u.op === 'narrow' && attrs.dim === node.shape.length - 1 ? attrs.length : undefined;
        if (s !== undefined && s > 0 && s < m.cols && (size === 0 || s < size)) size = s;
      }
    }
    return size;
  }

  private laneCuts(e: Entry, seen: Map<Entry, Set<number>>): Set<number> {
    const known = seen.get(e);
    if (known !== undefined) return known;
    const cuts = new Set(this.columnCuts(e).filter((k) => e.offsets[k] > e.offsets[k - 1]).map((k) => e.offsets[k]));
    const start = Math.min(...e.offsets);
    if (e.level.entries.some((o) => o !== e && o.track === e.track && o.offsets.includes(start - 1))) cuts.add(start);
    seen.set(e, cuts);
    const n = e.node;
    const board = this.g.boards.get(n);
    const from = board !== undefined ? [this.drawnA(board)] : n.inputs.map((i) => this.g.entryOf.get(this.g.source(i)));
    const hi = Math.max(...e.offsets);
    for (const f of from) {
      if (f === undefined || f === e || f.track !== e.track) continue;
      for (const c of this.laneCuts(f, seen)) {
        if (f.cols !== e.cols || f.offsets.join() === e.offsets.join()) {
          if (c <= hi) cuts.add(c);
          continue;
        }
        const k = f.offsets.indexOf(c);
        if (k > 0) cuts.add(e.offsets[k]);
      }
    }
    return cuts;
  }

  private columnCuts(e: Entry): number[] {
    const cuts = new Set<number>();
    const group = this.entryGroup(e);
    if (group > 0) for (let k = group; k < e.cols; k += group) cuts.add(k);
    const n = e.node;
    if (n.kind === 'op' && n.op === 'cat' && n.shape.length > 0) {
      const attrs = n.attrs as { dim: number; pieces: number | null };
      const last = n.shape.length - 1;
      if ((attrs.dim + n.shape.length) % n.shape.length === last) {
        const widths = attrs.pieces === null ? n.inputs.map((i) => i.shape[last]) : Array.from({ length: attrs.pieces }, () => n.inputs[0].shape[last]);
        let at = 0;
        for (const w of widths.slice(0, -1)) cuts.add((at += w));
      }
    }
    return [...cuts].filter((k) => k > 0 && k < e.cols);
  }

  private assignTitles(): void {
    for (const e of this.g.entries) e.title = 'none';
    for (const level of this.g.levels) {
      for (const e of level.entries) {
        if (e.kind === 'disc' || e.kind === 'box') {
          e.title = 'right';
          continue;
        }
        if (e.kind === 'vector') continue;
        if (e.kind === 'tile') {
          if (e.table !== undefined || !level.entries.some((o) => o.kind === 'tile' && o !== e && o.track === e.track && (o.xs?.[0] ?? 0) < (e.xs?.[0] ?? 0))) e.title = 'left';
          continue;
        }
        if (e.cols === 1) {
          e.title = 'right';
          continue;
        }
        const board = this.g.boards.get(e.node);
        if (board !== undefined && this.drawnA(board).level === level) {
          e.title = 'below';
          continue;
        }
        e.title = 'left';
      }
      const vectors = level.entries.filter((e) => e.kind === 'vector').sort((p, q) => p.xs![0] - q.xs![0]);
      if (vectors.length > 0) vectors[vectors.length - 1].title = 'right';
    }
  }
}
