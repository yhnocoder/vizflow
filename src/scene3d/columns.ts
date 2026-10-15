import type { Node } from '../autograd';
import { explainOf } from '../explain';
import { isLeafLike } from '../steps';
import { forEachIndex } from '../tensor';
import { plates, rowCol, rowsCols, sourceIndex } from './layout';
import type { LevelScheme } from './schemes';

export const STREAM_X = -7.2;
export const STREAM_SPACING = 0.95;
export const STREAM_BESIDE = 3;
export const STREAM_PARALLEL = 6;
export const BOARD_GAP = 1.45;
export const COLUMN_BREAK = 0.6;
export const TITLE_ROOM = 3.5;
export const SPACING_MAX = 1.45;
export const SPACING_MIN = 0.74;
export const SPACING_SPAN = 11.4;
export const DISC_SIDE = 1.6;
export const DISC_NEAR_SIDE = 0.62;
export const DISC_GROUP_SIDE = 0.67;
export const PLACE_GAP = 1.6;
export const PLACE_TITLE_ROOM = 1.2;
export const LEVEL_ROW = 1.0;
export const STACK_GAP = 0.9;

export type Kind = 'plate' | 'tile' | 'vector' | 'disc' | 'box';
export type TitleKind = 'left' | 'below' | 'right' | 'none';

export interface Lane {
  x0: number;
  spacing: number;
  stream: boolean;
  breaks: Set<number>;
  members: Entry[];
}

export interface Entry {
  key: string;
  node: Node;
  kind: Kind;
  level: Level;
  rows: number;
  cols: number;
  lane: Lane | undefined;
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
  scheme: LevelScheme;
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
}

export function laneSpacing(cols: number): number {
  return Math.min(SPACING_MAX, Math.max(SPACING_MIN, SPACING_SPAN / cols));
}

function consecutive(start: number, n: number): number[] {
  return Array.from({ length: n }, (_, i) => start + i);
}

export function position(lane: Lane, i: number): number {
  let breaks = 0;
  for (const b of lane.breaks) if (b <= i) breaks++;
  return lane.x0 + i * lane.spacing + breaks * COLUMN_BREAK;
}

function slabXs(e: Entry): number[] {
  const x0 = position(e.lane!, e.offsets[0]);
  return e.offsets.map((_, i) => x0 + i * e.lane!.spacing);
}

function boardStart(op: Node): number {
  const head = (op.attrs as { index?: number }).index;
  return head === undefined ? 0 : head * op.inputs[0].shape[2];
}

function rank(e: Entry): number {
  if (e.kind === 'tile' && e.table !== undefined) return 1;
  if (e.kind === 'vector' || isLeafLike(e.node)) return 4;
  return 2;
}

export class Columns {
  readonly lanes: Lane[] = [];
  readonly stream: Lane;

  constructor(private readonly g: ColumnGraph) {
    this.stream = this.lane(STREAM_X, STREAM_SPACING, true);
  }

  run(): void {
    this.assignColumns();
    this.assignTitles();
    this.makeTitleRoom();
    this.placeApart();
    this.assignTitles();
  }

  private lane(x0: number, spacing: number, stream = false): Lane {
    const lane: Lane = { x0, spacing, stream, breaks: new Set(), members: [] };
    this.lanes.push(lane);
    return lane;
  }

  private occupied(level: Level, lane: Lane, skip: Entry): Set<number> {
    const out = new Set<number>();
    const side = this.mergeSide(skip.node);
    const apart = (e: Entry) => {
      const other = this.mergeSide(e.node);
      return side !== undefined && other !== undefined && side !== other;
    };
    for (const e of level.entries) if (e !== skip && e.lane === lane && !this.g.copies.has(e) && e.kind !== 'tile' && e.kind !== 'vector' && !apart(e)) e.offsets.forEach((o) => out.add(o));
    return out;
  }

  private mergeSide(n: Node): number | undefined {
    const own = this.g.merges.findIndex((m) => m.left.includes(n) || m.right.includes(n));
    if (own >= 0) return 2 * own + (this.g.merges[own].left.includes(n) ? 0 : 1);
    if (!isLeafLike(n)) return undefined;
    const sides = new Set(this.g.consumers(n).map((u) => this.mergeSide(u)));
    return sides.size === 1 ? [...sides][0] : undefined;
  }

  private order(e: Entry): number {
    const fed = this.g.computed.has(e.node) && e.node.inputs.some((i) => this.g.computed.has(this.g.source(i)) || this.g.sourceOf.has(i));
    return rank(e) + (this.g.computed.has(e.node) && !fed && e.kind === 'plate' ? 1 : 0);
  }

  private vectorHost(e: Entry): Entry | undefined {
    const n = e.node;
    const user = this.g.consumers(n)[0];
    const target = this.g.entryOf.get(user);
    const placed = this.g.sides.find((side) => side.node === n);
    return placed !== undefined ? this.g.entryOf.get(placed.of) : target?.level === e.level ? target : this.g.entryOf.get(user.inputs.find((i) => this.g.source(i) !== n) ?? user);
  }

  private laneAnchor(e: Entry): Entry | undefined {
    const seen = new Set<Entry>();
    let at: Entry | undefined = e;
    while (at !== undefined && at.lane === undefined && !seen.has(at)) {
      seen.add(at);
      const input: Node | undefined = at.node.inputs[0];
      at = input === undefined ? undefined : (this.g.entryOf.get(input) ?? this.g.entryOf.get(this.g.source(input)));
    }
    return at?.lane === undefined ? undefined : at;
  }

  private setLane(e: Entry, lane: Lane, offsets: number[]): void {
    e.lane = lane;
    e.offsets = offsets;
    lane.members.push(e);
  }

  private drawnA(board: BoardRef): Entry {
    return this.g.entryOf.get(board.a)!;
  }

  private assignColumns(): void {
    const done = new Set<Entry>();
    const beside: [Entry, Entry][] = [];
    for (const level of this.g.levels) {
      const ordered = [...level.entries].sort((p, q) => this.order(p) - this.order(q));
      const hosted = (e: Entry) => e.kind === 'vector' && this.vectorHost(e)?.lane === undefined;
      for (const e of [...ordered.filter((e) => !hosted(e)), ...ordered.filter(hosted)]) {
        if (done.has(e) || this.g.copies.has(e)) continue;
        done.add(e);
        const n = e.node;
        if (e.kind === 'tile' && e.table === undefined) continue;
        const table = e.table ?? (this.g.tables.has(n) ? n : undefined);
        if (table !== undefined) {
          this.setLane(e, this.stream, consecutive(0, e.cols));
          continue;
        }
        const board = this.g.boards.get(n);
        if (board !== undefined) {
          this.placeBoardGroup(board, level, done);
          continue;
        }
        if (e.kind === 'vector') {
          const host = this.vectorHost(e)!;
          if (host.lane !== undefined) this.setLane(e, host.lane, host.offsets.length >= e.cols ? host.offsets.slice(0, e.cols) : consecutive(host.offsets[0] ?? 0, e.cols));
          continue;
        }
        if (isLeafLike(n)) {
          const partners = new Set(this.g.consumers(n).flatMap((u) => u.inputs).filter((i) => i !== n).map((i) => this.g.entryOf.get(this.g.sourceOf.get(i) ?? i)));
          const taken = this.occupied(level, this.stream, e);
          const blocked = level.entries.some((o) => partners.has(o) && o.lane === this.stream && consecutive(0, e.cols).some((k) => o.offsets.includes(k)));
          const start = blocked ? Math.max(...taken) + this.besideOffset() : 0;
          this.setLane(e, this.stream, consecutive(start, e.cols));
          continue;
        }
        const inputs = n.inputs.map((i) => this.g.entryOf.get(i)).filter((i): i is Entry => i !== undefined);
        const main = inputs[0];
        if (main === undefined) {
          const taken = this.occupied(level, this.stream, e);
          this.setLane(e, this.stream, consecutive(e.kind === 'plate' && consecutive(0, e.cols).some((k) => taken.has(k)) ? Math.max(...taken) + this.besideOffset() : 0, e.cols));
          continue;
        }
        if (e.kind === 'disc' || e.kind === 'box' || (e.cols === 1 && main.lane === undefined)) {
          beside.push([e, main]);
          continue;
        }
        if (main.lane === undefined) {
          const anchor = this.laneAnchor(main);
          const lane = anchor?.lane ?? this.stream;
          const taken = this.occupied(level, lane, e);
          const wanted = consecutive(anchor?.offsets[0] ?? 0, e.cols);
          this.setLane(e, lane, wanted.some((k) => taken.has(k)) ? consecutive(Math.max(...taken) + this.besideOffset(), e.cols) : wanted);
          continue;
        }
        const traced = this.tracedOffsets(e, main.lane);
        if (traced !== undefined && !traced.some((o) => this.occupied(level, main.lane!, e).has(o))) {
          this.setLane(e, main.lane, traced);
          continue;
        }
        const joined = inputs.flatMap((i) => i.offsets);
        if (n.kind === 'op' && n.op === 'cat' && inputs.every((i) => i.lane === main.lane) && new Set(joined).size === joined.length) {
          this.setLane(e, main.lane, joined);
          continue;
        }
        const start = n.kind === 'op' && n.op === 'narrow' ? ((n.attrs as { start: number }).start ?? 0) : n.kind === 'op' && n.op === 'select' && n.inputs[0].shape.length === 3 && (n.attrs as { dim: number }).dim === 1 ? (n.attrs as { index: number }).index * n.inputs[0].shape[2] : 0;
        const stride = n.kind === 'op' && n.op === 'narrow' ? ((n.attrs as { step?: number }).step ?? 1) : 1;
        let offsets = main.offsets.length >= start + (e.cols - 1) * stride + 1 ? Array.from({ length: e.cols }, (_, i) => main.offsets[start + i * stride]) : consecutive(main.offsets[0] ?? 0, e.cols);
        const taken = this.occupied(level, main.lane, e);
        if (offsets.some((o) => taken.has(o))) offsets = consecutive(Math.max(...taken) + 1, e.cols);
        this.setLane(e, main.lane, offsets);
      }
    }
    for (const [op, tile] of this.g.kernelTiles) {
      const out = this.g.entryOf.get(op);
      if (out?.lane !== undefined) this.setLane(tile, out.lane, out.offsets);
    }
    for (const [e, base] of this.g.copies) if (base.lane !== undefined) this.setLane(e, base.lane, base.offsets);
    const seen = new Map<Entry, Set<number>>();
    const levelCuts = new Map<string, Set<number>>();
    const keyOf = (e: Entry) => `${this.g.levels.indexOf(e.level)}:${this.lanes.indexOf(e.lane!)}`;
    for (const e of this.g.entries) {
      if (e.lane === undefined) continue;
      const cuts = levelCuts.get(keyOf(e)) ?? new Set<number>();
      for (const c of this.laneCuts(e, seen)) cuts.add(c);
      levelCuts.set(keyOf(e), cuts);
    }
    for (const e of this.g.entries) {
      if (e.xs !== undefined || e.lane === undefined) continue;
      if (this.g.slabbed.has(e.node)) {
        e.xs = slabXs(e);
        continue;
      }
      const cuts = [...(levelCuts.get(keyOf(e)) ?? [])];
      const aligned = e.node.inputs.map((i) => this.g.entryOf.get(i) ?? this.g.entryOf.get(this.g.source(i))).filter((f) => f !== undefined && f !== e && f.lane === e.lane && f.xs !== undefined && !this.g.slabbed.has(f.node));
      e.xs = e.offsets.map((o) => {
        const f = aligned.find((a) => a!.offsets.includes(o));
        if (f === undefined) return position(e.lane!, o) + cuts.filter((c) => c <= o).length * COLUMN_BREAK;
        const inherited = this.laneCuts(f, seen);
        const extra = [...this.laneCuts(e, seen)].filter((c) => c > Math.min(...e.offsets) && c <= o && !inherited.has(c)).length;
        return f.xs![f.offsets.indexOf(o)] + extra * COLUMN_BREAK;
      });
    }
    for (const [op, tile] of this.g.kernelTiles) tile.xs = this.g.entryOf.get(op)?.xs ?? tile.xs;
    for (const [e, main] of beside) e.xs = this.sideXs(e, main);
    for (const e of this.g.entries) if (e.xs === undefined && !this.g.copies.has(e)) e.xs = this.sideXs(e, this.g.entryOf.get(e.node.inputs[0]) ?? e);
    for (const [e, base] of this.g.copies) e.xs = base.xs;
  }

  private makeTitleRoom(): void {
    const crowded = (): { at: number; need: number } | undefined => {
      let found: { at: number; need: number } | undefined;
      for (const level of this.g.levels) {
        const group = level.entries.filter((e) => e.xs !== undefined && e.title === 'left' && e.kind !== 'tile' && e.kind !== 'disc' && e.kind !== 'box');
        group.sort((p, q) => Math.min(...p.xs!) - Math.min(...q.xs!));
        group.forEach((e, i) => {
          if (i === 0) return;
          const at = Math.min(...e.xs!);
          const gap = at - Math.max(...group[i - 1].xs!);
          if (gap > 0 && gap < TITLE_ROOM - 1e-9 && (found === undefined || at < found.at)) found = { at, need: TITLE_ROOM - gap };
        });
      }
      return found;
    };
    for (let cut = crowded(); cut !== undefined; cut = crowded()) {
      const { at, need } = cut;
      for (const e of this.g.entries) if (e.xs !== undefined && Math.min(...e.xs) >= at - 1e-9) e.xs = e.xs.map((x) => x + need);
    }
  }

  private overlapping(e: Entry, others: Entry[]): Entry[] {
    const xs = e.xs ?? [];
    return others.filter((o) => (o.xs ?? []).some((x) => xs.some((own) => Math.abs(own - x) < COLUMN_BREAK)));
  }

  private placeApart(): void {
    const span = (entries: Entry[]) => {
      const xs = entries.flatMap((e) => e.xs ?? []);
      return xs.length === 0 ? undefined : [Math.min(...xs), Math.max(...xs)];
    };
    const shift = (entries: Entry[], dx: number) => entries.forEach((e) => (e.xs = e.xs?.map((x) => x + dx)));
    const facing = (leftOnes: Entry[], rightOnes: Entry[]) =>
      PLACE_GAP + (leftOnes.some((e) => e.title === 'right' || e.kind === 'vector') ? PLACE_TITLE_ROOM : 0) + (rightOnes.some((e) => e.title === 'left' || e.kind === 'tile') ? PLACE_TITLE_ROOM : 0);
    for (const { node, side, of } of this.g.sides) {
      const e = this.g.entryOf.get(node);
      const target = this.g.entryOf.get(of);
      if (e === undefined || target === undefined) continue;
      const others = e.level.entries.filter((o) => o !== e && o.node !== node);
      const [tlo, thi] = span([target]) ?? [0, 0];
      if (side === 'left') {
        const [, hi] = span([e]) ?? [0, 0];
        shift([e], tlo - facing([e], [target]) - hi);
      } else if (side === 'right') {
        const [lo] = span([e]) ?? [0, 0];
        shift([e], thi + facing([target], [e]) - lo);
      }
      for (let blocking = this.overlapping(e, others); blocking.length > 0; blocking = this.overlapping(e, others)) {
        const [lo, hi] = span([e]) ?? [0, 0];
        const [blo, bhi] = span(blocking) ?? [0, 0];
        shift([e], side === 'left' ? blo - facing([e], blocking) - hi : bhi + facing(blocking, [e]) - lo);
      }
    }
    this.g.merges.forEach((_, i) => {
      const leftEntries = this.g.entries.filter((e) => this.mergeSide(e.node) === 2 * i);
      const rightEntries = this.g.entries.filter((e) => this.mergeSide(e.node) === 2 * i + 1);
      let need = 0;
      for (const level of new Set(rightEntries.map((e) => e.level))) {
        const l = span(leftEntries.filter((e) => e.level === level));
        const r = span(rightEntries.filter((e) => e.level === level));
        if (l !== undefined && r !== undefined) need = Math.max(need, l[1] + facing(leftEntries.filter((e) => e.level === level), rightEntries.filter((e) => e.level === level)) - r[0]);
      }
      if (need > 0) shift(rightEntries, need);
    });
    const companions = (e: Entry): Entry[] => [e, ...[this.g.tiles.get(e.node), this.g.kernelTiles.get(e.node)].filter((t): t is Entry => t !== undefined && t.lane === e.lane && !this.g.stacked.has(e.node))];
    for (const level of this.g.levels) {
      const plates = level.entries.filter((e) => (e.kind === 'plate' || e.kind === 'disc') && e.xs !== undefined && !this.g.stacked.has(e.node) && !this.g.copies.has(e));
      plates.forEach((e, i) => {
        const group = companions(e);
        const earlier = plates.slice(0, i).filter((o) => o.lane !== e.lane).flatMap(companions);
        const hit = () => [...new Set(group.flatMap((g) => this.overlapping(g, earlier)))];
        for (let blocking = hit(); blocking.length > 0; blocking = hit()) {
          const [lo] = span(group)!;
          const [, hi] = span(blocking)!;
          const riders = this.g.entries.filter((o) => o.level === level && ([...this.g.stacked].some(([op, base]) => op === o.node && group.some((g) => g.node === base)) || group.includes(this.g.copies.get(o)!)));
          shift([...group, ...riders], hi + facing(blocking, group) - lo);
        }
      });
    }
  }

  private tracedOffsets(e: Entry, lane: Lane): number[] | undefined {
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
      const at = from === undefined || from.lane !== lane || from.cols !== rowsCols(from.node)[1] ? undefined : sourceIndex(input!, from.node, picks[0].index);
      if (at === undefined) traced = false;
      else offsets[col] = from!.offsets[rowCol(from!.node, at)[1]];
    });
    if (!traced || offsets.some((o) => o === undefined) || new Set(offsets).size !== offsets.length) return undefined;
    return offsets as number[];
  }

  private placeBatchedBoard(board: BoardRef, level: Level, done: Set<Entry>): void {
    const a = this.drawnA(board);
    const out = this.g.entryOf.get(board.op)!;
    const tile = this.g.tiles.get(board.op)!;
    const aXs = a.xs ?? (a.lane === undefined ? [0] : a.offsets.map((o) => position(a.lane!, o)));
    const lane = this.lane(a.level === level ? Math.max(...aXs) + BOARD_GAP : Math.min(...aXs), a.lane?.spacing ?? STREAM_SPACING);
    this.setLane(out, lane, consecutive(0, out.cols));
    this.setLane(tile, lane, consecutive(0, tile.cols));
    done.add(out);
    done.add(tile);
  }

  private placeBoardGroup(board: BoardRef, level: Level, done: Set<Entry>): void {
    const first = this.g.stacked.get(board.op);
    if (first !== undefined) {
      const below = this.g.entryOf.get(first)!;
      if (below.lane === undefined) this.placeBoardGroup(this.g.boards.get(first)!, level, done);
      const out = this.g.entryOf.get(board.op)!;
      this.setLane(out, below.lane!, below.offsets);
      out.zShift = below.zShift + below.rows * LEVEL_ROW + STACK_GAP;
      done.add(out);
      return;
    }
    if (board.batched) {
      this.placeBatchedBoard(board, level, done);
      return;
    }
    const a = this.drawnA(board);
    const group = [...this.g.boards.values()].filter((b) => !b.batched && !this.g.stacked.has(b.op) && this.g.entryOf.get(b.op)?.level === level && !this.g.copies.has(this.g.entryOf.get(b.op)!) && this.drawnA(b) === a).sort((p, q) => p.op.id - q.op.id);
    const same = a.level === level;
    const total = group.reduce((sum, b) => sum + b.b.shape[1], 0);
    let lane: Lane;
    let offset: number;
    if (same && a.lane?.stream !== false) {
      const aX = a.lane === undefined ? (a.xs ?? [0]) : a.offsets.map((o) => position(a.lane!, o));
      lane = this.lane(Math.max(...aX) + BOARD_GAP, laneSpacing(total));
      offset = 0;
    } else if (same) {
      lane = a.lane!;
      offset = Math.max(...a.offsets) + 1;
      lane.breaks.add(offset);
    } else {
      lane = a.lane ?? this.stream;
      const start = boardStart(board.op);
      offset = a.offsets[start] ?? 0;
    }
    const used = new Set<number>();
    group.forEach((b, i) => {
      const n = b.b.shape[1];
      if (i > 0 && same) lane.breaks.add(offset);
      const out = this.g.entryOf.get(b.op)!;
      let offsets = Array.from({ length: n }, (_, k) => (same ? offset + k : (this.drawnA(b).offsets[boardStart(b.op)] ?? 0) + k));
      if (!same) {
        const taken = new Set([...used, ...this.occupied(level, lane, out)]);
        if (offsets.some((o) => taken.has(o))) {
          const start = Math.max(...taken) + 1;
          lane.breaks.add(start);
          offsets = consecutive(start, n);
        }
        offsets.forEach((o) => used.add(o));
      }
      const tile = this.g.tiles.get(b.op)!;
      this.setLane(out, lane, offsets);
      this.setLane(tile, lane, offsets);
      done.add(out);
      done.add(tile);
      if (same) offset += n;
    });
  }

  private besideOffset(): number {
    return this.g.scheme === 'lines' ? STREAM_BESIDE : STREAM_PARALLEL;
  }

  private sideXs(e: Entry, main: Entry): number[] {
    const n = e.node;
    const mainXs = main.xs ?? (main.lane === undefined ? [0] : main.offsets.map((o) => position(main.lane!, o)));
    if (n.kind === 'op' && n.op === 'select' && main.node === n.inputs[0] && main.node.shape.length === 2 && e.cols === 1 && main.cols === mainXs.length) {
      const attrs = n.attrs as { dim: number; index: number };
      if (attrs.dim === 1 || attrs.dim === -1) return [mainXs[(attrs.index + main.cols) % main.cols]];
    }
    if (main.kind === 'disc' || main.kind === 'box' || (main.cols === 1 && main !== e)) {
      if (e.kind === 'box') return [mainXs[0]];
      return mainXs.slice(0, e.cols);
    }
    if (e.cols > main.cols && main.lane !== undefined) return consecutive(main.offsets[0] ?? 0, e.cols).map((o) => position(main.lane!, o) + DISC_GROUP_SIDE);
    if (e.cols > 1) {
      const size = main.cols / e.cols;
      return Array.from({ length: e.cols }, (_, g) => mainXs[Math.min(mainXs.length - 1, Math.ceil((g + 1) * size) - 1)] + DISC_GROUP_SIDE);
    }
    const group = main.lane === undefined ? 0 : this.groupSize(main.lane);
    const side = group > 0 && main.cols < group ? DISC_NEAR_SIDE : DISC_SIDE;
    if (n.shape.length === 0) return [Math.max(...mainXs) + side];
    return [Math.max(...mainXs) + side];
  }

  private groupSize(lane: Lane): number {
    let size = 0;
    for (const m of lane.members) {
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
    if (e.level.entries.some((o) => o !== e && o.lane === e.lane && o.offsets.includes(start - 1))) cuts.add(start);
    seen.set(e, cuts);
    const n = e.node;
    const board = this.g.boards.get(n);
    const from = board !== undefined ? [this.drawnA(board)] : n.inputs.map((i) => this.g.entryOf.get(this.g.source(i)));
    const hi = Math.max(...e.offsets);
    for (const f of from) {
      if (f === undefined || f === e || f.lane !== e.lane) continue;
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
          e.title = 'left';
          continue;
        }
        if (e.kind === 'vector') continue;
        if (e.kind === 'tile') {
          if (e.table !== undefined || !level.entries.some((o) => o.kind === 'tile' && o !== e && o.lane === e.lane && (o.xs?.[0] ?? 0) < (e.xs?.[0] ?? 0))) e.title = 'left';
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
