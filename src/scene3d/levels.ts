import type { Node } from '../autograd';
import { batchedCells, explainOf } from '../explain';
import { type StepModule, type StepsSpec, isLeafLike } from '../steps';
import { forEachIndex, indexOf } from '../tensor';
import { type Anchor, type Block, type BoardGeometry, type Element, type Plate, type Slab, type Vec3, isLayoutOp, labelBehind, plateAxes, plates, rowCol, rowsCols, sourceIndex } from './layout';
import type { SceneModule } from './modules';

export const LEVEL_FRONT = 2.9;
export const LEVEL_ROW = 1.0;
export const TILE_BACK = 1.3;
export const TILE_ROW = 0.72;
export const TILE_ROW_DENSE = 0.55;
export const TILE_DENSE_ROWS = 5;
export const VECTOR_BACK = 1.35;
export const STREAM_X = -7.2;
export const STREAM_SPACING = 0.95;
export const BOARD_GAP = 1.45;
export const COLUMN_BREAK = 0.6;
export const SPACING_MAX = 1.45;
export const SPACING_MIN = 0.74;
export const SPACING_SPAN = 11.4;
export const DISC_SIDE = 1.6;
export const DISC_NEAR_SIDE = 0.62;
export const DISC_GROUP_SIDE = 0.67;
export const STREAM_FOOTPRINT = 0.6;
export const BAR_FOOTPRINT = 0.62;
export const BAR_CLEARANCE = 0.18;
export const TILE_WIDTH = 0.58;
export const TILE_WIDTH_NARROW = 0.54;
export const TILE_DEPTH = 0.4;
export const TILE_DEPTH_DENSE = 0.34;
export const PLATE_MARGIN = 0.32;
export const BOX_FOOTPRINT: [number, number] = [1.1, 0.8];
export const DISC_FOOTPRINT: [number, number] = [0.54, 0.54];

export const GAP = {
  scalar: 1.2,
  scalarAfterBoard: 3.2,
  loss: 2.0,
  linearBoard: 3.5,
  ownBoard: 2.4,
  board: 2.7,
  copy: 5.3,
  copyAfterBoard: 3.2,
  copyPerTileRow: 0.3,
  afterBoard: 4.2,
  moduleAfterBoard: 4.6,
  afterCopy: 2.6,
  afterScalar: 1.3,
  afterBoardScalar: 2.0,
  vectorAfterScalar: 1.1,
  afterVector: 3.0,
  plate: 2.3,
} as const;

export const FOLD = {
  board: 2.2,
  buffer: 1.8,
  other: 1.4,
  first: 0.6,
  startAfterBoard: 2.4,
  start: 0.5,
  endBoard: 1.4,
  end: 0.8,
  minHeight: 1.3,
  afterBoard: 2.2,
  after: 0.75,
  emptyBelow: 3.5,
  emptyAbove: 0.8,
} as const;

export const BOX_MARGIN = { left: 0.6, right: 0.6, back: 0.55, front: 0.6 } as const;
export const TOP_ROOM = 1.5;

export const BRACE_PAD = { below: 0.6, belowBoard: 0.9, above: 1.0, aboveBoard: 1.4, aboveTable: 1.6 } as const;
export const BRACE_X = { left: 1.7, right: 2.2, outer: 4.0 } as const;

const TITLE = {
  left: [-0.55, 0.3] as const,
  board: [-0.55, -0.45] as const,
  below: [-0.55, 0.9] as const,
  right: [0.6, 0.35] as const,
  vector: [0.55, 0.35] as const,
};

type Kind = 'plate' | 'tile' | 'vector' | 'disc' | 'box';
type TitleKind = 'left' | 'below' | 'right' | 'none';

interface Lane {
  x0: number;
  spacing: number;
  stream: boolean;
  breaks: Set<number>;
  members: Entry[];
}

interface Copy {
  node: Node;
  key: string;
  consumers: Node[];
  board: Node | undefined;
}

interface Entry {
  key: string;
  node: Node;
  copy: Copy | undefined;
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
}

interface Level {
  entries: Entry[];
  module: string | null;
  copyOnly: boolean;
  y: number;
}

interface BoardRef {
  op: Node;
  a: Node;
  b: Node;
  linear: boolean;
  batched: boolean;
}

export interface FoldBox {
  module: StepModule;
  x: [number, number];
  z: [number, number];
  y: [number, number];
}

export interface ModuleIO {
  module: StepModule;
  input: Node;
  output: Node;
  outputBlock: Block;
  split: boolean;
}

export interface LevelLayout {
  blocks: Block[];
  boards: BoardGeometry[];
  boxes: FoldBox[];
  ranges: Map<string, [number, number]>;
  io: ModuleIO[];
  links: { copy: Block; original: Block }[];
  chips: Chip[];
  braces: { z: number; cx: number; cz: number };
  lo: number;
  top: number;
  sourceOf: ReadonlyMap<Node, Node>;
  copyFor(node: Node, consumer: Node): Block | undefined;
}

export interface Chip {
  block: Block;
  label: Node;
  x: number;
  arrow: boolean;
}

export interface LevelContext {
  folded(id: string): boolean;
  aligned: ReadonlyMap<Node, Node>;
}

export interface LevelPlan {
  readonly modules: SceneModule[];
  resolve(ctx: LevelContext): LevelLayout;
}

export const CHIP_SIDE = 0.55;
export const CHIP_LABEL_SIDE = 0.6;
export const CHIP_LIFT = 0.05;
export const FRAME_HALF: [number, number] = [0.42, 0.42];
export const COLUMN_NAME_BACK = 0.45;
export const COLUMN_NAME_PLATE_BACK = 0.55;
export const COLUMN_NAME_LIFT = 0.05;
export const COLUMN_NAME_PLATE_LIFT = 0.15;
export const ROW_NAME_SIDE = 0.45;
export const ROW_NAME_LIFT = 0.03;
export const BRACE_FRONT = 0.6;
export const ROW_GROUP_GAP = 0.45;

const zPlate = (r: number) => LEVEL_FRONT + r * LEVEL_ROW;
const zTile = (r: number, rows: number) => LEVEL_FRONT - TILE_BACK - (rows - 1 - r) * (rows > TILE_DENSE_ROWS ? TILE_ROW_DENSE : TILE_ROW);
const zVector = LEVEL_FRONT - VECTOR_BACK;

function laneSpacing(cols: number): number {
  return Math.min(SPACING_MAX, Math.max(SPACING_MIN, SPACING_SPAN / cols));
}

function position(lane: Lane, i: number): number {
  let breaks = 0;
  for (const b of lane.breaks) if (b <= i) breaks++;
  return lane.x0 + i * lane.spacing + breaks * COLUMN_BREAK;
}

const list = (nodes: Node | Node[]): Node[] => (Array.isArray(nodes) ? nodes : [nodes]);

export function levelPlan(spec: StepsSpec): LevelPlan {
  return new Plan(spec);
}

class Plan implements LevelPlan {
  readonly modules: SceneModule[] = [];
  private readonly stepModules: StepModule[];
  private readonly byId = new Map<string, StepModule>();
  private readonly visible: Node[];
  private readonly computed: Set<Node>;
  private readonly levels: Level[] = [];
  private readonly entries: Entry[] = [];
  private readonly entryOf = new Map<Node, Entry>();
  private readonly sourceOf = new Map<Node, Node>();
  private readonly drawnOn = new Map<Node, Node[]>();
  private readonly copies: Copy[] = [];
  private readonly copyEntries = new Map<Copy, Entry>();
  private readonly boards = new Map<Node, BoardRef>();
  private readonly tables = new Map<Node, Node>();
  private readonly rowGroups = new Map<Node, Node>();
  private readonly labelledGroups = new Set<Node>();
  private readonly slabbed = new Map<Node, Node>();
  private readonly groupWeights = new Map<Node, Node[]>();
  private readonly lanes: Lane[] = [];
  private readonly segments: { title: string; nodes: Node[]; params: boolean; nested: boolean }[] = [];
  private readonly stream: Lane;

  constructor(private readonly spec: StepsSpec) {
    this.stepModules = [...(spec.modules ?? [])];
    for (const m of this.stepModules) this.byId.set(m.id, m);
    this.visible = [...new Set(spec.forward.flatMap((s) => list(s.show)))].sort((a, b) => a.id - b.id);
    this.computed = new Set(this.visible.filter((n) => !isLeafLike(n)));
    this.stream = this.lane(STREAM_X, STREAM_SPACING, true);
    this.findSources();
    this.findBoards();
    this.findRowGroups();
    this.findCopies();
    this.buildLevels();
    this.placeSides();
    this.assignColumns();
    this.assignTitles();
    this.buildModules();
  }

  private lane(x0: number, spacing: number, stream = false): Lane {
    const lane: Lane = { x0, spacing, stream, breaks: new Set(), members: [] };
    this.lanes.push(lane);
    return lane;
  }

  private findSources(): void {
    const shown = new Set(this.visible);
    const operands = new Set([...this.computed].filter((u) => u.kind === 'op' && (u.op === 'matmul' || u.op === 'select_matmul')).map((u) => u.inputs[1]));
    for (const n of this.visible[0]?.graph.nodes ?? []) {
      if (!isLayoutOp(n) || shown.has(n) || isLeafLike(n) || operands.has(n)) continue;
      const source = this.source(n.inputs[0]);
      this.sourceOf.set(n, source);
      if (n.shape.length === 3 && drawnAlike(n, source)) this.drawnOn.set(source, [...(this.drawnOn.get(source) ?? []), n]);
    }
  }

  private source(n: Node): Node {
    return this.sourceOf.get(n) ?? n;
  }

  private isOutput(m: StepModule, n: Node): boolean {
    return m.outputs.some((o) => this.source(o) === n);
  }

  private isInput(m: StepModule, n: Node): boolean {
    return m.inputs.some((i) => this.source(i) === n);
  }

  private consumers(n: Node): Node[] {
    return [...this.computed].filter((u) => u.inputs.some((i) => this.source(i) === n));
  }

  private chain(id: string | null | undefined): StepModule[] {
    const out: StepModule[] = [];
    for (let m = id == null ? undefined : this.byId.get(id); m !== undefined; m = m.parent === null ? undefined : this.byId.get(m.parent)) out.unshift(m);
    return out;
  }

  private innermost(n: Node): StepModule | undefined {
    let best: StepModule | undefined;
    for (const m of this.stepModules) {
      if (!m.nodes.includes(n)) continue;
      if (best === undefined || this.chain(m.id).length > this.chain(best.id).length) best = m;
    }
    return best;
  }

  private split(m: StepModule): boolean {
    return this.stepModules.some((c) => c.parent === m.id && c.segment !== m.segment);
  }

  private head(n: Node): StepModule | undefined {
    const chain = this.chain(this.innermost(n)?.id);
    for (let i = chain.length - 1; i >= 0; i--) {
      const m = chain[i];
      if (m.sub === undefined && !this.split(m)) return m;
    }
    return undefined;
  }

  private findBoards(): void {
    for (const u of this.computed) {
      if (u.kind !== 'op') continue;
      if ((u.op === 'matmul' || u.op === 'select_matmul') && u.inputs.length === 2) {
        const [input, b] = u.inputs;
        const flat = (input.shape.length === 2 || (u.op === 'select_matmul' && input.shape.length === 3)) && b.shape.length === 2 && u.shape.length === 2;
        const batched = batchedCells(u);
        if (flat || batched) this.boards.set(u, { op: u, a: this.source(input), b, linear: isLeafLike(b), batched });
        continue;
      }
      const table = u.inputs.find((i) => i.kind === 'param' && i.shape.length === 2 && i.shape[1] === u.shape[u.shape.length - 1]);
      if (table !== undefined && u.inputs.some((i) => i.kind === 'label')) this.tables.set(u, table);
    }
  }

  private findRowGroups(): void {
    const source = (n: Node) => this.sourceOf.get(n) ?? n;
    const mark = (n: Node, group: Node): boolean => {
      const node = source(n);
      if (!this.computed.has(node) || this.rowGroups.has(node) || node.shape[0] !== group.shape[0]) return false;
      this.rowGroups.set(node, group);
      return true;
    };
    for (const u of this.computed) {
      if (u.kind !== 'op' || u.op !== 'grouped_linear') continue;
      const [, weight, group] = u.inputs;
      if (isLeafLike(weight) && weight.shape.length === 3 && !this.slabbed.has(weight)) {
        this.slabbed.set(weight, group);
        this.groupWeights.set(group, [...(this.groupWeights.get(group) ?? []), weight]);
      }
      mark(u, group);
      mark(u.inputs[0], group);
      this.labelledGroups.add(source(u.inputs[0]));
      mark(group, group);
    }
    for (let changed = this.rowGroups.size > 0; changed; ) {
      changed = false;
      for (const u of this.computed) {
        if (u.kind !== 'op' || explainOf(u.op as string)?.pattern !== 'elementwise') continue;
        const linked = [u, ...u.inputs].filter((n) => !isLeafLike(n) && n.shape.length > 0 && n.shape[0] === u.shape[0]);
        const group = linked.map((n) => this.rowGroups.get(source(n))).find((g) => g !== undefined);
        if (group === undefined) continue;
        for (const n of linked) if (mark(n, group)) changed = true;
      }
    }
  }

  private bands(group: Node): { start: number[]; height: number[]; rows: number[] } {
    const weights = this.groupWeights.get(group) ?? [];
    const data = Array.from(group.value.data, Math.round);
    const count = Math.max(0, ...weights.map((w) => w.shape[0]), ...data.map((g) => g + 1));
    const span = Math.max(0, ...weights.map((w) => (w.shape[1] - 1) * TILE_ROW_DENSE));
    const rows = Array.from({ length: count }, (_, g) => data.filter((v) => v === g).length);
    const start: number[] = [];
    const height: number[] = [];
    let z = LEVEL_FRONT;
    for (let g = 0; g < count; g++) {
      start.push(z);
      if (rows[g] === 0 && weights.length === 0) {
        height.push(0);
        continue;
      }
      height.push(Math.max((rows[g] - 1) * LEVEL_ROW, span));
      z += height[g] + LEVEL_ROW + ROW_GROUP_GAP;
    }
    return { start, height, rows };
  }

  private rowZs(n: Node, rows: number): number[] {
    const group = this.rowGroups.get(n);
    if (group === undefined) return Array.from({ length: rows }, (_, r) => zPlate(r));
    const { start, height, rows: counts } = this.bands(group);
    const seen = counts.map(() => 0);
    return Array.from({ length: rows }, (_, r) => {
      const g = Math.round(group.value.data[r]);
      return start[g] + (height[g] - (counts[g] - 1) * LEVEL_ROW) / 2 + seen[g]++ * LEVEL_ROW;
    });
  }

  private slabZ(weight: Node, g: number, row: number): number {
    const { start, height } = this.bands(this.slabbed.get(weight)!);
    return start[g] + (height[g] - (weight.shape[1] - 1) * TILE_ROW_DENSE) / 2 + row * TILE_ROW_DENSE;
  }

  private groupLabels(e: Entry, x: number, y: number, zs: number[]): Block['rowLabels'] {
    const group = this.rowGroups.get(e.node);
    if (group === undefined || !this.hasGroupLabels(e)) return [];
    const data = group.value.data;
    return zs.map((z, r) => ({ text: r === 0 || data[r] !== data[r - 1] ? groupText(group, data[r]) : '', col: r, x, y: y + ROW_NAME_LIFT, z, anchor: 'left' as const }));
  }

  private hasGroupLabels(e: Entry): boolean {
    const group = this.rowGroups.get(e.node);
    return group !== undefined && e.copy === undefined && e.table === undefined && this.labelledGroups.has(e.node) && e.rows === group.shape[0];
  }

  private findCopies(): void {
    for (const x of this.computed) {
      const producer = this.head(x);
      if (producer === undefined) continue;
      const groups = new Map<string, Node[]>();
      for (const u of this.consumers(x)) {
        const consumer = this.head(u);
        if (consumer === producer) continue;
        const key = consumer?.id ?? '';
        groups.set(key, [...(groups.get(key) ?? []), u]);
      }
      for (const [scope, consumers] of groups) {
        const board = consumers.find((u) => this.boards.get(u)?.a === x && this.boards.get(u)!.linear);
        this.copies.push({ node: x, key: `copy:${x.id}:${scope}`, consumers, board });
      }
    }
  }

  private copyOf(x: Node, consumer: Node): Copy | undefined {
    return this.copies.find((c) => c.node === x && c.consumers.includes(consumer));
  }

  private shapeOf(n: Node): [number, number] {
    if (this.slabbed.has(n)) return [n.shape[1], n.shape[2]];
    if (n.shape.length === 0) return [1, 1];
    if (n.shape.length === 1) {
      const main = n.inputs[0];
      const column = (u: Node) => u.shape.length === 1 && u.shape[0] === n.shape[0] && !isLeafLike(u) && this.shapeOf(u)[1] === 1;
      return !isLeafLike(n) && main !== undefined && ((main.shape.length === 2 && main.shape[0] === n.shape[0]) || n.inputs.some(column)) ? [n.shape[0], 1] : [1, n.shape[0]];
    }
    return rowsCols(n);
  }

  private discMemo = new Map<Node, boolean>();

  private isDisc(n: Node): boolean {
    const known = this.discMemo.get(n);
    if (known !== undefined) return known;
    this.discMemo.set(n, false);
    const cols = this.shapeOf(n)[1];
    const result =
      !this.boards.has(n) && !isLeafLike(n) &&
      (n.shape.length === 2 || (n.shape.length === 3 && plates(n).width === 1)) &&
      this.consumers(n).some((u) => {
        const main = u.inputs[0];
        if (this.source(main) === n) return u.shape.join() === main.shape.join() && this.isDisc(u);
        const used = u.inputs.slice(1).find((i) => this.source(i) === n);
        if (used === undefined) return false;
        const broadcasts = explainOf(u.op as string)?.pattern === 'elementwise' && used.shape.length <= u.shape.length && used.shape.every((size, i) => size === 1 || size === u.shape[u.shape.length - used.shape.length + i]);
        return main.shape.join() === u.shape.join() && broadcasts && this.shapeOf(u)[1] > cols;
      });
    this.discMemo.set(n, result);
    return result;
  }

  private entry(node: Node, level: Level, copy?: Copy): Entry {
    const [rows, cols] = this.shapeOf(node);
    const leaf = isLeafLike(node);
    let kind: Kind = 'plate';
    if (node.shape.length === 0) kind = 'box';
    else if (leaf && node.shape.length === 1) kind = 'vector';
    else if (!copy && this.isDisc(node)) kind = 'disc';
    const e: Entry = { key: copy?.key ?? `node:${node.id}`, node, copy, kind, level, rows, cols, lane: undefined, offsets: [], xs: undefined, title: 'none', quiet: false, table: undefined };
    level.entries.push(e);
    this.entries.push(e);
    if (copy === undefined) this.entryOf.set(node, e);
    else this.copyEntries.set(copy, e);
    return e;
  }

  private newLevel(copyOnly = false): Level {
    const level: Level = { entries: [], module: null, copyOnly, y: 0 };
    this.levels.push(level);
    return level;
  }

  private buildLevels(): void {
    const units = this.spec.forward.filter((s) => s.kind !== 'inputs').map((s) => list(s.show).filter((n) => this.computed.has(n)));
    const unitOf = new Map<Node, number>();
    units.forEach((prod, i) => prod.forEach((n) => unitOf.set(n, i)));
    const moveTo = new Map<Node, Node>();
    for (const board of this.boards.values()) {
      if (!board.linear) {
        if (this.computed.has(board.b)) moveTo.set(board.b, board.op);
        continue;
      }
      const a = board.a;
      if (!this.computed.has(a) || this.copyOf(a, board.op) !== undefined) continue;
      const users = this.consumers(a);
      if (unitOf.get(a) === unitOf.get(board.op)! - 1 && users.every((u) => unitOf.get(u) === unitOf.get(board.op) && this.boards.get(u)?.a === a)) moveTo.set(a, board.op);
    }
    const levelOfNode = new Map<Node, Level>();
    units.forEach((prod, index) => {
      for (const copy of this.copies) {
        if (copy.board !== undefined) continue;
        const first = Math.min(...copy.consumers.map((u) => unitOf.get(u) ?? Infinity));
        if (first === index) this.entry(copy.node, this.newLevel(true), copy);
      }
      const own = prod.filter((n) => !moveTo.has(n) && !this.sourceOf.has(n));
      if (own.length === 0) return;
      const depth = new Map<Node, number>();
      for (const n of own) depth.set(n, Math.max(0, ...n.inputs.map((i) => this.source(i)).filter((i) => depth.has(i)).map((i) => depth.get(i)! + 1)));
      const levels = Array.from({ length: Math.max(...depth.values()) + 1 }, () => this.newLevel());
      for (const n of own) {
        const level = levels[depth.get(n)!];
        for (const copy of this.copies) if (copy.board !== undefined && this.boards.get(n)?.a === copy.node && copy.consumers.includes(n) && !this.copyEntries.has(copy)) this.entry(copy.node, level, copy);
        for (const [moved, target] of moveTo) if (target === n && !this.entryOf.has(moved) && this.boards.get(n)?.a === moved) this.entry(moved, level);
        const board = this.boards.get(n);
        if (board !== undefined) this.entry(board.b, level).kind = 'tile';
        const table = this.tables.get(n);
        if (table !== undefined) {
          const t = this.entry(table, level);
          t.kind = 'tile';
          t.table = n;
        }
        this.entry(n, level);
        levelOfNode.set(n, level);
      }
    });
    for (const [view, source] of this.sourceOf) {
      const e = this.entryOf.get(source);
      if (e !== undefined) this.entryOf.set(view, e);
    }
    for (const level of this.levels) level.module = this.levelModule(level);
  }

  private levelModule(level: Level): string | null {
    const own = level.entries.filter((e) => e.copy === undefined && this.computed.has(e.node)).map((e) => e.node);
    if (own.length === 0) {
      const copy = level.entries.find((e) => e.copy !== undefined)?.copy;
      return copy === undefined ? null : (this.innermost(copy.consumers[0])?.id ?? null);
    }
    const chains = own.map((n) => this.chain(this.innermost(n)?.id));
    let common: StepModule | undefined;
    for (let i = 0; i < chains[0].length; i++) {
      if (chains.every((c) => c[i] === chains[0][i])) common = chains[0][i];
      else break;
    }
    if (common !== undefined && own.every((n) => this.isOutput(common!, n))) return common.parent;
    return common?.id ?? null;
  }

  private placeSides(): void {
    const transposed = new Set(this.visible.filter((n) => n.kind === 'op' && n.op === 'transpose' && isLeafLike(n)).map((n) => n.inputs[0]));
    for (const n of this.visible) {
      if (this.computed.has(n) || this.entryOf.has(n) || n.kind === 'label' || transposed.has(n)) continue;
      const user = this.consumers(n)[0];
      if (user === undefined) continue;
      const main = this.source(user.inputs[0]) === n ? user : user.inputs[0];
      const host = this.entryOf.get(main);
      if (host === undefined) continue;
      this.entry(n, host.level);
    }
  }

  private occupied(level: Level, lane: Lane, skip: Entry): Set<number> {
    const out = new Set<number>();
    for (const e of level.entries) if (e !== skip && e.lane === lane && e.kind !== 'tile' && e.kind !== 'vector') e.offsets.forEach((o) => out.add(o));
    return out;
  }

  private setLane(e: Entry, lane: Lane, offsets: number[]): void {
    e.lane = lane;
    e.offsets = offsets;
    lane.members.push(e);
  }

  private drawnA(board: BoardRef): Entry {
    const copy = this.copyOf(board.a, board.op);
    return (copy === undefined ? undefined : this.copyEntries.get(copy)) ?? this.entryOf.get(board.a)!;
  }

  private assignColumns(): void {
    const done = new Set<Entry>();
    const beside: [Entry, Entry][] = [];
    const range = (start: number, n: number) => Array.from({ length: n }, (_, i) => start + i);
    for (const level of this.levels) {
      const ordered = [...level.entries].sort((p, q) => rank(p) - rank(q));
      for (const e of ordered) {
        if (done.has(e)) continue;
        done.add(e);
        const n = e.node;
        if (e.copy !== undefined) {
          const original = this.entryOf.get(e.copy.node)!;
          if (original.lane !== undefined) this.setLane(e, original.lane, original.offsets);
          else e.xs = original.xs;
          continue;
        }
        if (e.kind === 'tile' && e.table === undefined) continue;
        const table = e.table ?? (this.tables.has(n) ? n : undefined);
        if (table !== undefined) {
          this.setLane(e, this.stream, range(0, e.cols));
          continue;
        }
        const board = this.boards.get(n);
        if (board !== undefined) {
          this.placeBoardGroup(board, level, done);
          continue;
        }
        if (e.kind === 'vector') {
          const user = this.consumers(n)[0];
          const host = this.entryOf.get(this.source(user.inputs[0]) === n ? user : user.inputs[0])!;
          if (host.lane !== undefined) this.setLane(e, host.lane, host.offsets.slice(0, e.cols));
          continue;
        }
        if (isLeafLike(n)) {
          const partners = new Set(this.consumers(n).flatMap((u) => u.inputs).filter((i) => i !== n).map((i) => this.entryOf.get(this.sourceOf.get(i) ?? i)));
          const taken = this.occupied(level, this.stream, e);
          const blocked = level.entries.some((o) => partners.has(o) && o.lane === this.stream && range(0, e.cols).some((k) => o.offsets.includes(k)));
          const start = blocked ? Math.max(...taken) + 3 : 0;
          this.setLane(e, this.stream, range(start, e.cols));
          continue;
        }
        const inputs = n.inputs.map((i) => this.entryOf.get(i)).filter((i): i is Entry => i !== undefined);
        const main = inputs[0];
        if (main === undefined) {
          this.setLane(e, this.stream, range(0, e.cols));
          continue;
        }
        if (e.kind === 'disc' || e.kind === 'box' || e.cols === 1) {
          beside.push([e, main]);
          continue;
        }
        if (main.lane === undefined) {
          this.setLane(e, this.stream, range(0, e.cols));
          continue;
        }
        const traced = this.tracedOffsets(e, main.lane);
        if (traced !== undefined && !traced.some((o) => this.occupied(level, main.lane!, e).has(o))) {
          this.setLane(e, main.lane, traced);
          continue;
        }
        if (n.kind === 'op' && n.op === 'cat' && inputs.every((i) => i.lane === main.lane)) {
          this.setLane(e, main.lane, inputs.flatMap((i) => i.offsets));
          continue;
        }
        const start = n.kind === 'op' && n.op === 'narrow' ? ((n.attrs as { start: number }).start ?? 0) : n.kind === 'op' && n.op === 'select' && n.inputs[0].shape.length === 3 && (n.attrs as { dim: number }).dim === 1 ? (n.attrs as { index: number }).index * n.inputs[0].shape[2] : 0;
        let offsets = main.offsets.length >= start + e.cols ? main.offsets.slice(start, start + e.cols) : range(main.offsets[0] ?? 0, e.cols);
        const taken = this.occupied(level, main.lane, e);
        if (offsets.some((o) => taken.has(o))) offsets = range(Math.max(...taken) + 1, e.cols);
        this.setLane(e, main.lane, offsets);
      }
    }
    for (const lane of this.lanes) {
      const group = this.groupSize(lane);
      const width = Math.max(0, ...lane.members.flatMap((m) => m.offsets)) + 1;
      if (group > 0) for (let b = group; b < width; b += group) lane.breaks.add(b);
    }
    for (const e of this.entries) if (e.xs === undefined && e.lane !== undefined) e.xs = this.slabbed.has(e.node) ? slabXs(e) : e.offsets.map((o) => position(e.lane!, o));
    for (const [e, main] of beside) e.xs = this.sideXs(e, main);
    for (const e of this.entries) if (e.xs === undefined && e.copy !== undefined) e.xs = this.entryOf.get(e.copy.node)!.xs;
    for (const e of this.entries) if (e.xs === undefined) e.xs = this.sideXs(e, this.entryOf.get(e.node.inputs[0]) ?? e);
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
      const from = input === undefined ? undefined : this.entryOf.get(input);
      const at = from === undefined || from.lane !== lane || from.cols !== rowsCols(from.node)[1] ? undefined : sourceIndex(input!, from.node, picks[0].index);
      if (at === undefined) traced = false;
      else offsets[col] = from!.offsets[rowCol(from!.node, at)[1]];
    });
    if (!traced || offsets.some((o) => o === undefined) || new Set(offsets).size !== offsets.length) return undefined;
    return offsets as number[];
  }

  private placeBatchedBoard(board: BoardRef, level: Level, done: Set<Entry>): void {
    const a = this.drawnA(board);
    const out = this.entryOf.get(board.op)!;
    const tile = level.entries.find((e) => e.node === board.b && e.kind === 'tile')!;
    const aXs = a.xs ?? (a.lane === undefined ? [0] : a.offsets.map((o) => position(a.lane!, o)));
    const lane = this.lane(a.level === level ? Math.max(...aXs) + BOARD_GAP : Math.min(...aXs), a.lane?.spacing ?? STREAM_SPACING);
    const range = (n: number) => Array.from({ length: n }, (_, i) => i);
    this.setLane(out, lane, range(out.cols));
    this.setLane(tile, lane, range(tile.cols));
    done.add(out);
    done.add(tile);
  }

  private placeBoardGroup(board: BoardRef, level: Level, done: Set<Entry>): void {
    if (board.batched) {
      this.placeBatchedBoard(board, level, done);
      return;
    }
    const a = this.drawnA(board);
    const group = [...this.boards.values()].filter((b) => !b.batched && this.entryOf.get(b.op)?.level === level && this.drawnA(b) === a).sort((p, q) => p.op.id - q.op.id);
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
    group.forEach((b, i) => {
      const n = b.b.shape[1];
      if (i > 0 && same) lane.breaks.add(offset);
      const offsets = Array.from({ length: n }, (_, k) => (same ? offset + k : (this.drawnA(b).offsets[boardStart(b.op)] ?? 0) + k));
      const out = this.entryOf.get(b.op)!;
      const tile = level.entries.find((e) => e.node === b.b && e.kind === 'tile')!;
      this.setLane(out, lane, offsets);
      this.setLane(tile, lane, offsets);
      done.add(out);
      done.add(tile);
      if (same) offset += n;
    });
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
      for (const node of [m.node, ...(this.drawnOn.get(m.node) ?? [])]) {
        const { count, width } = plates(node);
        if (node.shape.length === 3 && count > 1 && width > 1 && (size === 0 || width < size)) size = width;
        for (const u of this.computed) {
          if (u.kind !== 'op' || !u.inputs.includes(node)) continue;
          const attrs = u.attrs as { size?: number; length?: number; dim?: number };
          const s = typeof attrs.size === 'number' ? attrs.size : u.op === 'narrow' && attrs.dim === node.shape.length - 1 ? attrs.length : undefined;
          if (s !== undefined && s > 0 && s < m.cols && (size === 0 || s < size)) size = s;
        }
      }
    }
    return size;
  }

  private assignTitles(): void {
    for (const level of this.levels) {
      const lanes = new Map<Lane | undefined, Entry[]>();
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
        const board = this.boards.get(e.node);
        const passedOn = this.consumers(e.node).every((u) => this.copies.some((c) => c.node === e.node && c.board === undefined && c.consumers.includes(u)));
        if (board !== undefined && e.copy === undefined && this.drawnA(board).level === level) {
          e.title = passedOn ? 'right' : 'below';
          continue;
        }
        if (e.copy !== undefined && e.copy.board === undefined) {
          e.title = this.boards.has(e.node) && passedOn ? 'right' : 'below';
          continue;
        }
        lanes.set(e.lane, [...(lanes.get(e.lane) ?? []), e]);
      }
      for (const group of lanes.values()) {
        group.sort((p, q) => p.xs![0] - q.xs![0]).forEach((e, i) => {
          if (i === 0) e.title = 'left';
          else if (e.lane !== undefined && isLeafLike(e.node) && Math.min(...e.offsets) > Math.max(...group[i - 1].offsets) + 1) e.title = 'below';
        });
      }
      const vectors = level.entries.filter((e) => e.kind === 'vector').sort((p, q) => p.xs![0] - q.xs![0]);
      if (vectors.length > 0) vectors[vectors.length - 1].title = 'right';
    }
    const norms = new Set(this.stepModules.filter((m) => !this.stepModules.some((c) => c.parent === m.id) && m.nodes.some((n) => this.computed.has(n) && this.isDisc(n))));
    for (const e of this.entries) {
      if (e.kind === 'vector') e.quiet = true;
      if (e.copy === undefined && this.computed.has(e.node) && [...norms].some((m) => m.nodes.includes(e.node) && !this.isOutput(m, e.node))) e.quiet = true;
    }
  }

  private buildModules(): void {
    for (const m of this.stepModules) {
      if (!m.nodes.some((n) => this.computed.has(n))) continue;
      const parent = m.parent === null ? undefined : this.byId.get(m.parent);
      const side = m.parent === null && this.split(m) ? 'outer' : parent !== undefined && !this.split(parent) ? 'left' : 'right';
      this.modules.push({ id: m.id, title: m.title, subtitle: m.subtitle || undefined, nodes: [...m.nodes, ...m.params], side, inner: side === 'left' });
    }
    const runs: { title: string; subtitle?: string; nodes: Node[]; inside: boolean }[] = [];
    let last: string | undefined;
    for (const step of this.spec.forward) {
      const owner = step.module === undefined ? undefined : this.byId.get(step.module);
      const isModule = owner !== undefined && this.chain(owner.id).some((m) => m.segment === step.segment);
      const title = step.kind === 'inputs' ? undefined : (step.brace?.title ?? (isModule ? undefined : step.segment));
      if (title === undefined) {
        last = undefined;
        continue;
      }
      const nodes = list(step.show);
      if (last === title) runs[runs.length - 1].nodes.push(...nodes);
      else runs.push({ title, subtitle: step.brace?.subtitle, nodes: [...nodes], inside: step.brace !== undefined && isModule });
      last = title;
    }
    runs.forEach((run, i) => {
      const params = run.nodes.some((n) => isLeafLike(n));
      const nested = run.nodes.some((n) => this.innermost(n) !== undefined);
      const copies = this.copies.filter((c) => c.board === undefined && c.consumers.some((u) => run.nodes.includes(u))).map((c) => c.node);
      const side = params && !run.inside ? 'right' : 'left';
      this.segments.push({ title: run.title, nodes: run.nodes, params, nested });
      this.modules.push({ id: `segment:${i}`, title: run.title, subtitle: run.subtitle, nodes: [...run.nodes, ...copies], side, inner: side === 'left' && (nested || run.inside) });
    });
  }

  resolve(ctx: LevelContext): LevelLayout {
    const folded = (id: string) => ctx.folded(id);
    const isFolded = (m: StepModule) => folded(m.id);
    const hiddenNode = (n: Node) => this.computed.has(n) && this.stepModules.some((m) => isFolded(m) && m.nodes.includes(n) && !this.isOutput(m, n));
    const outerFold = (id: string | null): StepModule | undefined => this.chain(id).find(isFolded);
    const hidden = new Map<Entry, boolean>();
    const hiddenOf = (e: Entry): boolean => {
      const known = hidden.get(e);
      if (known !== undefined) return known;
      let value: boolean;
      if (e.copy !== undefined) value = e.copy.consumers.every(hiddenNode);
      else if (this.computed.has(e.node)) value = hiddenNode(e.node);
      else if (e.node.kind === 'buffer') {
        const fold = outerFold(e.level.module);
        const leaf = fold !== undefined && !this.stepModules.some((m) => m.parent === fold.id);
        value = fold !== undefined && !(leaf && this.consumers(e.node).some((u) => this.innermost(u)?.id === fold.id));
      }
      else value = false;
      hidden.set(e, value);
      return value;
    };
    const flags = (level: Level) => {
      const shown = level.entries.filter((e) => !hiddenOf(e));
      const own = shown.filter((e) => e.kind !== 'tile' && e.kind !== 'vector' && !(isLeafLike(e.node) && e.copy === undefined));
      return {
        visible: shown.length > 0,
        board: shown.some((e) => e.kind === 'tile' && e.table === undefined),
        tileRows: Math.max(0, ...shown.filter((e) => e.kind === 'tile' && e.table === undefined).map((e) => e.rows)),
        linear: shown.some((e) => e.kind === 'tile' && isLeafLike(e.node) && e.table === undefined),
        scalar: own.length > 0 && own.every((e) => e.kind === 'disc' || e.kind === 'box'),
        loss: own.some((e) => e.kind === 'box'),
        vector: shown.some((e) => e.kind === 'vector'),
        buffer: shown.some((e) => e.node.kind === 'buffer'),
        copy: level.copyOnly,
        copied: shown.some((e) => e.copy !== undefined),
        module: level.module,
      };
    };
    type Flags = ReturnType<typeof flags>;
    const opens = (f: Flags, prev: Flags) => f.module !== null && f.module !== prev.module && this.chain(f.module).some((m) => m.id === prev.module);
    const gap = (f: Flags, prev: Flags, before: Flags | undefined): number => {
      if (f.scalar) return prev.board ? GAP.scalarAfterBoard : f.loss ? GAP.loss : GAP.scalar;
      if (f.board) return !f.linear ? GAP.board : f.copied || prev.board ? GAP.linearBoard : GAP.ownBoard;
      if (f.copy) return prev.board ? GAP.copyAfterBoard + GAP.copyPerTileRow * prev.tileRows : GAP.copy;
      if (prev.board) return opens(f, prev) ? GAP.moduleAfterBoard : GAP.afterBoard;
      if (prev.copy) return GAP.afterCopy;
      if (prev.scalar) return f.vector ? GAP.vectorAfterScalar : before?.board ? GAP.afterBoardScalar : GAP.afterScalar;
      if (prev.vector) return GAP.afterVector;
      return GAP.plate;
    };
    const boxY = new Map<string, [number, number]>();
    let y = 0;
    let box: StepModule | undefined;
    let bs = 0;
    let first = true;
    let inBox = 0;
    let after = false;
    let brd = false;
    let boxBrd = false;
    let prev: Flags | undefined;
    let before: Flags | undefined;
    const close = () => {
      const top = Math.max(y + (inBox > 0 ? (brd ? FOLD.endBoard : FOLD.end) : 0), bs + FOLD.minHeight);
      boxBrd = inBox > 0 && brd;
      boxY.set(box!.id, [bs, top]);
      y = top;
      box = undefined;
      after = true;
    };
    const levelY: number[] = [];
    this.levels.forEach((level) => {
      const fm = outerFold(level.module);
      const f = flags(level);
      if (box !== undefined && fm !== box) close();
      if (fm !== undefined && box === undefined) {
        box = fm;
        bs = y + (brd ? FOLD.startAfterBoard : FOLD.start);
        y = bs;
        inBox = 0;
      }
      if (f.visible) {
        if (first) y = 0;
        else if (box !== undefined) {
          y += inBox > 0 ? (f.board ? FOLD.board : f.buffer ? FOLD.buffer : FOLD.other) : FOLD.first;
          inBox++;
        } else if (after) y += boxBrd ? FOLD.afterBoard : FOLD.after;
        else y += gap(f, prev!, before);
        first = false;
        after = false;
        brd = f.board;
        before = prev;
        prev = f;
      }
      level.y = y;
      levelY.push(y);
    });
    if (box !== undefined) close();
    const top = y + TOP_ROOM;

    for (const m of this.stepModules) {
      if (!isFolded(m) || boxY.has(m.id) || outerFold(m.parent) !== undefined) continue;
      const outLevel = this.levels.findIndex((l) => l.entries.some((e) => this.isOutput(m, e.node) && e.copy === undefined));
      if (outLevel <= 0) continue;
      const below = [...this.levels.slice(0, outLevel)].reverse().find((l) => flags(l).visible);
      boxY.set(m.id, [(below?.y ?? 0) + FOLD.emptyBelow, this.levels[outLevel].y - FOLD.emptyAbove]);
    }

    const blockOf = new Map<Entry, Block>();
    for (const e of this.entries) blockOf.set(e, this.block(e, hiddenOf(e), ctx.aligned));
    const blocks = [...blockOf.values()];

    const boards: BoardGeometry[] = [];
    for (const board of this.boards.values()) {
      const out = this.entryOf.get(board.op)!;
      const tile = out.level.entries.find((e) => e.node === board.b && e.kind === 'tile')!;
      const a = this.drawnA(board);
      const siblings = [...this.boards.values()].filter((b) => this.entryOf.get(b.op)?.level === out.level && this.drawnA(b) === a && a.level === out.level).sort((p, q) => p.op.id - q.op.id);
      const at = siblings.findIndex((b) => b === board);
      const previous = at > 0 ? out.level.entries.find((e) => e.node === siblings[at - 1].b && e.kind === 'tile') : undefined;
      boards.push({ op: board.op, a: blockOf.get(a)!, sharedA: a.level !== out.level, b: blockOf.get(tile)!, out: blockOf.get(out)!, y: out.level.y, previous: previous === undefined ? undefined : blockOf.get(previous) });
    }

    const boxes: FoldBox[] = [];
    for (const m of this.stepModules) {
      const range = boxY.get(m.id);
      if (range === undefined || !isFolded(m) || outerFold(m.parent) !== undefined) continue;
      const inside = this.entries.filter((e) => {
        const l = e.level.module;
        return l !== null && this.chain(l).includes(m);
      });
      const keys = inside.length > 0 ? inside : this.entries.filter((e) => e.copy === undefined && (this.isOutput(m, e.node) || this.isInput(m, e.node)));
      const xs = keys.flatMap((e) => e.xs ?? []);
      const zs = keys.flatMap((e) => this.zs(e));
      boxes.push({ module: m, x: [Math.min(...xs) - BOX_MARGIN.left, Math.max(...xs) + BOX_MARGIN.right], z: [Math.min(...zs) - BOX_MARGIN.back, Math.max(...zs) + BOX_MARGIN.front], y: range });
    }

    const ranges = new Map<string, [number, number]>();
    const spanOf = (levels: Level[]): [number, number] | undefined => {
      const shown = levels.filter((l) => flags(l).visible);
      if (shown.length === 0) return undefined;
      const lo = shown[0];
      const hi = shown[shown.length - 1];
      const board = (l: Level) => flags(l).board;
      const table = (l: Level) => l.entries.some((e) => e.kind === 'tile' && e.table !== undefined && !hiddenOf(e));
      const above = shown.length > 1 ? BRACE_PAD.above : table(hi) ? BRACE_PAD.aboveTable : board(hi) ? BRACE_PAD.aboveBoard : BRACE_PAD.above;
      return [lo.y - (board(lo) || table(lo) ? BRACE_PAD.belowBoard : BRACE_PAD.below), hi.y + above];
    };
    for (const m of this.stepModules) {
      if (outerFold(m.parent) !== undefined) continue;
      const own = this.levels.filter((l) => (l.module !== null && this.chain(l.module).includes(m)) || l.entries.some((e) => e.copy === undefined && this.isOutput(m, e.node)));
      const range = isFolded(m) ? boxY.get(m.id) : spanOf(own);
      if (range !== undefined) ranges.set(m.id, range);
    }
    const sideOf = new Map(this.modules.map((m) => [m.id, m.side]));
    const segmentLevels = this.segments.map((segment) =>
      segment.nodes.some((n) => hiddenNode(n) || this.chain(this.innermost(n)?.id).some(isFolded))
        ? []
        : this.levels.filter((l) => l.entries.some((e) => !hiddenOf(e) && (e.copy === undefined ? segment.nodes.includes(e.node) : e.copy.consumers.some((u) => segment.nodes.includes(u))))),
    );
    segmentLevels.forEach((levels, i) => {
      const side = sideOf.get(`segment:${i}`);
      const next = segmentLevels.find((other, j) => j > i && other.length > 0 && sideOf.get(`segment:${j}`) === side);
      const own = next === undefined ? levels : levels.filter((l) => !next.includes(l));
      const range = spanOf(own.length > 0 ? own : levels);
      if (range !== undefined) ranges.set(`segment:${i}`, range);
    });
    for (const m of this.stepModules) {
      const range = ranges.get(m.id);
      if (range === undefined || (isFolded(m) && this.split(m))) continue;
      const direct = (input: Node) => this.consumers(input).some((u) => m.nodes.includes(u) && !this.copies.some((c) => c.node === input && c.consumers.includes(u)));
      const inputLevels = this.levels.filter((l) => flags(l).visible && l.y < range[0] && l.entries.some((e) => e.copy === undefined && !hiddenOf(e) && this.isInput(m, e.node) && direct(e.node)));
      const covered = (y: number) => [...ranges].some(([id, r]) => id !== m.id && sideOf.get(id) === sideOf.get(m.id) && r[0] <= range[0] && y <= r[1] && r[1] < range[1]);
      const lows = inputLevels.filter((l) => !covered(l.y)).map((l) => l.y - BRACE_PAD.below);
      if (lows.length > 0) ranges.set(m.id, [Math.min(range[0], ...lows), range[1]]);
    }

    const io: ModuleIO[] = [];
    for (const m of this.stepModules) {
      if (!isFolded(m) || outerFold(m.parent) !== undefined) continue;
      for (const output of m.outputs) {
        const outside = this.copies.find((c) => c.node === this.source(output) && c.consumers.some((u) => !m.nodes.includes(u)));
        const outputEntry = (outside === undefined ? undefined : this.copyEntries.get(outside)) ?? this.entryOf.get(output);
        if (outputEntry === undefined) continue;
        for (const input of m.inputs) if (this.entryOf.has(input)) io.push({ module: m, input, output, outputBlock: blockOf.get(outputEntry)!, split: this.split(m) });
      }
    }

    const chips: Chip[] = [];
    for (const e of this.entries) {
      if (e.copy !== undefined || e.kind !== 'plate') continue;
      const block = blockOf.get(e)!;
      const right = Math.max(...(e.xs ?? [0]));
      const table = this.tables.get(e.node);
      const input = table === undefined ? undefined : e.node.inputs.find((i) => i.kind === 'label');
      if (input !== undefined) chips.push({ block, label: input, x: right + CHIP_SIDE, arrow: true });
      const select = this.selectLabel(e.node);
      if (select !== undefined) chips.push({ block, label: select, x: right + CHIP_LABEL_SIDE, arrow: false });
    }
    const streamWidth = Math.max(1, ...this.stream.members.flatMap((m) => m.offsets.map((o) => o + 1)));
    const rows = this.rowsOf();
    const links = [...this.copyEntries.entries()].map(([copy, e]) => ({ copy: blockOf.get(e)!, original: blockOf.get(this.entryOf.get(copy.node)!)! }));
    return {
      blocks,
      boards,
      boxes,
      ranges,
      io,
      links,
      chips,
      braces: { z: Math.max(zPlate(rows - 1), ...this.entries.filter((en) => en.kind === 'plate').flatMap((en) => this.zs(en))) + BRACE_FRONT, cx: position(this.stream, (streamWidth - 1) / 2), cz: zPlate((rows - 1) / 2) },
      lo: 0,
      top,
      sourceOf: this.sourceOf,
      copyFor: (node, consumer) => {
        const copy = this.copyOf(this.source(node), consumer);
        const e = copy === undefined ? undefined : this.copyEntries.get(copy);
        return e === undefined ? undefined : blockOf.get(e);
      },
    };
  }

  private slabsOf(e: Entry, elements: Element[], footprint: readonly [number, number], y: number): Slab[] | undefined {
    const group = this.slabbed.get(e.node);
    if (group === undefined) return undefined;
    const used = new Set(Array.from(group.value.data, Math.round));
    return Array.from({ length: e.node.shape[0] }, (_, g) => {
      const members = elements.filter((el) => el.index[0] === g);
      const xs = members.map((el) => el.x);
      const zs = members.map((el) => el.z);
      const [x0, x1, z0, z1] = [Math.min(...xs), Math.max(...xs), Math.min(...zs), Math.max(...zs)];
      return { plate: { x: (x0 + x1) / 2, y, z: (z0 + z1) / 2, w: x1 - x0 + footprint[0] + PLATE_MARGIN, d: z1 - z0 + footprint[1] + PLATE_MARGIN }, used: used.has(g) };
    });
  }

  private rowGroupsOf(e: Entry): number[] | undefined {
    const group = this.rowGroups.get(e.node);
    if (group === undefined || e.rows !== group.shape[0]) return undefined;
    return Array.from(group.value.data, Math.round);
  }

  private selectLabel(n: Node): Node | undefined {
    for (const u of this.consumers(n)) {
      const label = u.inputs.map(labelBehind).find((l) => l !== undefined);
      if (label !== undefined && u.inputs[0] === n && !this.tables.has(u)) return label;
    }
    return undefined;
  }

  private zs(e: Entry): number[] {
    if (e.kind === 'tile') return Array.from({ length: e.rows }, (_, r) => zTile(r, e.rows));
    if (e.kind === 'vector') return [zVector];
    if (e.kind === 'box') return [zPlate((this.rowsOf() - 1) / 2)];
    if (this.slabbed.has(e.node)) return Array.from({ length: e.node.shape[0] * e.rows }, (_, i) => this.slabZ(e.node, Math.floor(i / e.rows), i % e.rows));
    return this.rowZs(e.node, e.rows);
  }

  private rowsOf(): number {
    return Math.max(1, ...this.entries.filter((e) => e.kind === 'plate').map((e) => e.rows));
  }

  private block(e: Entry, hidden: boolean, aligned: ReadonlyMap<Node, Node>): Block {
    const y = e.level.y;
    const xs = e.xs ?? [0];
    const zs = this.zs(e);
    const n = e.node;
    const elements: Element[] = [];
    const size = n.value.size;
    for (let offset = 0; offset < size; offset++) {
      const index = indexOf(n.shape, offset);
      const slab = this.slabbed.has(n);
      const [row, col] = slab ? [index[1], index[2]] : e.rows > 1 && e.cols === 1 ? [offset, 0] : n.shape.length >= 2 ? rowCol(n, index) : [0, offset];
      const z = slab ? this.slabZ(n, index[0], row) : zs[Math.min(row, zs.length - 1)];
      elements.push({ index, row, col, x: xs[Math.min(col, xs.length - 1)], y, z });
    }
    const spacing = e.lane?.spacing ?? STREAM_SPACING;
    const footprint: [number, number] =
      e.kind === 'tile'
        ? [spacing >= STREAM_SPACING ? TILE_WIDTH : TILE_WIDTH_NARROW, e.rows > TILE_DENSE_ROWS ? TILE_DEPTH_DENSE : TILE_DEPTH]
        : e.kind === 'disc'
          ? DISC_FOOTPRINT
          : e.kind === 'box'
            ? BOX_FOOTPRINT
            : this.slabbed.has(n)
              ? [STREAM_FOOTPRINT, TILE_DEPTH_DENSE]
              : e.lane?.stream === true
                ? [STREAM_FOOTPRINT, STREAM_FOOTPRINT]
                : [Math.min(BAR_FOOTPRINT, spacing - BAR_CLEARANCE), Math.min(BAR_FOOTPRINT, spacing - BAR_CLEARANCE)];
    const ex = elements.map((el) => el.x);
    const ez = elements.map((el) => el.z);
    const x0 = Math.min(...ex);
    const x1 = Math.max(...ex);
    const z0 = Math.min(...ez);
    const z1 = Math.max(...ez);
    const scalar = e.kind === 'disc' || e.kind === 'box';
    const front = Math.max(...zs);
    let title: Anchor;
    if (e.kind === 'tile') title = { x: x0 + TITLE.board[0], y, z: Math.min(...zs) + TITLE.board[1], anchor: 'left' };
    else if (e.kind === 'vector') title = { x: x1 + TITLE.vector[0], y: y + TITLE.vector[1], z: zVector, anchor: 'right' };
    else if (e.title === 'below' || (e.title === 'left' && this.hasGroupLabels(e))) title = { x: (x0 + x1) / 2, y: y + TITLE.below[0], z: front + TITLE.below[1], anchor: 'below' };
    else if (e.title === 'right') title = { x: x1 + TITLE.right[0], y: y + TITLE.right[1], z: front, anchor: 'right' };
    else title = { x: x0 + TITLE.left[0], y: y + TITLE.left[1], z: front, anchor: 'left' };
    const label = e.table === undefined ? undefined : e.table.inputs.find((i) => i.kind === 'label');
    const tokens = aligned.get(n)?.tokens;
    const named = e.kind === 'tile' ? tokens : this.selectLabel(n) !== undefined ? (this.selectLabel(n)!.tokens ?? undefined) : undefined;
    const slabs = this.slabsOf(e, elements, footprint, y);
    const chunks = slabs === undefined && !scalar ? chunksOf(n, elements, footprint, y) : undefined;
    const slabGroup = this.slabbed.get(n);
    const slabLabels = slabGroup === undefined ? undefined : slabs!.map((slab, g) => ({ text: groupText(slabGroup, g), col: g, x: x1 + ROW_NAME_SIDE, y: y + ROW_NAME_LIFT, z: slab.plate.z, anchor: 'right' as const }));
    const classLabels = (named ?? []).slice(0, e.cols).map((text, col) =>
      e.kind === 'tile'
        ? { text, col, x: xs[col], y: y + COLUMN_NAME_LIFT, z: Math.min(...zs) - COLUMN_NAME_BACK, anchor: 'above' as const }
        : { text, col, x: xs[col], y: y + COLUMN_NAME_PLATE_LIFT, z: Math.min(...zs) - COLUMN_NAME_PLATE_BACK, anchor: 'above' as const },
    );
    const rowTokens = label?.tokens;
    const rowLabels = slabLabels !== undefined ? slabLabels : rowTokens === undefined ? this.groupLabels(e, x0 - ROW_NAME_SIDE, y, zs) : rowTokens.slice(0, e.rows).map((text, r) => ({ text, col: r, x: x0 - ROW_NAME_SIDE, y: y + ROW_NAME_LIFT, z: zs[r], anchor: 'left' as const }));
    const framed = e.kind === 'plate' && aligned.has(n) && e.copy === undefined;
    return {
      node: n,
      key: e.key,
      copyOf: e.copy?.node,
      hidden,
      quiet: e.quiet,
      untitled: e.title === 'none',
      role: e.kind === 'tile' ? 'board-b' : scalar ? 'scalar' : 'planar',
      shape: e.kind === 'tile' ? 'tile' : e.kind === 'disc' ? 'disc' : e.kind === 'box' ? 'box' : 'bar',
      rows: e.rows,
      cols: e.cols,
      batched: false,
      elements,
      footprint,
      plate: scalar ? undefined : { x: (x0 + x1) / 2, y, z: (z0 + z1) / 2, w: x1 - x0 + footprint[0] + PLATE_MARGIN, d: z1 - z0 + footprint[1] + PLATE_MARGIN },
      slabs,
      chunks,
      rowGroups: this.rowGroupsOf(e),
      title,
      classLabels,
      rowLabels,
      frameHalf: framed ? FRAME_HALF : label !== undefined ? [(x1 - x0) / 2 + TILE_FRAME_PAD[0], TILE_FRAME_PAD[1]] : undefined,
      frameRows: framed ? zs.map((z) => ({ z, y })) : label !== undefined ? Array.from({ length: label.value.size }, () => ({ z: zs[0], y })) : [],
      scales: {},
      extent: [],
      table: label === undefined ? undefined : { label, half: [(x1 - x0) / 2 + TILE_FRAME_PAD[0], TILE_FRAME_PAD[1]] },
    };
  }
}

export const TILE_FRAME_PAD: [number, number] = [0.42, 0.3];

function groupText(group: Node, value: number): string {
  return `${group.name.slice(group.name.lastIndexOf('.') + 1)}=${Math.round(value)}`;
}

function chunksOf(n: Node, elements: Element[], footprint: readonly [number, number], y: number): Plate[] | undefined {
  const { count } = plates(n);
  if (n.shape.length !== 3 || count < 2) return undefined;
  const { chunk } = plateAxes(n);
  return Array.from({ length: count }, (_, p) => {
    const members = elements.filter((el) => el.index[chunk] === p);
    const xs = members.map((el) => el.x);
    const zs = members.map((el) => el.z);
    const [x0, x1, z0, z1] = [Math.min(...xs), Math.max(...xs), Math.min(...zs), Math.max(...zs)];
    return { x: (x0 + x1) / 2, y, z: (z0 + z1) / 2, w: x1 - x0 + footprint[0] + PLATE_MARGIN, d: z1 - z0 + footprint[1] + PLATE_MARGIN };
  });
}

function drawnAlike(n: Node, source: Node): boolean {
  if (rowsCols(n).join() !== rowsCols(source).join()) return false;
  let alike = true;
  forEachIndex(n.shape, (index) => {
    const at = sourceIndex(n, source, index);
    if (alike && (at === undefined || rowCol(n, index).join() !== rowCol(source, at).join())) alike = false;
  });
  return alike;
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
  if (e.copy !== undefined) return 0;
  if (e.kind === 'tile' && e.table !== undefined) return 1;
  if (e.kind === 'vector' || isLeafLike(e.node)) return 4;
  return 2;
}

export type { Vec3 };
