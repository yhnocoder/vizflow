import type { Node } from '../autograd';
import { batchedCells, explainOf } from '../explain';
import { type StepBrace, type StepModule, type StepsSpec, isLeafLike } from '../steps';
import { forEachIndex, indexOf } from '../tensor';
import { type Anchor, type Block, type BoardGeometry, type Element, type Plate, type Slab, type Vec3, isLayoutOp, labelBehind, plates, rowCol, rowsCols, sourceIndex } from './layout';
import { type BoardRef, Columns, type Entry, type Kind, LEVEL_ROW, type Level, STACK_GAP, STREAM_SPACING, TITLE, position } from './columns';
import type { SceneModule } from './modules';
import { DEFAULT_LAYOUT_OPTIONS, type LayoutOptions, type LevelScheme, type Place, schemeLevels } from './schemes';

export { DEFAULT_LAYOUT_OPTIONS, ITERATION_MODES, LEVEL_SCHEMES, type LayoutOptions, type LevelScheme, PREPARE_MODES, SPAN_MODES } from './schemes';
export { BLOCK_HALF, BOARD_GAP, CHIP_ROOM, COLUMN_BREAK, DISC_GROUP_SIDE, DISC_NEAR_SIDE, DISC_SIDE, LABEL_CHAR, LEVEL_ROW, SPACING_MAX, SPACING_MIN, SPACING_SPAN, STACK_GAP, STREAM_SPACING, STREAM_X, texWidth } from './columns';

export const LEVEL_FRONT = 2.9;
export const TILE_BACK = 1.3;
export const TILE_ROW = 0.72;
export const TILE_ROW_DENSE = 0.55;
export const TILE_DENSE_ROWS = 5;
export const VECTOR_BACK = 1.35;
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
  afterBoard: 4.2,
  moduleAfterBoard: 4.6,
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
  chips: Chip[];
  braces: { z: number; cx: number; cz: number };
  lo: number;
  top: number;
  sourceOf: ReadonlyMap<Node, Node>;
  drawnAs: ReadonlyMap<Node, Node>;
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
  dump(): string;
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
const COPY_INDEX = /\[\d+\]/g;
export const BRACE_TITLE_JOIN = ' · ';

const zPlate = (r: number) => LEVEL_FRONT + r * LEVEL_ROW;
const zTile = (r: number, rows: number) => LEVEL_FRONT - TILE_BACK - (rows - 1 - r) * (rows > TILE_DENSE_ROWS ? TILE_ROW_DENSE : TILE_ROW);
const zVector = LEVEL_FRONT - VECTOR_BACK;

const list = (nodes: Node | Node[]): Node[] => (Array.isArray(nodes) ? nodes : [nodes]);

export interface LevelSpec {
  scheme?: LevelScheme;
  options?: LayoutOptions;
  lines: Node[][];
  sides: { node: Node; side: 'with' | 'left' | 'right'; of: Node }[];
  merges: { left: Node[]; right: Node[] }[];
  places?: Place[];
  iterations?: ReadonlyMap<Node, string>;
}


export function levelPlan(spec: StepsSpec, layout: LevelSpec): LevelPlan {
  return new Plan(spec, layout);
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
  private readonly columns = new Set<Node>();
  private readonly boards = new Map<Node, BoardRef>();
  private readonly tiles = new Map<Node, Entry>();
  private readonly kernels = new Map<Node, Node>();
  private readonly kernelTiles = new Map<Node, Entry>();
  private readonly stacked = new Map<Node, Node>();
  private readonly copies = new Map<Entry, Entry>();
  private readonly tables = new Map<Node, Node>();
  private readonly rowGroups = new Map<Node, Node>();
  private readonly labelledGroups = new Set<Node>();
  private readonly slabbed = new Map<Node, Node>();
  private readonly groupWeights = new Map<Node, Node[]>();
  private readonly braces: StepBrace[];
  private readonly cols: Columns;

  constructor(
    spec: StepsSpec,
    private readonly layout: LevelSpec,
  ) {
    this.braces = spec.braces ?? [];
    this.stepModules = [...(spec.modules ?? [])];
    for (const m of this.stepModules) this.byId.set(m.id, m);
    this.visible = [...new Set(spec.forward.flatMap((s) => list(s.show)))].sort((a, b) => a.id - b.id);
    this.computed = new Set(this.visible.filter((n) => !isLeafLike(n)));
    this.findSources();
    this.findBoards();
    this.findRowGroups();
    this.buildLevels();
    this.placeSides();
    this.findCopies();
    this.cols = new Columns({
      levels: this.levels,
      entries: this.entries,
      options: this.layout.options ?? DEFAULT_LAYOUT_OPTIONS,
      sides: this.layout.sides,
      merges: this.layout.merges,
      boards: this.boards,
      tiles: this.tiles,
      kernelTiles: this.kernelTiles,
      stacked: this.stacked,
      copies: this.copies,
      tables: this.tables,
      slabbed: this.slabbed,
      drawnOn: this.drawnOn,
      computed: this.computed,
      entryOf: this.entryOf,
      sourceOf: this.sourceOf,
      source: (n) => this.source(n),
      consumers: (n) => this.consumers(n),
      prepared: (n) => this.constant(n),
      labelled: (n) => this.tables.has(n) || this.selectLabel(n) !== undefined,
    });
    this.cols.run();
    this.markQuiet();
    this.buildModules();
  }

  private markQuiet(): void {
    const norms = new Set(this.stepModules.filter((m) => !this.stepModules.some((c) => c.parent === m.id) && m.nodes.some((n) => this.computed.has(n) && this.isDisc(n))));
    for (const e of this.entries) {
      if (e.kind === 'vector') e.quiet = true;
      if (this.computed.has(e.node) && [...norms].some((m) => m.nodes.includes(e.node) && !this.isOutput(m, e.node))) e.quiet = true;
    }
  }

  private findSources(): void {
    const shown = new Set(this.visible);
    const operands = new Set([...this.computed].filter((u) => u.kind === 'op' && (u.op === 'matmul' || u.op === 'select_matmul')).map((u) => u.inputs[1]));
    const nodes = this.visible[0]?.graph.nodes ?? [];
    const users = new Map<Node, Node[]>();
    for (const n of nodes) for (const i of n.inputs) users.set(i, [...(users.get(i) ?? []), n]);
    const undone = (n: Node) => isTranspose(n) && !isLeafLike(n) && (users.get(n) ?? []).length > 0 && users.get(n)!.every((u) => isTranspose(u) && sameAxes(u, n));
    const roundTrip = new Set(nodes.filter(undone).flatMap((n) => [n, ...users.get(n)!]));
    for (const n of nodes) {
      if (n.kind === 'op' && n.op === 'unsqueeze' && n.shape.length === 2 && n.shape[1] === 1) this.columns.add(n.inputs[0]);
      if (!isLayoutOp(n) || (shown.has(n) && !roundTrip.has(n)) || isLeafLike(n) || operands.has(n)) continue;
      const source = this.source(n.inputs[0]);
      this.sourceOf.set(n, source);
      if (n.shape.length >= 3 && drawnAlike(n, source)) this.drawnOn.set(source, [...(this.drawnOn.get(source) ?? []), n]);
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
      if (u.op === 'causal_conv1d') {
        this.kernels.set(u, u.inputs[1]);
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
    return group !== undefined && e.table === undefined && this.labelledGroups.has(e.node) && e.rows === group.shape[0];
  }

  private shapeOf(n: Node): [number, number] {
    if (this.slabbed.has(n)) return [n.shape[1], n.shape[2]];
    if (n.shape.length === 0) return [1, 1];
    if (n.shape.length === 1) {
      const main = n.inputs[0];
      const column = (u: Node) => u.shape.length === 1 && u.shape[0] === n.shape[0] && !isLeafLike(u) && this.shapeOf(u)[1] === 1;
      if (isLeafLike(n)) return [1, n.shape[0]];
      return this.columns.has(n) || (main !== undefined && ((main.shape.length === 2 && main.shape[0] === n.shape[0]) || n.inputs.some(column))) ? [n.shape[0], 1] : [1, n.shape[0]];
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
      !this.boards.has(n) && !isLeafLike(n) && n.inputs.length > 0 &&
      (n.shape.length === 2 || (n.shape.length >= 3 && plates(n).width === 1)) &&
      n.shape[n.shape.length - 1] === 1 &&
      this.consumers(n).some((u) => {
        const main = u.inputs[0];
        if (this.source(main) === n) return u.shape.join() === main.shape.join() && this.isDisc(u);
        const used = u.inputs.slice(1).find((i) => this.source(i) === n);
        if (used === undefined) return false;
        const broadcasts = explainOf(u.op as string)?.pattern === 'elementwise' && used.shape.length <= u.shape.length && used.shape.every((size, i) => size === 1 || size === u.shape[u.shape.length - used.shape.length + i]);
        return main.shape.join() === u.shape.join() && broadcasts && used.shape[used.shape.length - 1] === 1 && u.shape[u.shape.length - 1] > 1 && this.shapeOf(u)[1] > cols;
      });
    this.discMemo.set(n, result);
    return result;
  }

  private entry(node: Node, level: Level, key = `node:${node.id}`): Entry {
    const [rows, cols] = this.shapeOf(node);
    const leaf = isLeafLike(node);
    let kind: Kind = 'plate';
    if (node.shape.length === 0) kind = 'box';
    else if (leaf && node.shape.length === 1) kind = 'vector';
    else if (this.isDisc(node)) kind = 'disc';
    const e: Entry = { key, node, kind, level, rows, cols, track: undefined, host: undefined, offsets: [], xs: undefined, title: 'none', quiet: false, table: undefined, transposed: false, zShift: 0 };
    level.entries.push(e);
    this.entries.push(e);
    if (!this.entryOf.has(node) || key === `node:${node.id}`) this.entryOf.set(node, e);
    return e;
  }

  private newLevel(): Level {
    const level: Level = { entries: [], module: null, y: 0 };
    this.levels.push(level);
    return level;
  }

  private buildLevels(): void {
    const lines = this.layout.lines.map((line) => line.filter((n) => this.computed.has(n) && !this.sourceOf.has(n)));
    const scheme = this.layout.scheme ?? 'lines';
    if (scheme === 'lines') this.buildLineLevels(lines);
    else this.buildSchemeLevels(scheme, [...new Set(lines.flat())].sort((a, b) => a.id - b.id));
    for (const [view, source] of this.sourceOf) {
      const e = this.entryOf.get(source);
      if (e !== undefined) this.entryOf.set(view, e);
    }
    for (const level of this.levels) level.module = this.levelModule(level);
  }

  private buildLineLevels(lines: Node[][]): void {
    const lineOf = new Map<Node, number>();
    lines.forEach((line, i) => line.forEach((n) => lineOf.set(n, i)));
    const moveTo = new Map<Node, Node>();
    for (const board of this.boards.values()) {
      if (!board.linear) {
        if (this.computed.has(board.b) && this.consumers(board.b).length === 1) moveTo.set(board.b, board.op);
        continue;
      }
      const a = board.a;
      if (!this.computed.has(a)) continue;
      const users = this.consumers(a);
      if (lineOf.get(a) === lineOf.get(board.op)! - 1 && users.every((u) => lineOf.get(u) === lineOf.get(board.op) && this.boards.get(u)?.a === a)) moveTo.set(a, board.op);
    }
    this.moveKernels(moveTo);
    for (const line of lines) {
      const own = line.filter((n) => !moveTo.has(n) && !this.entryOf.has(n));
      if (own.length === 0) continue;
      const depth = new Map<Node, number>();
      for (const n of own) depth.set(n, Math.max(0, ...n.inputs.map((i) => this.source(i)).filter((i) => depth.has(i)).map((i) => depth.get(i)! + 1)));
      const levels = Array.from({ length: Math.max(...depth.values()) + 1 }, () => this.newLevel());
      for (const n of own) this.placeNode(n, levels[depth.get(n)!], moveTo);
    }
  }

  private buildSchemeLevels(scheme: Exclude<LevelScheme, 'lines'>, nodes: Node[]): void {
    const members = new Set(nodes);
    const predsOf = new Map<Node, Node[]>();
    const preds = (n: Node): Node[] => {
      const known = predsOf.get(n);
      if (known !== undefined) return known;
      const out = new Set<Node>();
      const seen = new Set<Node>();
      const visit = (i: Node) => {
        const s = this.source(i);
        if (seen.has(s)) return;
        seen.add(s);
        if (members.has(s)) out.add(s);
        else if (!isLeafLike(s)) s.inputs.forEach(visit);
      };
      n.inputs.forEach(visit);
      out.delete(n);
      predsOf.set(n, [...out]);
      return predsOf.get(n)!;
    };
    const index = schemeLevels(scheme, {
      nodes,
      preds,
      constant: (n) => this.constant(n) || (n.kind === 'op' && n.op === 'alias'),
      ...this.schemeScopes(nodes),
      pack: (n) => {
        const group = (n.attrs as { group?: unknown } | undefined)?.group;
        return n.kind === 'op' && typeof group === 'string' ? `${n.op}:${group}` : undefined;
      },
      rides: (n) => this.ridesBoards(n),
      starts: (n) => this.startsTrack(n, preds),
      late: (n) => n.kind === 'op' && n.op === 'alias',
      iteration: (n) => this.layout.iterations?.get(n),
      places: this.layout.places ?? [],
    }, this.layout.options);
    const moveTo = new Map<Node, Node>();
    for (const board of this.boards.values()) {
      if (!board.linear) {
        if (this.computed.has(board.b) && this.consumers(board.b).length === 1) moveTo.set(board.b, board.op);
        continue;
      }
      if (members.has(board.a) && index.get(board.a) === index.get(board.op)) moveTo.set(board.a, board.op);
    }
    this.moveKernels(moveTo);
    const used = [...new Set(nodes.filter((n) => !moveTo.has(n)).map((n) => index.get(n)!))].sort((p, q) => p - q);
    const levelAt = new Map(used.map((l) => [l, this.newLevel()]));
    for (const n of nodes) if (!moveTo.has(n) && !this.entryOf.has(n)) this.placeNode(n, levelAt.get(index.get(n)!)!, moveTo);
  }

  private readonly constantOf = new Map<Node, boolean>();

  private constant(n: Node): boolean {
    const known = this.constantOf.get(n);
    if (known !== undefined) return known;
    this.constantOf.set(n, true);
    const value = n.kind !== 'label' && n.kind !== 'input' && n.inputs.every((i) => this.constant(i));
    this.constantOf.set(n, value);
    return value;
  }

  private startsTrack(n: Node, preds: (n: Node) => Node[]): boolean {
    const board = this.boards.get(n);
    if (board !== undefined) return this.ridesBoards(board.a);
    const main = n.inputs[0];
    return main === undefined || !preds(n).includes(this.source(main));
  }

  private ridesBoards(n: Node): boolean {
    const users = this.consumers(n);
    return this.computed.has(n) && users.length > 0 && users.every((u) => this.boards.get(u)?.linear && this.boards.get(u)?.a === n);
  }

  private schemeScopes(nodes: Node[]): { scope(n: Node): string | null; parent(id: string): string | null } {
    const members = new Set(nodes);
    const own = (list: Node[]) => new Set(list.filter((n) => members.has(n)));
    const modules = this.stepModules.map((m) => ({ id: m.id, nodes: own(m.nodes) })).filter((m) => m.nodes.size > 0);
    const crosses = (p: Set<Node>, q: Set<Node>) => {
      const shared = [...p].filter((n) => q.has(n)).length;
      return shared > 0 && shared < p.size && shared < q.size;
    };
    const same = (p: Set<Node>, q: Set<Node>) => p.size === q.size && [...p].every((n) => q.has(n));
    const groups = [...modules];
    for (const b of this.braces) {
      const set = own(b.nodes);
      if (b.module !== undefined || set.size === 0 || groups.some((g) => crosses(set, g.nodes) || same(set, g.nodes))) continue;
      groups.push({ id: `brace:${b.id}`, nodes: set });
    }
    type Group = (typeof groups)[number];
    const inside = (inner: Group, outer: Group) => inner !== outer && inner.nodes.size < outer.nodes.size && [...inner.nodes].every((n) => outer.nodes.has(n));
    const smallest = (found: Group[]) => found.reduce<Group | undefined>((best, group) => (best === undefined || group.nodes.size < best.nodes.size ? group : best), undefined);
    const scopeOf = new Map(nodes.map((n) => [n, smallest(groups.filter((group) => group.nodes.has(n)))?.id ?? null]));
    const parentOf = new Map(groups.map((group) => [group.id, smallest(groups.filter((outer) => inside(group, outer)))?.id ?? null]));
    return { scope: (n) => scopeOf.get(n) ?? null, parent: (id) => parentOf.get(id) ?? null };
  }

  private moveKernels(moveTo: Map<Node, Node>): void {
    for (const [op, kernel] of this.kernels) if (this.computed.has(kernel) && this.consumers(kernel).length === 1) moveTo.set(kernel, op);
  }

  private placeNode(n: Node, level: Level, moveTo: Map<Node, Node>): void {
    for (const [moved, target] of moveTo) if (target === n && !this.entryOf.has(moved) && this.boards.get(n)?.a === moved) this.entry(moved, level);
    const kernel = this.kernels.get(n);
    if (kernel !== undefined) {
      const tile = this.entry(kernel, level, this.entryOf.has(kernel) ? `kernel:${n.id}` : `node:${kernel.id}`);
      tile.kind = 'tile';
      tile.transposed = true;
      [tile.rows, tile.cols] = [tile.cols, tile.rows];
      this.kernelTiles.set(n, tile);
    }
    const board = this.boards.get(n);
    const twin = board === undefined ? undefined : [...this.tiles].find(([op, t]) => t.level === level && t.node === board.b && this.boards.get(op)!.batched === board.batched && this.boards.get(op)!.a !== board.a);
    if (twin !== undefined) {
      this.tiles.set(n, twin[1]);
      this.stacked.set(n, twin[0]);
    } else if (board !== undefined) {
      const tile = this.entry(board.b, level, this.entryOf.has(board.b) ? `tile:${n.id}` : `node:${board.b.id}`);
      tile.kind = 'tile';
      this.tiles.set(n, tile);
    }
    const table = this.tables.get(n);
    if (table !== undefined) {
      const t = this.entry(table, level);
      t.kind = 'tile';
      t.table = n;
    }
    this.entry(n, level);
  }

  private levelModule(level: Level): string | null {
    const own = level.entries.filter((e) => this.computed.has(e.node)).map((e) => e.node);
    if (own.length === 0) return null;
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
      const placed = this.layout.sides.find((s) => s.node === n);
      if (placed !== undefined) {
        const of = this.entryOf.get(placed.of);
        if (of !== undefined) this.entry(n, of.level);
        continue;
      }
      const user = this.consumers(n)[0];
      if (user === undefined) continue;
      const main = user.inputs.find((i) => this.source(i) !== n) ?? user;
      const host = this.entryOf.get(main);
      if (host === undefined) continue;
      const target = this.entryOf.get(user);
      const apart = target !== undefined && this.levels.indexOf(target.level) - this.levels.indexOf(host.level) > 1;
      this.entry(n, apart ? target.level : host.level);
    }
    for (const { node, of } of this.layout.sides) {
      if (node.kind === 'op') continue;
      const drawn = this.entryOf.get(node) ?? this.visible.map((v) => this.entryOf.get(v)).find((e) => e !== undefined && transposed.has(node) && e.node.inputs[0] === node);
      if (drawn !== undefined && drawn.level !== this.entryOf.get(of)?.level) throw new Error(`place cannot move '${node.name}': it is drawn as the matrix of the matmul that uses it`);
    }
  }

  private drawnA(board: BoardRef): Entry {
    return this.entryOf.get(board.a)!;
  }

  private findCopies(): void {
    const width = this.layout.options?.width ?? Infinity;
    if ((this.layout.scheme ?? 'lines') === 'lines' || !Number.isFinite(width)) return;
    const familyOf = (e: Entry) => `${e.kind}:${e.node.name.replace(COPY_INDEX, '[]')}`;
    const indexOf = (e: Entry) => [...e.node.name.matchAll(COPY_INDEX)].map((m) => m[0]).join('');
    const wide = new Set<string>();
    const byLevel = new Map<Level, Map<string, Entry[]>>();
    for (const level of this.levels) {
      const families = new Map<string, Entry[]>();
      for (const e of level.entries) if (/\[\d+\]/.test(e.node.name)) families.set(familyOf(e), [...(families.get(familyOf(e)) ?? []), e]);
      byLevel.set(level, families);
      if (level.entries.filter((e) => this.computed.has(e.node) && e.kind !== 'tile').length <= width) continue;
      for (const [family, members] of families) if (new Set(members.map(indexOf)).size > 1) wide.add(family);
    }
    for (const families of byLevel.values()) {
      for (const [family, members] of families) {
        if (!wide.has(family)) continue;
        const ordered = [...members].sort((p, q) => indexOf(p).localeCompare(indexOf(q), undefined, { numeric: true }));
        ordered.slice(1).forEach((e) => this.copies.set(e, ordered[0]));
      }
    }
    const step = Math.max(0, ...[...this.copies.values()].map((e) => e.rows * LEVEL_ROW)) + STACK_GAP;
    for (const families of byLevel.values()) {
      for (const members of families.values()) {
        const ordered = members.filter((e) => this.copies.has(e)).sort((p, q) => indexOf(p).localeCompare(indexOf(q), undefined, { numeric: true }));
        ordered.forEach((e, k) => (e.zShift = -(k + 1) * step));
      }
    }
  }

  private braceLevels(brace: StepBrace): Level[] {
    const own = new Set(brace.nodes.flatMap((n) => this.entryOf.get(n)?.level ?? []));
    return this.levels.filter((l) => own.has(l));
  }

  private buildModules(): void {
    const spans = new Map<string, SceneModule>();
    for (const brace of this.braces) {
      const levels = this.braceLevels(brace);
      if (levels.length === 0) continue;
      const first = this.levels.indexOf(levels[0]);
      const last = this.levels.indexOf(levels[levels.length - 1]);
      const span = `${brace.depth}:${first}:${last}`;
      const twin = spans.get(span);
      if (twin !== undefined && !twin.nodes.some((n) => brace.nodes.includes(n))) {
        if (!twin.title.split(BRACE_TITLE_JOIN).includes(brace.title)) twin.title = `${twin.title}${BRACE_TITLE_JOIN}${brace.title}`;
        if (brace.subtitle !== undefined && !(twin.subtitle ?? '').split(BRACE_TITLE_JOIN).includes(brace.subtitle)) twin.subtitle = twin.subtitle === undefined ? brace.subtitle : `${twin.subtitle}${BRACE_TITLE_JOIN}${brace.subtitle}`;
        twin.nodes = [...twin.nodes, ...brace.nodes];
        continue;
      }
      const gap = this.levels.slice(first, last + 1).find((l) => !levels.includes(l));
      if (gap !== undefined && !['asap', 'alap'].includes(this.layout.scheme ?? 'lines')) {
        const between = gap.entries.map((e) => e.node.name).join(', ');
        throw new Error(`steps: the brace of '${brace.id}' (${brace.title}) frames tensors that are not next to each other; ${between} lies between them`);
      }
      const nested = this.braces.some((other) => other !== brace && other.nodes.length < brace.nodes.length && other.nodes.every((n) => brace.nodes.includes(n)));
      const chosen = brace.side;
      const side = chosen ?? (brace.depth >= 2 ? 'left' : brace.depth === 0 && nested ? 'outer' : 'right');
      const module: SceneModule = { id: brace.module ?? brace.id, title: brace.title, subtitle: brace.subtitle, nodes: brace.nodes, side, depth: brace.depth };
      if (side === 'left') module.column = Math.max(0, brace.depth - 2);
      if (chosen !== undefined) module.chosen = true;
      this.modules.push(module);
      spans.set(span, module);
    }
  }

  dump(): string {
    const trackOf = (e: Entry) => (e.track === undefined ? '-1' : e.track === this.cols.stream ? 's' : String(e.track.id));
    const entry = (e: Entry) => `${e.node.name}[${e.kind}|${e.offsets.join(',')}]@${trackOf(e)}:${(e.xs ?? []).map((x) => x.toFixed(1)).join(',')}`;
    return this.levels.map((level, i) => `${String(i).padStart(3)} ${level.entries.map(entry).join('  ')}\n`).join('');
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
      if (this.computed.has(e.node)) value = hiddenNode(e.node);
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
      const own = shown.filter((e) => e.kind !== 'tile' && e.kind !== 'vector' && !isLeafLike(e.node));
      return {
        visible: shown.length > 0,
        board: shown.some((e) => e.kind === 'tile' && e.table === undefined),
        tiled: shown.some((e) => e.kind === 'tile'),
        tileRows: Math.max(0, ...shown.filter((e) => e.kind === 'tile' && e.table === undefined).map((e) => e.rows)),
        linear: shown.some((e) => e.kind === 'tile' && isLeafLike(e.node) && e.table === undefined),
        scalar: own.length > 0 && own.every((e) => e.kind === 'disc' || e.kind === 'box'),
        loss: own.some((e) => e.kind === 'box'),
        vector: shown.some((e) => e.kind === 'vector'),
        buffer: shown.some((e) => e.node.kind === 'buffer'),
        module: level.module,
      };
    };
    type Flags = ReturnType<typeof flags>;
    const opens = (f: Flags, prev: Flags) => f.module !== null && f.module !== prev.module && this.chain(f.module).some((m) => m.id === prev.module);
    const gap = (f: Flags, prev: Flags, before: Flags | undefined): number => {
      if (f.scalar) return prev.tiled ? GAP.scalarAfterBoard : f.loss ? GAP.loss : GAP.scalar;
      if (f.board) return !f.linear ? GAP.board : prev.board ? GAP.linearBoard : GAP.ownBoard;
      if (prev.tiled) return opens(f, prev) ? GAP.moduleAfterBoard : GAP.afterBoard;
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
      const outLevel = this.levels.findIndex((l) => l.entries.some((e) => this.isOutput(m, e.node)));
      if (outLevel <= 0) continue;
      const below = [...this.levels.slice(0, outLevel)].reverse().find((l) => flags(l).visible);
      boxY.set(m.id, [(below?.y ?? 0) + FOLD.emptyBelow, this.levels[outLevel].y - FOLD.emptyAbove]);
    }

    const blockOf = new Map<Entry, Block>();
    for (const e of this.entries) blockOf.set(e, this.block(e, hiddenOf(e), ctx.aligned));
    for (const op of this.kernels.keys()) {
      const input = this.entryOf.get(op.inputs[0]);
      const block = input === undefined ? undefined : blockOf.get(input);
      if (block !== undefined && block.plate !== undefined) block.pad = convPad(op, block);
    }
    const blocks = [...blockOf.values()];

    const boards: BoardGeometry[] = [];
    for (const board of this.boards.values()) {
      const out = this.entryOf.get(board.op)!;
      const tile = this.tiles.get(board.op)!;
      const a = this.drawnA(board);
      const siblings = [...this.boards.values()].filter((b) => this.entryOf.get(b.op)?.level === out.level && this.drawnA(b) === a && a.level === out.level).sort((p, q) => p.op.id - q.op.id);
      const at = siblings.findIndex((b) => b === board);
      const previous = at > 0 ? this.tiles.get(siblings[at - 1].op) : undefined;
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
      const keys = inside.length > 0 ? inside : this.entries.filter((e) => this.isOutput(m, e.node) || this.isInput(m, e.node));
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
    for (const brace of this.braces) {
      const fold = brace.module === undefined ? undefined : this.byId.get(brace.module);
      if (fold !== undefined && outerFold(fold.parent) !== undefined) continue;
      const range = fold !== undefined && isFolded(fold) ? boxY.get(fold.id) : spanOf(this.braceLevels(brace));
      if (range !== undefined) ranges.set(brace.module ?? brace.id, range);
    }

    const io: ModuleIO[] = [];
    for (const m of this.stepModules) {
      if (!isFolded(m) || outerFold(m.parent) !== undefined) continue;
      for (const output of m.outputs) {
        const outputEntry = this.entryOf.get(output);
        if (outputEntry === undefined) continue;
        for (const input of m.inputs) if (this.entryOf.has(input)) io.push({ module: m, input, output, outputBlock: blockOf.get(outputEntry)!, split: this.stepModules.some((c) => c.parent === m.id) });
      }
    }

    const chips: Chip[] = [];
    for (const e of this.entries) {
      if (e.kind !== 'plate') continue;
      const block = blockOf.get(e)!;
      const right = Math.max(...(e.xs ?? [0]));
      const table = this.tables.get(e.node);
      const input = table === undefined ? undefined : e.node.inputs.find((i) => i.kind === 'label');
      if (input !== undefined) chips.push({ block, label: input, x: right + CHIP_SIDE, arrow: true });
      const select = this.selectLabel(e.node);
      if (select !== undefined) chips.push({ block, label: select, x: right + CHIP_LABEL_SIDE, arrow: false });
    }
    const streamWidth = Math.max(1, ...this.cols.stream.members.flatMap((m) => m.offsets.map((o) => o + 1)));
    const rows = this.rowsOf();
    return {
      blocks,
      boards,
      boxes,
      ranges,
      io,
      chips,
      braces: { z: Math.max(zPlate(rows - 1), ...this.entries.filter((en) => en.kind === 'plate').flatMap((en) => this.zs(en))) + BRACE_FRONT, cx: position(this.cols.stream, (streamWidth - 1) / 2), cz: zPlate((rows - 1) / 2) },
      lo: 0,
      top,
      sourceOf: this.sourceOf,
      drawnAs: new Map(this.entries.flatMap((e) => (e.node.shape.length >= 3 ? [] : (this.drawnOn.get(e.node) ?? []).filter((v) => v.shape.length >= 3).slice(0, 1).map((v) => [e.node, v] as const)))),
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
    return this.rowZs(e.node, e.rows).map((z) => z + e.zShift);
  }

  private rowsOf(): number {
    return Math.max(1, ...this.entries.filter((e) => e.kind === 'plate').map((e) => e.rows));
  }

  private chunkOf(e: Entry): (col: number) => number {
    const n = e.node;
    const view = n.shape.length >= 3 ? n : this.drawnOn.get(n)?.find((v) => v.shape.length >= 3);
    if (view !== undefined && plates(view).count >= 2) return (col) => Math.floor(col / plates(view).width);
    const runs = runsOf(e.offsets);
    return (col) => runs[col] ?? 0;
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
      const [row, col] = slab ? [index[1], index[2]] : e.transposed ? [index[1], index[0]] : e.rows > 1 && e.cols === 1 ? [offset, 0] : n.shape.length >= 2 ? rowCol(n, index) : [0, offset];
      const z = slab ? this.slabZ(n, index[0], row) : zs[Math.min(row, zs.length - 1)];
      elements.push({ index, row, col, x: xs[Math.min(col, xs.length - 1)], y, z });
    }
    const spacing = e.track?.spacing ?? STREAM_SPACING;
    const footprint: [number, number] =
      e.kind === 'tile'
        ? [spacing >= STREAM_SPACING ? TILE_WIDTH : TILE_WIDTH_NARROW, e.rows > TILE_DENSE_ROWS ? TILE_DEPTH_DENSE : TILE_DEPTH]
        : e.kind === 'disc'
          ? DISC_FOOTPRINT
          : e.kind === 'box'
            ? BOX_FOOTPRINT
            : this.slabbed.has(n)
              ? [STREAM_FOOTPRINT, TILE_DEPTH_DENSE]
              : e.track?.stream === true
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
    const label = e.table === undefined ? undefined : e.table.inputs.find((i) => i.kind === 'label');
    const rowStep = label?.tokens !== undefined && zs.length > 1 ? Math.abs(zs[1] - zs[0]) : 0;
    let title: Anchor;
    if (e.kind === 'tile') title = { x: x0 + TITLE.board[0], y, z: Math.min(...zs) + TITLE.board[1] - rowStep, anchor: 'left' };
    else if (e.kind === 'vector') title = { x: x1 + TITLE.vector[0], y: y + TITLE.vector[1], z: zVector, anchor: 'right' };
    else if (e.title === 'below' || (e.title === 'left' && this.hasGroupLabels(e))) title = { x: (x0 + x1) / 2, y: y + TITLE.below[0], z: front + TITLE.below[1], anchor: 'below' };
    else if (scalar) title = { x: elements[0].x + TITLE.node[0], y: y + TITLE.node[1], z: elements[0].z + TITLE.node[2], anchor: 'right' };
    else if (e.title === 'right') title = { x: x1 + TITLE.right[0], y: y + TITLE.right[1], z: front, anchor: 'right' };
    else title = { x: x0 + TITLE.left[0], y: y + TITLE.left[1], z: front, anchor: 'left' };
    const tokens = aligned.get(n)?.tokens;
    const named = e.kind === 'tile' ? tokens : this.selectLabel(n) !== undefined ? (this.selectLabel(n)!.tokens ?? undefined) : undefined;
    const slabs = this.slabsOf(e, elements, footprint, y);
    const chunks = slabs === undefined && !scalar ? chunksOf(this.chunkOf(e), elements, footprint, y) : undefined;
    const slabGroup = this.slabbed.get(n);
    const slabLabels = slabGroup === undefined ? undefined : slabs!.map((slab, g) => ({ text: groupText(slabGroup, g), col: g, x: x1 + ROW_NAME_SIDE, y: y + ROW_NAME_LIFT, z: slab.plate.z, anchor: 'right' as const }));
    const classLabels = (named ?? []).slice(0, e.cols).map((text, col) =>
      e.kind === 'tile'
        ? { text, col, x: xs[col], y: y + COLUMN_NAME_LIFT, z: Math.min(...zs) - COLUMN_NAME_BACK, anchor: 'above' as const }
        : { text, col, x: xs[col], y: y + COLUMN_NAME_PLATE_LIFT, z: Math.min(...zs) - COLUMN_NAME_PLATE_BACK, anchor: 'above' as const },
    );
    const rowTokens = label?.tokens;
    const rowLabels = slabLabels !== undefined ? slabLabels : rowTokens === undefined ? this.groupLabels(e, x0 - ROW_NAME_SIDE, y, zs) : rowTokens.slice(0, e.rows).map((text, r) => ({ text, col: r, x: x0 - ROW_NAME_SIDE, y: y + ROW_NAME_LIFT, z: zs[r], anchor: 'left' as const }));
    const framed = e.kind === 'plate' && aligned.has(n);
    return {
      node: n,
      key: e.key,
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

function convPad(op: Node, block: Block): Block['pad'] {
  const input = op.inputs[0];
  const [T, C] = input.shape;
  const rows = (op.inputs[1].shape[1] - 1) * ((op.attrs as { dilation?: number }).dilation ?? 1);
  const at = (index: number[]) => {
    const mapped = block.node === input ? index : sourceIndex(input, block.node, index);
    return mapped === undefined ? undefined : block.elements.find((e) => e.index.every((v, d) => v === mapped[d]));
  };
  const cells: Element[] = [];
  for (let c = 0; c < C; c++) {
    const first = at([0, c]);
    if (first === undefined) return undefined;
    const second = T > 1 ? at([1, c]) : undefined;
    const step = second === undefined ? [0, 0, LEVEL_ROW] : [second.x - first.x, 0, second.z - first.z];
    for (let s = -rows; s < 0; s++) cells.push({ index: [s, c], row: s, col: c, x: first.x + s * step[0], y: first.y, z: first.z + s * step[2] });
  }
  if (cells.length === 0) return undefined;
  const [w, d] = block.footprint;
  const xs = cells.map((e) => e.x);
  const zs = cells.map((e) => e.z);
  const plate = { x: (Math.min(...xs) + Math.max(...xs)) / 2, y: block.plate!.y, z: (Math.min(...zs) + Math.max(...zs)) / 2, w: Math.max(...xs) - Math.min(...xs) + w + PLATE_MARGIN, d: Math.max(...zs) - Math.min(...zs) + d + PLATE_MARGIN };
  return { of: input, cells, plate };
}

function groupText(group: Node, value: number): string {
  return `${group.name.slice(group.name.lastIndexOf('.') + 1)}=${Math.round(value)}`;
}

function runsOf(offsets: number[]): number[] {
  const sorted = offsets.map((o, col) => ({ o, col })).sort((p, q) => p.o - q.o);
  const runs: number[] = [];
  let run = 0;
  sorted.forEach(({ o, col }, i) => {
    if (i > 0 && o !== sorted[i - 1].o + 1) run++;
    runs[col] = run;
  });
  return runs;
}

function chunksOf(chunk: (col: number) => number, elements: Element[], footprint: readonly [number, number], y: number): Plate[] | undefined {
  const count = new Set(elements.map((el) => chunk(el.col))).size;
  if (count < 2) return undefined;
  return Array.from({ length: count }, (_, p) => {
    const members = elements.filter((el) => chunk(el.col) === p);
    const xs = members.map((el) => el.x);
    const zs = members.map((el) => el.z);
    const [x0, x1, z0, z1] = [Math.min(...xs), Math.max(...xs), Math.min(...zs), Math.max(...zs)];
    return { x: (x0 + x1) / 2, y, z: (z0 + z1) / 2, w: x1 - x0 + footprint[0] + PLATE_MARGIN, d: z1 - z0 + footprint[1] + PLATE_MARGIN };
  });
}

function isTranspose(n: Node): boolean {
  return n.kind === 'op' && n.op === 'transpose';
}

function sameAxes(p: Node, q: Node): boolean {
  const axes = (n: Node) => {
    const { dim0, dim1 } = n.attrs as { dim0: number; dim1: number };
    const rank = n.shape.length;
    return [(dim0 + rank) % rank, (dim1 + rank) % rank].sort().join();
  };
  return axes(p) === axes(q);
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

export type { Vec3 };
