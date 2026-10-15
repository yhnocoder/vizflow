import type { Node, Size } from '../autograd';
import { batchedCells, explainOf } from '../explain';
import { type StepBrace, type StepModule, type StepsSpec, isLeafLike } from '../steps';
import { forEachIndex, indexOf } from '../tensor';
import { type Anchor, type Block, type BoardGeometry, type Element, type Plate, type Slab, type Vec3, SCALAR_ROW_GAP, isIndexView, isView, labelBehind, plates, rowCol, rowsCols, sourceIndex } from './layout';
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
export const TITLE_ROOM = 3.5;
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

interface Entry {
  key: string;
  node: Node;
  shape: readonly number[];
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
  dz: number;
  original?: Entry;
  ground?: Ground;
}

interface Level {
  entries: Entry[];
  module: string | null;
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

export interface CardBox {
  x: [number, number];
  z: [number, number];
}

export interface ChipCard {
  expand: readonly number[];
  card: CardBox;
  shared?: CardBox;
  registers?: CardBox;
}

export interface KernelGround {
  grids: readonly LevelGrid[];
  y: number;
  hbm?: CardBox;
  chips: ChipCard[];
  blocks: Block[];
}

interface KernelGroup {
  grids: LevelGrid[];
  expand: number[];
  shared: Entry[];
  registers: Entry[];
  fixed: Entry[];
  card?: CardBox;
  sharedBox?: CardBox;
  registersBox?: CardBox;
}

interface Ground {
  grids: LevelGrid[];
  chips: KernelGroup[];
  reads: Entry[];
  writes: Entry[];
  copies: Map<Entry, Entry>;
  level: Level;
  hbm?: CardBox;
}

interface Unit {
  members: Entry[];
  x: [number, number];
  z: [number, number];
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
  kernels: KernelGround[];
  links: { original: Block; copy: Block }[];
  copyFor(node: Node, consumer: Node): Block | undefined;
  braces: { z: number; cx: number; cz: number };
  lo: number;
  top: number;
  sourceOf: ReadonlyMap<Node, Node>;
  leadOf: ReadonlyMap<Node, Node>;
  drawnAs: ReadonlyMap<Node, Node>;
  grids: readonly LevelGrid[];
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

function consecutive(start: number, n: number): number[] {
  return Array.from({ length: n }, (_, i) => start + i);
}

function position(lane: Lane, i: number): number {
  let breaks = 0;
  for (const b of lane.breaks) if (b <= i) breaks++;
  return lane.x0 + i * lane.spacing + breaks * COLUMN_BREAK;
}

function compareIndex(a: readonly number[], b: readonly number[]): number {
  for (let d = 0; d < Math.min(a.length, b.length); d++) if (a[d] !== b[d]) return a[d] - b[d];
  return a.length - b.length;
}

const list = (nodes: Node | Node[]): Node[] => (Array.isArray(nodes) ? nodes : [nodes]);

export interface LevelSpec {
  lines: Node[][];
  sides: { node: Node; side: 'with' | 'left' | 'right'; of: Node }[];
  merges: { left: Node[]; right: Node[] }[];
  slots: Node[][];
  grids: LevelGrid[];
}

export interface LevelGrid {
  module: string;
  names: string[];
  counts: number[];
  real: Size[];
  arrangement: ('expand' | 'stack')[];
  copyOf: ReadonlyMap<Node, number[]>;
  shared: ReadonlySet<Node>;
  registers: ReadonlySet<Node>;
  inputs: readonly Node[];
  outputs: readonly Node[];
  rows: readonly (readonly Node[])[];
}

const ROUND_SUFFIX = /\[\d+\]$/;

export const PLACE_GAP = 1.6;
export const COPY_GAP = 3.0;
export const CARD_GAP = 5.0;
export const CARD_MARGIN = 0.8;
export const CARD_ROW_GAP = 6.0;
export const CARD_COL_GAP = 2.4;
export const CARD_STAGGER = 1.8;
export const PLACE_TITLE_ROOM = 1.2;

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
  private readonly slotOf = new Map<Node, readonly Node[]>();
  private readonly leadOf = new Map<Node, Node>();
  private consumersOf: Map<Node, Node[]> | undefined;
  private readonly drawnOn = new Map<Node, Node[]>();
  private readonly columns = new Set<Node>();
  private readonly boards = new Map<Node, BoardRef>();
  private readonly tiles = new Map<Node, Entry>();
  private readonly tables = new Map<Node, Node>();
  private readonly rowGroups = new Map<Node, Node>();
  private readonly labelledGroups = new Set<Node>();
  private readonly slabbed = new Map<Node, Node>();
  private readonly groupWeights = new Map<Node, Node[]>();
  private readonly lanes: Lane[] = [];
  private readonly braces: StepBrace[];
  private readonly braceNodes = new Map<StepBrace, Node[]>();
  private grounds: Ground[] = [];
  private readonly groundOf = new Map<Node, Ground>();
  private readonly chainAt = new Map<Entry, number>();
  private readonly chainHost = new Map<Entry, Entry>();
  private readonly chains = new Map<Entry, Entry[]>();
  private readonly stepOf = new Map<Node, number>();
  private readonly roundOf = new Map<Node, string>();
  private readonly stream: Lane;

  constructor(
    private readonly spec: StepsSpec,
    private readonly layout: LevelSpec,
  ) {
    this.braces = spec.braces ?? [];
    this.stepModules = [...(spec.modules ?? [])];
    for (const m of this.stepModules) this.byId.set(m.id, m);
    this.visible = [...new Set([...spec.forward.flatMap((s) => list(s.show)), ...(spec.functions ?? []).flatMap((f) => f.steps.flatMap((s) => s.show))])].sort((a, b) => a.id - b.id);
    this.computed = new Set(this.visible.filter((n) => !isLeafLike(n)));
    this.stream = this.lane(STREAM_X, STREAM_SPACING, true);
    this.findSources();
    this.findSlots();
    this.findBoards();
    this.findRowGroups();
    this.buildLevels();
    this.placeSides();
    this.assignColumns();
    this.assignTitles();
    this.makeTitleRoom();
    this.placeApart();
    this.findRounds();
    this.attachChains();
    this.grounds = this.buildGrounds();
    this.assignTitles();
    this.layoutGrounds();
    this.buildModules();
  }

  private findKernels(): KernelGroup[] {
    const tileOut = new Map<Entry, Entry>();
    for (const [op, tile] of this.tiles) {
      const out = this.entryOf.get(op);
      if (out !== undefined) tileOut.set(tile, out);
    }
    const groups = new Map<string, KernelGroup>();
    for (const grid of this.layout.grids) {
      const expandOf = (index: readonly number[]) => index.filter((_, d) => grid.arrangement[d] === 'expand');
      const roleOf = (e: Entry): { role: 'shared' | 'registers'; expand: number[] } | undefined => {
        const members = this.slotOf.get(e.node) ?? [e.node];
        const role = members.some((m) => grid.shared.has(m)) ? 'shared' : members.some((m) => grid.registers.has(m)) ? 'registers' : undefined;
        if (role === undefined) return undefined;
        const index = members.map((m) => grid.copyOf.get(m)).find((i) => i !== undefined) ?? [];
        return { role, expand: expandOf(index) };
      };
      const byExpand = new Map<string, { expand: number[]; shared: Entry[]; registers: Entry[]; fixed: Entry[] }>();
      for (const e of this.entries) {
        const found = roleOf(e);
        if (found === undefined) continue;
        const key = found.expand.join(',');
        const group = byExpand.get(key) ?? byExpand.set(key, { expand: found.expand, shared: [], registers: [], fixed: [] }).get(key)!;
        group[tileOut.has(e) && found.role === 'shared' ? 'fixed' : found.role].push(e);
      }
      for (const group of byExpand.values()) {
        const key = [...group.shared, ...group.registers, ...group.fixed].map((e) => this.entries.indexOf(e)).sort((a, b) => a - b).join(',');
        const known = groups.get(key);
        if (known === undefined) groups.set(key, { grids: [grid], ...group });
        else known.grids.push(grid);
      }
    }
    return [...groups.values()];
  }

  private buildGrounds(): Ground[] {
    const grounds: Ground[] = [];
    for (const group of this.findKernels()) {
      const known = grounds.find((g) => g.grids.length === group.grids.length && g.grids.every((grid, i) => grid === group.grids[i]));
      if (known !== undefined) known.chips.push(group);
      else grounds.push({ grids: group.grids, chips: [group], reads: [], writes: [], copies: new Map(), level: { entries: [], module: null, y: 0 } });
    }
    grounds.forEach((ground, i) => {
      ground.chips.sort((a, b) => compareIndex(a.expand, b.expand));
      for (const grid of ground.grids) for (const n of [...grid.registers, ...grid.shared, ...grid.outputs, ...grid.copyOf.keys()]) this.groundOf.set(n, ground);
      const chipEntries = new Set(ground.chips.flatMap((c) => [...c.shared, ...c.registers, ...c.fixed]));
      const absorbed = new Set([...chipEntries].map((e) => e.level));
      for (const grid of ground.grids) {
        for (const output of grid.outputs) {
          const e = this.entryOf.get(output);
          if (e !== undefined && !chipEntries.has(e) && !ground.writes.includes(e)) ground.writes.push(e);
        }
      }
      for (const grid of ground.grids) {
        for (const input of grid.inputs) {
          const e = this.drawnEntry(input);
          if (e === undefined || this.writtenOnly(grid, input) || chipEntries.has(e) || ground.writes.includes(e) || ground.reads.includes(e) || ground.copies.has(e)) continue;
          if (absorbed.has(e.level) && e.node.kind !== 'op' && e.ground === undefined && e.original === undefined) {
            ground.reads.push(e);
            continue;
          }
          const copy = this.entry(e.node, ground.level, `hbm:${i}:${e.key}`);
          Object.assign(copy, { kind: e.kind, rows: e.rows, cols: e.cols, lane: e.lane, offsets: e.offsets, xs: e.xs === undefined ? undefined : [...e.xs], original: e });
          ground.copies.set(e, copy);
          ground.reads.push(copy);
        }
      }
      const first = Math.min(...[...absorbed].map((l) => this.levels.indexOf(l)));
      this.levels.splice(first, 0, ground.level);
      for (const e of [...chipEntries, ...ground.writes, ...ground.reads]) {
        e.ground = ground;
        if (e.level === ground.level) continue;
        e.level.entries.splice(e.level.entries.indexOf(e), 1);
        ground.level.entries.push(e);
        e.level = ground.level;
      }
    });
    this.levels.splice(0, this.levels.length, ...this.levels.filter((level) => level.entries.length > 0));
    for (const ground of grounds) ground.level.module = this.levelModule(ground.level);
    return grounds;
  }

  private writtenOnly(grid: LevelGrid, input: Node): boolean {
    const outputs = new Set(grid.outputs);
    const users = [...grid.registers, ...grid.shared, ...grid.outputs].filter((u) => u.inputs.includes(input));
    return users.length > 0 && users.every((u) => u.kind === 'op' && u.op === 'scatter_tiles' && u.inputs[0] === input && outputs.has(u) && (u.attrs as { accumulate?: boolean }).accumulate !== true);
  }

  private copyEntry(node: Node, consumer: Node): Entry | undefined {
    const ground = this.groundOf.get(consumer);
    const e = this.drawnEntry(node);
    return ground === undefined || e === undefined ? undefined : ground.copies.get(e);
  }

  private drawnEntry(node: Node): Entry | undefined {
    let n = node;
    while (!this.entryOf.has(n) && n.kind === 'op' && n.op === 'saved') n = n.inputs[0];
    return this.entryOf.get(n);
  }

  private layoutGrounds(): void {
    const main = this.entries.filter((e) => e.ground === undefined && e.xs !== undefined);
    const x0 = main.length === 0 ? STREAM_X : Math.min(...main.flatMap((e) => e.xs!)) - BOX_MARGIN.left;
    const front = main.length === 0 ? LEVEL_FRONT : Math.max(...main.flatMap((e) => this.zs(e))) + BOX_MARGIN.front;
    for (const ground of this.grounds) this.layoutGround(ground, x0, front);
  }

  private layoutGround(ground: Ground, x0: number, front: number): void {
    const boxes: CardBox[] = [];
    for (const e of [...ground.reads, ...ground.writes]) {
      const spacing = e.lane?.spacing ?? STREAM_SPACING;
      if (spacing <= STREAM_SPACING || e.xs === undefined) continue;
      const x0 = e.xs[0];
      e.xs = e.xs.map((x) => x0 + ((x - x0) * STREAM_SPACING) / spacing);
    }
    let left = 0;
    let depth = 0;
    const placed: Entry[][] = [];
    for (const chip of ground.chips) {
      const { shared, registers } = this.chipUnits(chip, ground.grids[0].rows);
      placed.push([...shared, ...registers].flat().flatMap((u) => u.members));
      const top = shared.length === 0 ? undefined : this.placeRows(shared, left, 0, CARD_STAGGER);
      const bottom = registers.length === 0 ? undefined : this.placeRows(registers, left, top === undefined ? 0 : top.z[1] + CARD_ROW_GAP, CARD_STAGGER);
      const right = Math.max(top?.x[1] ?? left, bottom?.x[1] ?? left) + CARD_MARGIN;
      const pad = (z: [number, number]): CardBox => ({ x: [left - CARD_MARGIN, right], z: [z[0] - CARD_MARGIN, z[1] + CARD_MARGIN] });
      if (top !== undefined) chip.sharedBox = pad(top.z);
      if (bottom !== undefined) chip.registersBox = pad(bottom.z);
      const parts = [chip.sharedBox, chip.registersBox].filter((b): b is CardBox => b !== undefined);
      chip.card = { x: [left - 2 * CARD_MARGIN, right + CARD_MARGIN], z: [Math.min(0, ...parts.map((b) => b.z[0])) - CARD_MARGIN, Math.max(0, ...parts.map((b) => b.z[1])) + CARD_MARGIN] };
      boxes.push(chip.card, ...parts);
      depth = Math.max(depth, chip.card.z[1] - CARD_MARGIN);
      left = chip.card.x[1] + COPY_GAP + 2 * CARD_MARGIN;
    }
    const first = ground.chips[0]?.card;
    const limit = first === undefined ? Infinity : first.x[1] - first.x[0];
    const hbmRows = [ground.reads, ground.writes].flatMap((row) => this.packRow(row.map((e) => this.unit([e])), limit));
    const hbm = hbmRows.length === 0 ? undefined : this.placeRows(hbmRows, 0, 0);
    depth = Math.max(depth, hbm?.z[1] ?? 0);
    if (hbm !== undefined) {
      const shift = hbm.x[1] + CARD_MARGIN + CARD_GAP - Math.min(...boxes.map((b) => b.x[0]));
      for (const box of boxes) box.x = [box.x[0] + shift, box.x[1] + shift];
      for (const e of placed.flat()) e.xs = e.xs?.map((x) => x + shift);
    }
    const writeRows = hbmRows.filter((row) => row.some((u) => ground.writes.includes(u.members[0])));
    if (hbm !== undefined && hbm.z[1] < depth && writeRows.length < hbmRows.length) {
      const lift = depth - hbm.z[1];
      for (const row of writeRows) for (const unit of row) for (const e of unit.members) e.dz += lift;
      hbm.z[1] = depth;
    }
    if (hbm !== undefined) {
      ground.hbm = { x: [hbm.x[0] - CARD_MARGIN, hbm.x[1] + CARD_MARGIN], z: [-CARD_MARGIN, depth + CARD_MARGIN] };
      boxes.push(ground.hbm);
    }
    const back = Math.min(...boxes.map((b) => b.z[0]));
    const near = Math.max(...boxes.map((b) => b.z[1]));
    if (ground.hbm !== undefined) ground.hbm.z = [back, near];
    for (const chip of ground.chips) if (chip.card !== undefined) chip.card.z = [back, near];
    const dx = x0 - Math.min(...boxes.map((b) => b.x[0]));
    const dz = front - near;
    for (const box of boxes) {
      box.x = [box.x[0] + dx, box.x[1] + dx];
      box.z = [box.z[0] + dz, box.z[1] + dz];
    }
    for (const e of ground.level.entries) {
      if (e.ground !== ground) continue;
      e.xs = e.xs?.map((x) => x + dx);
      e.dz += dz;
    }
  }

  private unit(members: Entry[]): Unit {
    const x: [number, number] = [Infinity, -Infinity];
    const z: [number, number] = [Infinity, -Infinity];
    for (const e of members) {
      const [w, d] = this.footprintOf(e);
      const xs = e.xs ?? [0];
      const zs = this.zs(e);
      x[0] = Math.min(x[0], Math.min(...xs) - w / 2 - (e.title === 'left' ? TITLE_ROOM : 0));
      x[1] = Math.max(x[1], Math.max(...xs) + w / 2);
      z[0] = Math.min(z[0], Math.min(...zs) - d / 2);
      z[1] = Math.max(z[1], Math.max(...zs) + d / 2 + (e.title === 'below' ? TITLE.below[1] : 0));
    }
    return { members, x, z };
  }

  private packRow(units: Unit[], limit: number): Unit[][] {
    const rows: Unit[][] = [];
    let width = Infinity;
    for (const unit of units) {
      const w = unit.x[1] - unit.x[0];
      if (width + CARD_COL_GAP + w > limit) {
        rows.push([unit]);
        width = w;
      } else {
        rows[rows.length - 1].push(unit);
        width += CARD_COL_GAP + w;
      }
    }
    return rows;
  }

  private placeRows(rows: Unit[][], x0: number, z0: number, stagger = 0): { x: [number, number]; z: [number, number] } {
    let z = z0;
    let right = x0;
    for (const row of rows) {
      const lo = Math.min(...row.map((u) => u.z[0]));
      const shift = row.length > 1 ? stagger : 0;
      const hi = Math.max(...row.map((u) => u.z[1])) + shift;
      let x = x0;
      for (const [k, unit] of row.entries()) {
        const dx = x - unit.x[0];
        for (const e of unit.members) {
          e.xs = e.xs?.map((v) => v + dx);
          e.dz += z - lo + (k % 2) * shift;
        }
        x += unit.x[1] - unit.x[0] + CARD_COL_GAP;
      }
      right = Math.max(right, x - CARD_COL_GAP);
      z += hi - lo + CARD_ROW_GAP;
    }
    return { x: [x0, right], z: [z0, z - CARD_ROW_GAP] };
  }

  private chipUnits(chip: KernelGroup, rows: readonly (readonly Node[])[]): { shared: Unit[][]; registers: Unit[][] } {
    const own = new Set([...chip.shared, ...chip.registers, ...chip.fixed]);
    const listed = rows.map((row) => [...new Set(row.flatMap((n) => (this.sourceOf.has(n) ? [] : (this.entryOf.get(n) ?? []))))].filter((e) => own.has(e) && e.kind !== 'tile'));
    const sharedSet = new Set(chip.shared);
    const sharedRows = listed.map((row) => row.filter((e) => sharedSet.has(e))).filter((row) => row.length > 0);
    const registerRows = listed.map((row) => row.filter((e) => !sharedSet.has(e))).filter((row) => row.length > 0);
    const pinned = new Map<Entry, number>();
    registerRows.forEach((row, r) => row.forEach((e) => pinned.set(e, r)));
    const sharedPlaced = new Set(sharedRows.flat());
    const sharedLeft = chip.shared.filter((e) => e.kind !== 'tile' && !sharedPlaced.has(e)).sort((a, b) => a.node.id - b.node.id);
    if (sharedLeft.length > 0) {
      if (sharedRows.length === 0) sharedRows.push([]);
      sharedRows[0].push(...sharedLeft);
    }
    const registers = new Set([...chip.registers, ...chip.fixed]);
    const boardOf = new Map<Entry, Entry>();
    for (const [op, tile] of this.tiles) {
      const out = this.entryOf.get(op);
      if (out !== undefined && registers.has(tile) && registers.has(out)) boardOf.set(tile, out);
    }
    const hostOf = (e: Entry): Entry | undefined => {
      const host = this.chainHost.get(e);
      return host !== undefined && !pinned.has(e) && registers.has(host) && !sharedSet.has(host) ? host : undefined;
    };
    const attachedTo = (e: Entry) => boardOf.get(e) ?? hostOf(e);
    const roots = [...registers].filter((e) => attachedTo(e) === undefined);
    const inChip = (n: Node) => this.groundOf.has(n);
    const deps = (e: Entry): Entry[] => {
      const found = new Set<Entry>();
      for (const m of this.slotOf.get(e.node) ?? [e.node]) {
        const round = this.roundOf.get(m);
        const seen = new Set<Node>();
        const visit = (n: Node) => {
          for (const i of n.inputs) {
            if (seen.has(i)) continue;
            seen.add(i);
            const r = this.roundOf.get(i);
            if (round !== undefined && r !== undefined && r !== round) continue;
            const f = this.entryOf.get(i);
            if (f === e) continue;
            if (f !== undefined && registers.has(f) && f.kind !== 'tile') found.add(f);
            else if ((f === undefined || (registers.has(f) && f.kind === 'tile')) && inChip(i)) visit(i);
          }
        };
        visit(m);
      }
      return [...found];
    };
    const rowOf = new Map<Entry, number>();
    const visiting = new Set<Entry>();
    const row = (e: Entry): number => {
      const attached = attachedTo(e);
      if (attached !== undefined) return row(attached);
      const known = rowOf.get(e) ?? pinned.get(e);
      if (known !== undefined) return known;
      if (visiting.has(e)) return 0;
      visiting.add(e);
      const value = Math.max(0, ...deps(e).map((d) => row(d) + (pinned.size > 0 ? 0 : 1)));
      visiting.delete(e);
      rowOf.set(e, value);
      return value;
    };
    const unitOf = (root: Entry): Unit => {
      const members = [root, ...[...registers].filter((e) => e !== root && boardOf.get(e) === root)];
      const chain = (this.chains.get(root) ?? []).filter((e) => hostOf(e) === root);
      return this.unit([...members, ...chain]);
    };
    const numbers = [...new Set(roots.map(row))].sort((a, b) => a - b);
    const registerUnits = numbers.map((r) => {
      const inRow = roots.filter((e) => row(e) === r);
      const order = (e: Entry) => {
        const at = registerRows[r]?.indexOf(e) ?? -1;
        return at >= 0 ? at : Infinity;
      };
      inRow.sort((a, b) => order(a) - order(b) || a.node.id - b.node.id);
      return inRow.map(unitOf);
    });
    return { shared: sharedRows.map((r) => r.map((e) => this.unit([e]))), registers: registerUnits };
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
      if (n.kind === 'op' && n.op === 'unsqueeze' && n.shape.length === 2 && n.shape[1] === 1) this.columns.add(n.inputs[0]);
      if (!isView(n) || (shown.has(n) && !isIndexView(n)) || isLeafLike(n) || operands.has(n)) continue;
      const source = this.source(n.inputs[0]);
      this.sourceOf.set(n, source);
      if (n.shape.length >= 3 && drawnAlike(n, source)) this.drawnOn.set(source, [...(this.drawnOn.get(source) ?? []), n]);
    }
  }

  private source(n: Node): Node {
    return this.sourceOf.get(n) ?? n;
  }

  private findSlots(): void {
    for (const members of this.layout.slots) {
      const lead = members.find((m) => this.computed.has(m) && !this.sourceOf.has(m));
      if (lead === undefined) continue;
      const odd = members.find((m) => m.shape.length !== lead.shape.length);
      if (odd !== undefined) throw new Error(`levels: '${odd.name}' and '${lead.name}' share a position but have ranks ${odd.shape.length} and ${lead.shape.length}`);
      for (const m of members) {
        this.slotOf.set(m, members);
        if (m !== lead && !this.sourceOf.has(m)) this.leadOf.set(m, lead);
      }
    }
  }

  private lead(n: Node): Node {
    return this.leadOf.get(n) ?? n;
  }

  private slotShape(n: Node): readonly number[] {
    const members = this.slotOf.get(n);
    if (members === undefined) return n.shape;
    return n.shape.map((_, d) => Math.max(...members.map((m) => m.shape[d])));
  }

  private isOutput(m: StepModule, n: Node): boolean {
    return m.outputs.some((o) => this.source(o) === n);
  }

  private isInput(m: StepModule, n: Node): boolean {
    return m.inputs.some((i) => this.source(i) === n);
  }

  private consumers(n: Node): readonly Node[] {
    if (this.consumersOf === undefined) {
      this.consumersOf = new Map();
      for (const u of this.computed) {
        for (const source of new Set(u.inputs.map((i) => this.source(i)))) {
          const list = this.consumersOf.get(source);
          if (list === undefined) this.consumersOf.set(source, [u]);
          else list.push(u);
        }
      }
    }
    return this.consumersOf.get(n) ?? [];
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
      if (u.kind !== 'op' || this.leadOf.has(u)) continue;
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
    return group !== undefined && e.table === undefined && this.labelledGroups.has(e.node) && e.rows === group.shape[0];
  }

  private shapeOf(n: Node, shape: readonly number[] = n.shape): [number, number] {
    if (this.slabbed.has(n)) return [shape[1], shape[2]];
    if (shape.length === 0) return [1, 1];
    if (shape.length === 1) {
      const main = n.inputs[0];
      const column = (u: Node) => u.shape.length === 1 && u.shape[0] === n.shape[0] && !isLeafLike(u) && this.shapeOf(u)[1] === 1;
      if (isLeafLike(n)) return [1, shape[0]];
      return this.columns.has(n) || (main !== undefined && ((main.shape.length === 2 && main.shape[0] === n.shape[0]) || n.inputs.some(column))) ? [shape[0], 1] : [1, shape[0]];
    }
    return rowsCols(n, shape);
  }

  private discs: Set<Node> | undefined;

  private isDisc(n: Node): boolean {
    this.discs ??= this.findDiscs();
    return (this.slotOf.get(n) ?? [n]).some((m) => this.discs!.has(m));
  }

  private findDiscs(): Set<Node> {
    const sameShape = (a: Node, b: Node) => a.shape.join() === b.shape.join();
    const sameDrawn = (a: Node, b: Node) => this.shapeOf(a).join() === this.shapeOf(b).join();
    const columnLike = (n: Node) => {
      if (this.boards.has(n) || isLeafLike(n) || n.inputs.length === 0) return false;
      if (n.shape.length === 1) return this.shapeOf(n)[1] === 1;
      return (n.shape.length === 2 || (n.shape.length >= 3 && plates(n).width === 1)) && n.shape[n.shape.length - 1] === 1;
    };
    const broadcastsWider = (n: Node) => {
      const cols = this.shapeOf(n)[1];
      return this.consumers(n).some((u) => {
        const main = u.inputs[0];
        const used = u.inputs.slice(1).find((i) => this.source(i) === n);
        if (used === undefined || this.source(main) === n) return false;
        const broadcasts = explainOf(u.op as string)?.pattern === 'elementwise' && used.shape.length <= u.shape.length && used.shape.every((size, i) => size === 1 || size === u.shape[u.shape.length - used.shape.length + i]);
        return sameShape(main, u) && broadcasts && used.shape[used.shape.length - 1] === 1 && u.shape[u.shape.length - 1] > 1 && this.shapeOf(u)[1] > cols;
      });
    };
    const candidates = [...this.computed].filter((n) => (this.slotOf.get(n) ?? [n]).every(columnLike));
    const discs = new Set(candidates.filter(broadcastsWider));
    const isDisc = (n: Node) => (this.slotOf.get(n) ?? [n]).some((m) => discs.has(m));
    for (let grown = true; grown; ) {
      grown = false;
      for (const n of candidates) {
        if (discs.has(n)) continue;
        const feeds = this.consumers(n).some((u) => sameDrawn(u, n) && isDisc(u));
        const inputs = n.inputs.map((i) => this.source(i)).filter((i) => i.shape.length > 0);
        const fed = explainOf(n.op as string)?.pattern === 'elementwise' && inputs.length > 0 && inputs.every((i) => sameDrawn(i, n) && isDisc(i));
        if (feeds || fed) {
          discs.add(n);
          grown = true;
        }
      }
    }
    return discs;
  }

  private entry(node: Node, level: Level, key = `node:${node.id}`): Entry {
    const shape = this.slotShape(node);
    const [rows, cols] = this.shapeOf(node, shape);
    const leaf = isLeafLike(node);
    let kind: Kind = 'plate';
    if (node.shape.length === 0) kind = 'box';
    else if (leaf && node.shape.length === 1) kind = 'vector';
    else if (this.isDisc(node)) kind = 'disc';
    const e: Entry = { key, node, shape, kind, level, rows, cols, lane: undefined, offsets: [], xs: undefined, title: 'none', quiet: false, table: undefined, dz: 0 };
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
    const lines = this.layout.lines.map((line) => line.map((n) => this.lead(n)).filter((n) => this.computed.has(n) && !this.sourceOf.has(n)));
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
    for (const line of lines) {
      const own = line.filter((n) => !moveTo.has(n) && !this.entryOf.has(n));
      if (own.length === 0) continue;
      const depth = new Map<Node, number>();
      for (const n of own) depth.set(n, Math.max(0, ...n.inputs.map((i) => this.source(i)).filter((i) => depth.has(i)).map((i) => depth.get(i)! + 1)));
      const levels = Array.from({ length: Math.max(...depth.values()) + 1 }, () => this.newLevel());
      for (const n of own) {
        const level = levels[depth.get(n)!];
        for (const [moved, target] of moveTo) if (target === n && !this.entryOf.has(moved) && this.boards.get(n)?.a === moved) this.entry(moved, level);
        const board = this.boards.get(n);
        if (board !== undefined) {
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
    }
    for (const [member, lead] of this.leadOf) {
      const e = this.entryOf.get(lead);
      if (e !== undefined) this.entryOf.set(member, e);
    }
    for (const [view, source] of this.sourceOf) {
      const e = this.entryOf.get(source);
      if (e !== undefined) this.entryOf.set(view, e);
    }
    for (const level of this.levels) level.module = this.levelModule(level);
  }

  private levelModule(level: Level): string | null {
    const own = level.entries.filter((e) => e.original === undefined && this.computed.has(e.node)).map((e) => e.node);
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

  private occupied(level: Level, lane: Lane, skip: Entry): Set<number> {
    const out = new Set<number>();
    const side = this.mergeSide(skip.node);
    const apart = (e: Entry) => {
      const other = this.mergeSide(e.node);
      return side !== undefined && other !== undefined && side !== other;
    };
    for (const e of level.entries) if (e !== skip && e.lane === lane && e.kind !== 'tile' && e.kind !== 'vector' && !apart(e)) e.offsets.forEach((o) => out.add(o));
    return out;
  }

  private mergeSide(n: Node): number | undefined {
    const own = this.layout.merges.findIndex((m) => m.left.includes(n) || m.right.includes(n));
    if (own >= 0) return 2 * own + (this.layout.merges[own].left.includes(n) ? 0 : 1);
    if (!isLeafLike(n)) return undefined;
    const sides = new Set(this.consumers(n).map((u) => this.mergeSide(u)));
    return sides.size === 1 ? [...sides][0] : undefined;
  }

  private setLane(e: Entry, lane: Lane, offsets: number[]): void {
    e.lane = lane;
    e.offsets = offsets;
    lane.members.push(e);
  }

  private drawnA(board: BoardRef): Entry {
    return this.entryOf.get(board.a)!;
  }

  private assignColumns(): void {
    const done = new Set<Entry>();
    const beside: [Entry, Entry][] = [];
    for (const level of this.levels) {
      const ordered = [...level.entries].sort((p, q) => rank(p) - rank(q));
      for (const e of ordered) {
        if (done.has(e)) continue;
        done.add(e);
        const n = e.node;
        if (e.kind === 'tile' && e.table === undefined) continue;
        const table = e.table ?? (this.tables.has(n) ? n : undefined);
        if (table !== undefined) {
          this.setLane(e, this.stream, consecutive(0, e.cols));
          continue;
        }
        const board = this.boards.get(n);
        if (board !== undefined) {
          this.placeBoardGroup(board, level, done);
          continue;
        }
        if (e.kind === 'vector') {
          const user = this.consumers(n)[0];
          const target = this.entryOf.get(user);
          const placed = this.layout.sides.find((side) => side.node === n);
          const host = placed !== undefined ? this.entryOf.get(placed.of)! : target?.level === e.level ? target : this.entryOf.get(user.inputs.find((i) => this.source(i) !== n) ?? user)!;
          if (host.lane !== undefined) this.setLane(e, host.lane, host.offsets.length >= e.cols ? host.offsets.slice(0, e.cols) : consecutive(host.offsets[0] ?? 0, e.cols));
          continue;
        }
        if (isLeafLike(n)) {
          const partners = new Set(this.consumers(n).flatMap((u) => u.inputs).filter((i) => i !== n).map((i) => this.entryOf.get(this.sourceOf.get(i) ?? i)));
          const taken = this.occupied(level, this.stream, e);
          const blocked = level.entries.some((o) => partners.has(o) && o.lane === this.stream && consecutive(0, e.cols).some((k) => o.offsets.includes(k)));
          const start = blocked ? Math.max(...taken) + 3 : 0;
          this.setLane(e, this.stream, consecutive(start, e.cols));
          continue;
        }
        const inputs = n.inputs.map((i) => this.entryOf.get(i)).filter((i): i is Entry => i !== undefined);
        const main = inputs[0];
        if (main === undefined) {
          this.setLane(e, this.stream, consecutive(0, e.cols));
          continue;
        }
        if (e.kind === 'disc' || e.kind === 'box' || e.cols === 1) {
          beside.push([e, main]);
          continue;
        }
        if (main.lane === undefined) {
          this.setLane(e, this.stream, consecutive(0, e.cols));
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
    const seen = new Map<Entry, Set<number>>();
    const levelCuts = new Map<string, Set<number>>();
    const keyOf = (e: Entry) => `${this.levels.indexOf(e.level)}:${this.lanes.indexOf(e.lane!)}`;
    for (const e of this.entries) {
      if (e.lane === undefined) continue;
      const cuts = levelCuts.get(keyOf(e)) ?? new Set<number>();
      for (const c of this.laneCuts(e, seen)) cuts.add(c);
      levelCuts.set(keyOf(e), cuts);
    }
    for (const e of this.entries) {
      if (e.xs !== undefined || e.lane === undefined) continue;
      if (this.slabbed.has(e.node)) {
        e.xs = slabXs(e);
        continue;
      }
      const cuts = [...(levelCuts.get(keyOf(e)) ?? [])];
      const aligned = e.node.inputs.map((i) => this.entryOf.get(i) ?? this.entryOf.get(this.source(i))).filter((f) => f !== undefined && f !== e && f.lane === e.lane && f.xs !== undefined && !this.slabbed.has(f.node));
      e.xs = e.offsets.map((o) => {
        const f = aligned.find((a) => a!.offsets.includes(o));
        if (f === undefined) return position(e.lane!, o) + cuts.filter((c) => c <= o).length * COLUMN_BREAK;
        const inherited = this.laneCuts(f, seen);
        const extra = [...this.laneCuts(e, seen)].filter((c) => c > Math.min(...e.offsets) && c <= o && !inherited.has(c)).length;
        return f.xs![f.offsets.indexOf(o)] + extra * COLUMN_BREAK;
      });
    }
    for (const [e, main] of beside) e.xs = this.sideXs(e, main);
    for (const e of this.entries) if (e.xs === undefined) e.xs = this.sideXs(e, this.entryOf.get(e.node.inputs[0]) ?? e);
  }

  private makeTitleRoom(): void {
    const crowded = (): { at: number; need: number } | undefined => {
      let found: { at: number; need: number } | undefined;
      for (const level of this.levels) {
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
      for (const e of this.entries) if (e.xs !== undefined && Math.min(...e.xs) >= at - 1e-9) e.xs = e.xs.map((x) => x + need);
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
    for (const { node, side, of } of this.layout.sides) {
      const e = this.entryOf.get(node);
      const target = this.entryOf.get(of);
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
    this.layout.merges.forEach((_, i) => {
      const leftEntries = this.entries.filter((e) => this.mergeSide(e.node) === 2 * i);
      const rightEntries = this.entries.filter((e) => this.mergeSide(e.node) === 2 * i + 1);
      let need = 0;
      for (const level of new Set(rightEntries.map((e) => e.level))) {
        const l = span(leftEntries.filter((e) => e.level === level));
        const r = span(rightEntries.filter((e) => e.level === level));
        if (l !== undefined && r !== undefined) need = Math.max(need, l[1] + facing(leftEntries.filter((e) => e.level === level), rightEntries.filter((e) => e.level === level)) - r[0]);
      }
      if (need > 0) shift(rightEntries, need);
    });
    for (const level of this.levels) {
      const plates = level.entries.filter((e) => e.kind === 'plate' && e.xs !== undefined);
      plates.forEach((e, i) => {
        const earlier = plates.slice(0, i).filter((o) => o.lane !== e.lane);
        for (let blocking = this.overlapping(e, earlier); blocking.length > 0; blocking = this.overlapping(e, earlier)) {
          const [lo] = span([e])!;
          const [, hi] = span(blocking)!;
          shift([e], hi + facing(blocking, [e]) - lo);
        }
      });
    }
  }

  private findRounds(): void {
    const sizeOf = new Map<Node, number>();
    [...this.spec.forward, ...(this.spec.functions ?? []).flatMap((f) => f.steps)].forEach((step, k) => {
      const shown = list(step.show);
      const group = step.groups?.[step.groups.length - 1] ?? '';
      for (const n of shown) {
        if ((sizeOf.get(n) ?? Infinity) <= shown.length) continue;
        this.stepOf.set(n, k);
        sizeOf.set(n, shown.length);
        if (ROUND_SUFFIX.test(group)) this.roundOf.set(n, group);
        else this.roundOf.delete(n);
      }
    });
  }

  private attachChains(): void {
    const stepOf = this.stepOf;
    const roundOf = this.roundOf;
    const together = (a: Node, b: Node) => (stepOf.has(a) && stepOf.get(a) === stepOf.get(b)) || (roundOf.has(a) && roundOf.get(a) === roundOf.get(b));
    const chipOf = (e: Entry) => this.layout.grids.find((g) => (this.slotOf.get(e.node) ?? [e.node]).some((m) => g.shared.has(m) || g.registers.has(m)));
    const members = (e: Entry) => this.slotOf.get(e.node) ?? [e.node];
    const linked = (e: Entry, ends: (m: Node) => readonly Node[]) =>
      members(e).flatMap((m) => ends(m).filter((o) => together(m, this.source(o))).flatMap((o) => this.entryOf.get(this.source(o)) ?? [])).filter((f) => f !== e);
    const inputsOf = (e: Entry) => linked(e, (m) => m.inputs);
    const consumersOf = (e: Entry) => linked(e, (m) => this.consumers(m));
    const levelOf = (e: Entry) => this.levels.indexOf(e.level);
    const tileShaped = (e: Entry) => e.kind === 'plate' && e.cols > 1;
    const scalar = (e: Entry) => (e.kind === 'disc' || e.kind === 'box') && e.cols === 1;
    const hosts = new Map<Entry, Entry | null>();
    const fromInputs = (e: Entry): Entry | undefined => {
      if (hosts.has(e)) return hosts.get(e) ?? undefined;
      hosts.set(e, null);
      let best: Entry | undefined;
      for (const f of inputsOf(e)) {
        const host = tileShaped(f) ? f : scalar(f) ? fromInputs(f) : undefined;
        if (host !== undefined && chipOf(host) === chipOf(e) && (best === undefined || levelOf(host) > levelOf(best))) best = host;
      }
      hosts.set(e, best ?? null);
      return best;
    };
    const reductions = this.entries.filter((e) => scalar(e) && e.xs !== undefined && this.computed.has(e.node));
    for (const e of reductions) fromInputs(e);
    for (let grown = true; grown; ) {
      grown = false;
      for (const e of reductions) {
        if (hosts.get(e) != null) continue;
        const host = consumersOf(e)
          .map((u) => (scalar(u) ? (hosts.get(u) ?? undefined) : tileShaped(u) ? u : undefined))
          .find((h) => h !== undefined && chipOf(h) === chipOf(e));
        if (host === undefined) continue;
        hosts.set(e, host);
        grown = true;
      }
    }
    const chains = new Map<Entry, Entry[]>();
    for (const e of reductions) {
      const host = hosts.get(e);
      if (host != null) chains.set(host, [...(chains.get(host) ?? []), e]);
    }
    for (const [host, chain] of chains) {
      chain.sort((p, q) => levelOf(p) - levelOf(q) || this.entries.indexOf(p) - this.entries.indexOf(q));
      const target = host.level;
      const placed = chain.filter((e) => inputsOf(e).every((f) => chain.includes(f) || levelOf(f) <= levelOf(host)));
      let x = Math.max(...host.xs!) + DISC_SIDE - SCALAR_ROW_GAP;
      for (const e of placed) {
        e.level.entries.splice(e.level.entries.indexOf(e), 1);
        target.entries.push(e);
        e.level = target;
        x += SCALAR_ROW_GAP;
        e.xs = e.xs!.map(() => x);
        this.chainAt.set(e, placed.indexOf(e));
        this.chainHost.set(e, host);
      }
      this.chains.set(host, placed);
      const right = target.entries.filter((o) => o !== host && !placed.includes(o) && o.xs !== undefined && Math.min(...o.xs) > Math.max(...host.xs!));
      const need = x + SCALAR_ROW_GAP - Math.min(...right.flatMap((o) => o.xs!));
      if (right.length > 0 && need > 0) for (const o of right) o.xs = o.xs!.map((v) => v + need);
    }
    this.levels.splice(0, this.levels.length, ...this.levels.filter((level) => level.entries.length > 0));
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
    const tile = this.tiles.get(board.op)!;
    const aXs = a.xs ?? (a.lane === undefined ? [0] : a.offsets.map((o) => position(a.lane!, o)));
    const lane = this.lane(a.level === level ? Math.max(...aXs) + BOARD_GAP : Math.min(...aXs), a.lane?.spacing ?? STREAM_SPACING);
    this.setLane(out, lane, consecutive(0, out.cols));
    this.setLane(tile, lane, consecutive(0, tile.cols));
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
      const tile = this.tiles.get(b.op)!;
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
    for (const node of [m.node, ...(this.drawnOn.get(m.node) ?? [])]) {
      const { count, width } = plates(node);
      if (node.shape.length >= 3 && count > 1 && width > 1 && (size === 0 || width < size)) size = width;
      for (const u of this.computed) {
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
    const board = this.boards.get(n);
    const from = board !== undefined ? [this.drawnA(board)] : n.inputs.map((i) => this.entryOf.get(this.source(i)));
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
    for (const e of this.entries) e.title = 'none';
    for (const level of this.levels) {
      for (const e of level.entries) {
        if (e.kind === 'disc' || e.kind === 'box') {
          e.title = (this.chainAt.get(e) ?? (e.ground === undefined ? 0 : 1)) % 2 === 1 ? 'below' : 'left';
          continue;
        }
        if (e.kind === 'vector') continue;
        if (e.ground !== undefined && e.kind === 'plate') {
          e.title = 'below';
          continue;
        }
        if (e.kind === 'tile') {
          if (e.table !== undefined || !level.entries.some((o) => o.kind === 'tile' && o !== e && o.lane === e.lane && (o.xs?.[0] ?? 0) < (e.xs?.[0] ?? 0))) e.title = 'left';
          continue;
        }
        if (e.cols === 1) {
          e.title = 'right';
          continue;
        }
        const board = this.boards.get(e.node);
        if (board !== undefined && this.drawnA(board).level === level) {
          e.title = 'below';
          continue;
        }
        e.title = 'left';
      }
      const vectors = level.entries.filter((e) => e.kind === 'vector').sort((p, q) => p.xs![0] - q.xs![0]);
      if (vectors.length > 0) vectors[vectors.length - 1].title = 'right';
    }
    const norms = new Set(this.stepModules.filter((m) => !this.stepModules.some((c) => c.parent === m.id) && m.nodes.some((n) => this.computed.has(n) && this.isDisc(n))));
    for (const e of this.entries) {
      if (e.kind === 'vector') e.quiet = true;
      if (this.computed.has(e.node) && [...norms].some((m) => m.nodes.includes(e.node) && !this.isOutput(m, e.node))) e.quiet = true;
    }
  }

  private braceLevels(brace: StepBrace): Level[] {
    const own = new Set(brace.nodes.flatMap((n) => this.entryOf.get(n)?.level ?? []));
    return this.levels.filter((l) => own.has(l));
  }

  private braceGroups(): StepBrace[][] {
    const slotsOf = (brace: StepBrace) => {
      const out = new Map<readonly Node[], Set<Node>>();
      for (const n of brace.nodes) {
        const slot = this.slotOf.get(n);
        if (slot !== undefined) out.set(slot, (out.get(slot) ?? new Set()).add(n));
      }
      return out;
    };
    const rounds = (a: Map<readonly Node[], Set<Node>>, b: Map<readonly Node[], Set<Node>>) => [...a].some(([slot, own]) => [...(b.get(slot) ?? [])].some((n) => !own.has(n)));
    const groups: { braces: StepBrace[]; slots: Map<readonly Node[], Set<Node>>[] }[] = [];
    for (const brace of this.braces) {
      const slots = slotsOf(brace);
      const group = groups.find((g) => g.braces[0].depth === brace.depth && g.slots.some((other) => rounds(other, slots)));
      if (group === undefined) groups.push({ braces: [brace], slots: [slots] });
      else {
        group.braces.push(brace);
        group.slots.push(slots);
      }
    }
    return groups.map((g) => g.braces);
  }

  private buildModules(): void {
    for (const group of this.braceGroups()) {
      const brace = group[0];
      const nodes = [...new Set(group.flatMap((b) => b.nodes))];
      const own = new Set(nodes.flatMap((n) => this.entryOf.get(n)?.level ?? []));
      const levels = this.levels.filter((l) => own.has(l));
      if (levels.length === 0) continue;
      const first = this.levels.indexOf(levels[0]);
      const last = this.levels.indexOf(levels[levels.length - 1]);
      const gap = this.levels.slice(first, last + 1).find((l) => !levels.includes(l) && l.entries.some((e) => e.node.phase !== 'backward' && e.original === undefined));
      if (gap !== undefined) {
        const between = gap.entries.map((e) => e.node.name).join(', ');
        throw new Error(`steps: the brace of '${brace.id}' (${brace.title}) frames tensors that are not next to each other; ${between} lies between them`);
      }
      const nested = this.braces.some((other) => !group.includes(other) && other.nodes.length < nodes.length && other.nodes.every((n) => nodes.includes(n)));
      const side = brace.depth >= 2 ? 'left' : brace.depth === 0 && nested ? 'outer' : 'right';
      const module: SceneModule = { id: brace.module ?? brace.id, title: brace.title, subtitle: brace.subtitle, nodes, side, depth: brace.depth };
      if (group.length > 1 || brace.copies !== undefined) module.rounds = group.map((b) => ({ title: b.title, subtitle: b.subtitle, nodes: b.nodes, grid: b.grid, copies: b.copies }));
      if (side === 'left') module.column = brace.depth - 2;
      this.modules.push(module);
      this.braceNodes.set(brace, nodes);
    }
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
      if (e.original !== undefined) value = e.ground!.chips.every((chip) => [...chip.shared, ...chip.registers].every((m) => hiddenNode(m.node)));
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
    const blocks = [...blockOf.values()];

    const boards: BoardGeometry[] = [];
    for (const board of this.boards.values()) {
      const out = this.entryOf.get(board.op)!;
      const tile = this.tiles.get(board.op)!;
      const a = this.copyEntry(board.a, board.op) ?? this.drawnA(board);
      const beside = (b: BoardRef, other: Entry) => other.level === out.level && other.dz === out.dz && (this.copyEntry(b.a, b.op) ?? this.drawnA(b)) === other;
      const siblings = [...this.boards.values()].filter((b) => this.entryOf.get(b.op)?.level === out.level && beside(b, a)).sort((p, q) => p.op.id - q.op.id);
      const at = siblings.findIndex((b) => b === board);
      const previous = at > 0 ? this.tiles.get(siblings[at - 1].op) : undefined;
      boards.push({ op: board.op, a: blockOf.get(a)!, sharedA: a.level !== out.level || a.dz !== out.dz, b: blockOf.get(tile)!, out: blockOf.get(out)!, y: out.level.y, previous: previous === undefined ? undefined : blockOf.get(previous) });
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
      boxes.push({ module: m, ...this.around(keys), y: range });
    }

    const kernels: KernelGround[] = [];
    const links: { original: Block; copy: Block }[] = [];
    for (const ground of this.grounds) {
      const own = ground.level.entries.filter((e) => e.ground === ground);
      if (own.every((e) => hiddenOf(e))) continue;
      kernels.push({
        grids: ground.grids,
        y: ground.level.y,
        ...(ground.hbm === undefined ? {} : { hbm: ground.hbm }),
        chips: ground.chips.flatMap((chip) => (chip.card === undefined ? [] : [{ expand: chip.expand, card: chip.card, ...(chip.sharedBox === undefined ? {} : { shared: chip.sharedBox }), ...(chip.registersBox === undefined ? {} : { registers: chip.registersBox }) }])),
        blocks: own.map((e) => blockOf.get(e)!),
      });
      for (const [original, copy] of ground.copies) if (!hiddenOf(original) && !hiddenOf(copy)) links.push({ original: blockOf.get(original)!, copy: blockOf.get(copy)! });
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
      const nodes = this.braceNodes.get(brace);
      if (nodes === undefined) continue;
      const range = fold !== undefined && isFolded(fold) ? boxY.get(fold.id) : spanOf(this.braceLevels({ ...brace, nodes }));
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
    const streamWidth = Math.max(1, ...this.stream.members.flatMap((m) => m.offsets.map((o) => o + 1)));
    const rows = this.rowsOf();
    return {
      blocks,
      boards,
      boxes,
      ranges,
      io,
      chips,
      kernels,
      links,
      copyFor: (node, consumer) => {
        const copy = this.copyEntry(node, consumer);
        return copy === undefined ? undefined : blockOf.get(copy);
      },
      braces: { z: Math.max(zPlate(rows - 1), ...this.entries.filter((en) => en.kind === 'plate').flatMap((en) => this.zs(en))) + BRACE_FRONT, cx: position(this.stream, (streamWidth - 1) / 2), cz: zPlate((rows - 1) / 2) },
      lo: 0,
      top,
      sourceOf: this.sourceOf,
      leadOf: this.leadOf,
      grids: this.layout.grids,
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

  private around(entries: Entry[]): { x: [number, number]; z: [number, number] } {
    const xs = entries.flatMap((e) => e.xs ?? []);
    const zs = entries.flatMap((e) => this.zs(e));
    return { x: [Math.min(...xs) - BOX_MARGIN.left, Math.max(...xs) + BOX_MARGIN.right], z: [Math.min(...zs) - BOX_MARGIN.back, Math.max(...zs) + BOX_MARGIN.front] };
  }

  private zs(e: Entry): number[] {
    return this.baseZs(e).map((z) => z + e.dz);
  }

  private baseZs(e: Entry): number[] {
    if (e.kind === 'tile') return Array.from({ length: e.rows }, (_, r) => zTile(r, e.rows));
    if (e.kind === 'vector') return [zVector];
    if (e.kind === 'box') return [zPlate((this.rowsOf() - 1) / 2)];
    if (this.slabbed.has(e.node)) return Array.from({ length: e.node.shape[0] * e.rows }, (_, i) => this.slabZ(e.node, Math.floor(i / e.rows), i % e.rows));
    return this.rowZs(e.node, e.rows);
  }

  private footprintOf(e: Entry): [number, number] {
    const spacing = e.lane?.spacing ?? STREAM_SPACING;
    if (e.kind === 'tile') return [spacing >= STREAM_SPACING ? TILE_WIDTH : TILE_WIDTH_NARROW, e.rows > TILE_DENSE_ROWS ? TILE_DEPTH_DENSE : TILE_DEPTH];
    if (e.kind === 'disc') return DISC_FOOTPRINT;
    if (e.kind === 'box') return BOX_FOOTPRINT;
    if (this.slabbed.has(e.node)) return [STREAM_FOOTPRINT, TILE_DEPTH_DENSE];
    if (e.lane?.stream === true) return [STREAM_FOOTPRINT, STREAM_FOOTPRINT];
    return [Math.min(BAR_FOOTPRINT, spacing - BAR_CLEARANCE), Math.min(BAR_FOOTPRINT, spacing - BAR_CLEARANCE)];
  }

  private rowsOf(): number {
    return Math.max(1, ...this.entries.filter((e) => e.kind === 'plate').map((e) => e.rows));
  }

  private headsOf(n: Node, shape: readonly number[]): { count: number; width: number } {
    if (shape.length >= 3) return plates(n, shape);
    const view = this.drawnOn.get(n)?.find((v) => v.shape.length >= 3);
    return view === undefined ? { count: 1, width: rowsCols(n, shape)[1] } : plates(view);
  }

  private block(e: Entry, hidden: boolean, aligned: ReadonlyMap<Node, Node>): Block {
    const y = e.level.y;
    const xs = e.xs ?? [0];
    const zs = this.zs(e);
    const n = e.node;
    const elements: Element[] = [];
    const size = e.shape.reduce((p, d) => p * d, 1);
    for (let offset = 0; offset < size; offset++) {
      const index = indexOf(e.shape, offset);
      const slab = this.slabbed.has(n);
      const [row, col] = slab ? [index[1], index[2]] : e.rows > 1 && e.cols === 1 ? [offset, 0] : e.shape.length >= 2 ? rowCol(n, index, e.shape) : [0, offset];
      const z = slab ? this.slabZ(n, index[0], row) + e.dz : zs[Math.min(row, zs.length - 1)];
      elements.push({ index, row, col, x: xs[Math.min(col, xs.length - 1)], y, z });
    }
    const footprint = this.footprintOf(e);
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
    else if (e.kind === 'vector') title = { x: x1 + TITLE.vector[0], y: y + TITLE.vector[1], z: zs[0], anchor: 'right' };
    else if (e.title === 'below' || (e.title === 'left' && this.hasGroupLabels(e))) title = { x: (x0 + x1) / 2, y: y + TITLE.below[0], z: front + TITLE.below[1], anchor: 'below' };
    else if (e.title === 'right') title = { x: x1 + TITLE.right[0], y: y + TITLE.right[1], z: front, anchor: 'right' };
    else title = { x: x0 + TITLE.left[0], y: y + TITLE.left[1], z: front, anchor: 'left' };
    const tokens = aligned.get(n)?.tokens;
    const named = e.kind === 'tile' ? tokens : this.selectLabel(n) !== undefined ? (this.selectLabel(n)!.tokens ?? undefined) : undefined;
    const slabs = this.slabsOf(e, elements, footprint, y);
    const chunks = slabs === undefined && !scalar ? chunksOf(this.headsOf(n, e.shape), elements, footprint, y) : undefined;
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
    const slot = this.slotOf.get(n);
    return {
      node: n,
      key: e.key,
      ...(e.original === undefined ? {} : { copyOf: n }),
      ...(slot === undefined ? {} : { slot }),
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

function chunksOf(heads: { count: number; width: number }, elements: Element[], footprint: readonly [number, number], y: number): Plate[] | undefined {
  const { count, width } = heads;
  if (count < 2) return undefined;
  return Array.from({ length: count }, (_, p) => {
    const members = elements.filter((el) => Math.floor(el.col / width) === p);
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
  if (e.kind === 'tile' && e.table !== undefined) return 1;
  if (e.kind === 'vector' || isLeafLike(e.node)) return 4;
  return 2;
}

export type { Vec3 };
