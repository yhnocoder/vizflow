import { BoxGeometry, Group, Mesh, MeshLambertMaterial, Vector3 } from 'three';
import type { Node } from '../autograd';
import { offsetsByPosition, positionsOf } from '../explain';
import { renderText } from '../formula';
import { type CacheDecl, type CompactViewSpec, type GenerationState, type ModuleTile, PAGE_REQUEST, type TileState, engineStepAt, pageTokens, tileLeaves, tileStates } from '../steps';
import type { LinearRgb } from '../theme';
import { BarMesh } from './blocks';
import { barColor, mix, outlineOpacity } from './colors';
import type { SceneController } from './controller';
import { boundsOf, graphPoints } from './framing';
import type { Label } from './labels';
import type { Vec3 } from './layout';
import { LineBatch, type Polyline, rectPoints } from './lines';
import { barHeight, scaleFor } from './scales';

export const SAMPLE_TITLE = '采样';
export const STATE_TEXT = '状态';
export const SHARED_TABLE_TEXT = '每一层按同一张块表存放';

const VIEW_GAP = 10;
const TILE_WIDTH = 7;
const TILE_DEPTH = 1.2;
const TILE_THICKNESS = 0.12;
const TILE_PITCH = 1.25;
const TOP_GAP = 0.45;
const TITLE_INSET = 0.3;
const BRACE_GAP = 0.35;
const CHIP_ROOM = 3.4;
const CHIP_INSET = 0.2;
const BRACE_TICK = 0.2;
const LABEL_ROOM = 6;
const CAPTION_ROOM = 3.5;
const NOTE_ROOM = 6;
const TOKEN_ROOM = 0.8;
const SLOT_GAP = 0.2;
const SLOT_PAD = 0.1;
const BLOCK_GAP = 0.6;
const BLOCK_PAD = 0.12;
const BAR_PITCH = 0.25;
const BAR_FOOTPRINT: [number, number] = [0.19, 0.19];
const NODE_PITCH = 0.4;
const CACHE_BAR_MAX = 8;
const CACHE_BAR_CAP = 0.5;
const LABEL_CLEAR = 0.4;
const PLAIN_SLOT_WIDTH = 0.8;
const PLAIN_SLOT_DEPTH = 1;
const OUTLINE_LIFT = 0.005;
const LIT_MIX = 0.55;
const IDLE_MIX = 0.6;
const TILE_OUTLINE: Record<TileState, number> = { lit: 0.95, done: 0.5, idle: 0.2 };
const SLOT_OUTLINE = { write: 0.95, read: 0.6, held: 0.25, empty: 0.12 };
const BLOCK_OUTLINE = 0.8;
const BRACE_OPACITY = 0.55;

type SlotStyle = 'write' | 'read' | 'held';

interface CacheView {
  decl: CacheDecl;
  leaf: number;
  offsets: Map<Node, number[][]>;
  width: number;
  depth: number;
  bars: BarMesh | undefined;
  plates: Mesh[];
  tokens: Label[];
  blocks: Label[];
  caption: Label;
  note: Label;
  blocked: boolean;
  shared: boolean;
  span: number;
}

interface Slot {
  x: number;
  width: number;
  position: number | undefined;
  nodes: Node[];
  style: SlotStyle;
  text: string;
}

export class CompactView {
  private readonly group = new Group();
  private readonly tiles: readonly ModuleTile[];
  private readonly leaves: ModuleTile[];
  private readonly levels: number[] = [];
  private readonly sampleLevel: number;
  private readonly tileGeometry = new BoxGeometry(TILE_WIDTH, TILE_THICKNESS, TILE_DEPTH);
  private readonly slotGeometry = new BoxGeometry(1, TILE_THICKNESS / 2, 1);
  private readonly tileMeshes: Mesh[];
  private readonly caches: CacheView[];
  private readonly outlines: LineBatch;
  private readonly titles: Label[];
  private readonly groupLabels: Label[];
  private readonly chip: Label;
  private readonly shared: Label;
  private readonly length: number;
  readonly labelSet: ReadonlySet<Label>;

  constructor(
    private readonly c: SceneController,
    private readonly spec: CompactViewSpec,
  ) {
    const steps = c.steps;
    if (steps === undefined) throw new Error('compact_view: the scene has no steps');
    this.tiles = steps.tiles;
    this.leaves = tileLeaves(this.tiles);
    this.length = spec.infer.input.shape[0];
    let y = 0;
    this.tiles.forEach((tile, k) => {
      if (k > 0) y += TOP_GAP;
      for (const _ of tile.children.length > 0 ? tile.children : [tile]) {
        this.levels.push(y);
        y += TILE_PITCH;
      }
    });
    this.sampleLevel = y + TOP_GAP;
    this.tileMeshes = [...this.leaves, undefined].map(() => {
      const mesh = new Mesh(this.tileGeometry, new MeshLambertMaterial());
      this.group.add(mesh);
      return mesh;
    });
    this.outlines = new LineBatch(c.theme.ink, c.theme.surface);
    this.group.add(this.outlines.object);
    this.titles = [...this.leaves, undefined].map((tile) => {
      const label = c.labels.add('kk', 'vf-tile-title');
      label.anchor = 'left-middle';
      const title = tile?.title ?? SAMPLE_TITLE;
      label.setContent(title, () => {
        const span = document.createElement('span');
        renderText(title, span);
        return span;
      });
      return label;
    });
    this.groupLabels = this.tiles.filter((tile) => tile.children.length > 0).map((tile) => {
      const label = c.labels.add('kk', 'vf-tile-group');
      label.anchor = 'right-middle';
      label.setContent(`${tile.title}\n${tile.subtitle ?? ''}`, () => {
        const fragment = document.createDocumentFragment();
        for (const [text, className] of [[tile.title, 'vf-mod-title'], [tile.subtitle, 'vf-mod-sub']] as const) {
          if (text === undefined) continue;
          const span = document.createElement('span');
          span.className = className;
          renderText(text, span);
          fragment.appendChild(span);
        }
        return fragment;
      });
      return label;
    });
    this.chip = c.labels.add('kk', 'vf-tile-token');
    this.chip.anchor = 'right-middle';
    this.shared = c.labels.add('kk', 'vf-cache-note');
    this.shared.anchor = 'left-middle';
    this.shared.setText(SHARED_TABLE_TEXT);
    const placed = this.leaves.flatMap((tile, leaf) => tile.caches.map((decl) => ({ decl, leaf })));
    const missing = spec.infer.caches.filter((decl) => !placed.some((p) => p.decl === decl));
    if (missing.length > 0) throw new Error(`compact_view: no module tile computes the cached tensor '${missing[0].nodes[0].name}'`);
    const positions = positionsOf(c.g, this.length);
    const pool = c.options.kvPool;
    let firstBlocked = true;
    this.caches = placed.map(({ decl, leaf }) => {
      const offsets = new Map(decl.nodes.map((n) => [n, offsetsByPosition(positions, n)]));
      const cols = Math.max(...[...offsets.values()].flatMap((rows) => rows.map((r) => r?.length ?? 0)));
      const perSlot = Math.max(...Array.from({ length: this.length }, (_, p) => decl.nodes.filter((n) => (offsets.get(n)![p]?.length ?? 0) > 0).length));
      const withBars = cols <= CACHE_BAR_MAX;
      const capacity = this.length + (pool?.blockSize ?? 0);
      const rowsOf = (n: Node) => offsets.get(n)!.filter((r) => r !== undefined).length;
      const single = decl.nodes.every((n) => rowsOf(n) === 1);
      const blocked = pool !== undefined && rowsOf(decl.nodes[0]) > 1;
      const shared = blocked && firstBlocked;
      if (blocked) firstBlocked = false;
      const bars = withBars ? new BarMesh(capacity * perSlot * cols, BAR_FOOTPRINT, c.theme) : undefined;
      if (bars !== undefined) this.group.add(bars.group);
      const plates = Array.from({ length: capacity }, () => {
        const mesh = new Mesh(this.slotGeometry, new MeshLambertMaterial());
        this.group.add(mesh);
        return mesh;
      });
      const labelAt = (className: string, anchor: Label['anchor']) => {
        const label = c.labels.add('kk', className);
        label.anchor = anchor;
        return label;
      };
      const caption = labelAt('vf-cache-caption', 'right-middle');
      const width = withBars ? cols * BAR_PITCH + 2 * SLOT_PAD : PLAIN_SLOT_WIDTH;
      const bs = pool?.blockSize ?? 1;
      const slots = single ? 1 : blocked ? Math.ceil(this.length / bs) * bs : this.length;
      const span = slots * width + (slots - 1) * SLOT_GAP + (blocked ? (slots / bs - 1) * BLOCK_GAP : 0);
      const note = labelAt('vf-cache-note', 'above-right');
      return {
        decl,
        leaf,
        offsets,
        width,
        depth: withBars ? perSlot * NODE_PITCH + 2 * SLOT_PAD : PLAIN_SLOT_DEPTH,
        bars,
        plates,
        tokens: Array.from({ length: capacity }, () => labelAt('vf-cache-token', 'top-center')),
        blocks: Array.from({ length: capacity }, () => labelAt('vf-cache-block', 'bottom-center')),
        caption,
        note,
        blocked,
        shared,
        span,
      };
    });
    this.labelSet = new Set([...this.titles, ...this.groupLabels, this.chip, this.shared, ...this.caches.flatMap((v) => [v.caption, v.note, ...v.tokens, ...v.blocks])]);
    c.compactScene.add(this.group);
  }

  get active(): boolean {
    return this.c.state.generation?.compact !== undefined;
  }

  private place(): void {
    const { min, max } = boundsOf(graphPoints(this.c));
    const direction = this.c.rig.direction;
    this.group.position.set(max[0] + VIEW_GAP + LABEL_ROOM, min[1], (min[2] + max[2]) / 2);
    this.group.rotation.set(0, Math.atan2(direction.x, direction.z), 0);
    this.group.updateMatrixWorld();
  }

  private world(local: Vec3): Vec3 {
    const p = this.group.localToWorld(new Vector3(...local));
    return [p.x, p.y, p.z];
  }

  private cacheStart(view: CacheView): number {
    const before = this.caches.filter((other) => other.leaf === view.leaf && this.caches.indexOf(other) < this.caches.indexOf(view));
    const room = before.reduce((x, other) => x + CAPTION_ROOM + other.span + SLOT_GAP, 0);
    return TILE_WIDTH + room + CAPTION_ROOM;
  }

  points(): Vec3[] {
    this.place();
    const top = this.sampleLevel + TILE_THICKNESS;
    const right = Math.max(TILE_WIDTH, ...this.caches.map((view) => this.cacheStart(view) + view.span + NOTE_ROOM * (view.shared ? 1 : 0)));
    const left = -CHIP_ROOM - BRACE_GAP - LABEL_ROOM;
    const zFront = TILE_DEPTH / 2 + TOKEN_ROOM;
    const zBack = -TILE_DEPTH / 2;
    return [
      this.world([left, 0, zFront]),
      this.world([left, top, zBack]),
      this.world([right, 0, zFront]),
      this.world([right, top, zBack]),
    ];
  }

  update(): void {
    const c = this.c;
    const generation = c.state.generation;
    const compact = generation?.compact;
    if (generation === undefined || compact === undefined) {
      for (const label of this.labelSet) label.visible = false;
      return;
    }
    this.place();
    const theme = c.theme;
    this.outlines.setColor(theme.ink, theme.surface);
    const states = tileStates(this.leaves, compact);
    const all: TileState[] = [...states, compact.sample ? 'lit' : 'idle'];
    const outlines: Polyline[] = [];
    const xc = TILE_WIDTH / 2;
    const levelY = (i: number) => (i < this.leaves.length ? this.levels[i] : this.sampleLevel);
    all.forEach((state, i) => {
      const y = levelY(i);
      const mesh = this.tileMeshes[i];
      mesh.position.set(xc, y + TILE_THICKNESS / 2, 0);
      const fill: LinearRgb = state === 'lit' ? mix(theme.pos, theme.surface, LIT_MIX) : state === 'done' ? theme.line : mix(theme.bg, theme.surface, IDLE_MIX);
      (mesh.material as MeshLambertMaterial).color.setRGB(...fill);
      outlines.push({ points: rectPoints(xc, y + TILE_THICKNESS + OUTLINE_LIFT, 0, TILE_WIDTH, TILE_DEPTH), opacity: TILE_OUTLINE[state] });
      const title = this.titles[i];
      title.pos = this.world([TITLE_INSET, y + TILE_THICKNESS, 0]);
      title.setFlag('vf-tile-lit', state === 'lit');
      title.setFlag('vf-tile-idle', state === 'idle');
      title.visible = true;
    });
    let leaf = 0;
    let loop = 0;
    for (const tile of this.tiles) {
      const count = Math.max(1, tile.children.length);
      if (tile.children.length > 0) {
        const lo = levelY(leaf) + TILE_THICKNESS / 2;
        const hi = levelY(leaf + count - 1) + TILE_THICKNESS / 2;
        const x = -CHIP_ROOM - BRACE_GAP;
        const z = 0;
        outlines.push({ points: [x + BRACE_TICK, lo, z, x, lo, z, x, hi, z, x + BRACE_TICK, hi, z], opacity: BRACE_OPACITY });
        const label = this.groupLabels[loop++];
        label.pos = this.world([x - BRACE_TICK / 2, (lo + hi) / 2, z]);
        label.visible = true;
      }
      leaf += count;
    }
    const tokens = pageTokens(this.spec.infer);
    const t = generation.position;
    const last = compact.sample ? this.leaves.length : states.lastIndexOf('lit');
    this.chip.visible = last >= 0;
    if (last >= 0) {
      this.chip.pos = this.world([-CHIP_INSET, levelY(last) + TILE_THICKNESS, 0]);
      this.chip.setText(compact.sample ? `抽到 ${tokens[t + 1]}` : `位置 ${t}：${tokens[t]}`);
    }
    this.shared.visible = false;
    for (const view of this.caches) this.drawCache(view, generation, states[view.leaf], levelY(view.leaf), outlines);
    this.outlines.setPolylines(outlines);
  }

  private blockTable(t: number): number[] {
    const pool = this.c.kvPool;
    if (pool === undefined) return [];
    const steps = pool.steps();
    return steps[engineStepAt(steps, t)].requests.find((r) => r.id === PAGE_REQUEST)?.table ?? [];
  }

  private slots(view: CacheView, generation: GenerationState, state: TileState): Slot[] {
    const entry = generation.caches.find((e) => e.decl === view.decl)!;
    const t = generation.position;
    const written = state !== 'idle';
    const width = view.width;
    let x = this.cacheStart(view);
    if (entry.kind === 'state') {
      return [{ x: x + width / 2, width, position: written ? t : t - 1, nodes: written ? entry.nodes : entry.previous, style: state === 'lit' ? 'write' : 'held', text: STATE_TEXT }];
    }
    const held = [...new Set([...entry.read, ...(written ? [t] : [])])].sort((a, b) => a - b);
    const styleOf = (p: number): SlotStyle => (state !== 'lit' ? 'held' : p === t ? 'write' : entry.read.includes(p) ? 'read' : 'held');
    const tokens = pageTokens(this.spec.infer);
    const slotAt = (p: number | undefined): Slot => {
      const slot: Slot = { x: x + width / 2, width, position: p, nodes: entry.nodes, style: p === undefined ? 'held' : styleOf(p), text: p === undefined ? '' : tokens[p] };
      x += width + SLOT_GAP;
      return slot;
    };
    if (!view.blocked) return held.map(slotAt);
    const bs = this.c.options.kvPool!.blockSize;
    const out: Slot[] = [];
    const first = Math.floor(held[0] / bs);
    const lastBlock = Math.floor(held[held.length - 1] / bs);
    for (let k = first; k <= lastBlock; k++) {
      if (k > first) x += BLOCK_GAP - SLOT_GAP;
      for (let j = 0; j < bs; j++) {
        const p = k * bs + j;
        out.push(slotAt(held.includes(p) ? p : undefined));
      }
    }
    return out;
  }

  private drawCache(view: CacheView, generation: GenerationState, state: TileState, y: number, outlines: Polyline[]): void {
    const theme = this.c.theme;
    const slots = this.slots(view, generation, state);
    const entry = generation.caches.find((e) => e.decl === view.decl)!;
    const elements: Vec3[] = [];
    const heights: number[] = [];
    const clipped: boolean[] = [];
    const colors: LinearRgb[] = [];
    const outline: number[] = [];
    const z0 = -view.depth / 2;
    view.plates.forEach((plate, i) => {
      const slot = slots[i];
      plate.visible = slot !== undefined;
      const token = view.tokens[i];
      token.visible = slot !== undefined && slot.text !== '';
      if (slot === undefined) return;
      plate.scale.set(slot.width, 1, view.depth);
      plate.position.set(slot.x, y + TILE_THICKNESS / 4, 0);
      const fill = slot.position === undefined ? theme.surface : slot.style === 'write' ? mix(theme.pos, theme.surface, LIT_MIX) : theme.bg;
      (plate.material as MeshLambertMaterial).color.setRGB(...fill);
      const opacity = slot.position === undefined ? SLOT_OUTLINE.empty : SLOT_OUTLINE[slot.style];
      outlines.push({ points: rectPoints(slot.x, y + TILE_THICKNESS / 2 + OUTLINE_LIFT, 0, slot.width, view.depth), opacity });
      token.pos = this.world([slot.x, y - CACHE_BAR_CAP - LABEL_CLEAR, view.depth / 2]);
      token.setText(slot.text);
      token.setFlag('vf-cache-new', slot.style === 'write');
      if (view.bars === undefined || slot.position === undefined) return;
      const active = slot.style !== 'held';
      slot.nodes.forEach((node, r) => {
        const rows = view.offsets.get(node)!;
        const offsets = rows[slot.position!] ?? [];
        const scale = scaleFor(node.value.data, node.dtype, CACHE_BAR_CAP);
        offsets.forEach((offset, q) => {
          const value = node.value.data[offset];
          const bar = barHeight(value, scale, true, CACHE_BAR_CAP);
          elements.push([slot.x - slot.width / 2 + SLOT_PAD + (q + 0.5) * BAR_PITCH, y + TILE_THICKNESS / 2, z0 + SLOT_PAD + (r + 0.5) * NODE_PITCH]);
          heights.push(bar.height);
          clipped.push(bar.clipped);
          colors.push(barColor(value, theme, { active, far: false, focused: true }));
          outline.push(outlineOpacity({ active, far: false, tile: false }));
        });
      });
    });
    if (view.bars !== undefined) {
      const capacity = view.bars.mesh.count;
      while (elements.length < capacity) {
        elements.push([0, 0, 0]);
        heights.push(0);
        clipped.push(false);
        colors.push(theme.surface);
        outline.push(0);
      }
      view.bars.setTheme(theme);
      view.bars.update(elements.map(([x, yy, z]) => ({ x, y: yy, z })), { heights, clipped, colors, outlineOpacity: outline, seeThrough: false });
    }
    const bs = this.c.options.kvPool?.blockSize ?? 1;
    const count = view.blocked ? slots.length / bs : 0;
    const table = view.blocked ? this.blockTable(generation.position) : [];
    const first = Math.floor((slots.find((s) => s.position !== undefined)?.position ?? 0) / bs);
    view.blocks.forEach((label, k) => {
      label.visible = view.shared && k < count;
      if (k >= count) return;
      const members = slots.slice(k * bs, (k + 1) * bs);
      const lo = members[0].x - members[0].width / 2 - BLOCK_PAD;
      const hi = members[bs - 1].x + members[bs - 1].width / 2 + BLOCK_PAD;
      outlines.push({ points: rectPoints((lo + hi) / 2, y + TILE_THICKNESS / 2 + OUTLINE_LIFT, 0, hi - lo, view.depth + 2 * BLOCK_PAD), opacity: BLOCK_OUTLINE });
      label.pos = this.world([(lo + hi) / 2, y + CACHE_BAR_CAP + LABEL_CLEAR, -view.depth / 2 - BLOCK_PAD]);
      label.setText(`块 ${table[first + k]}`);
    });
    const nodes = entry.kind === 'state' ? (state === 'idle' ? entry.previous : entry.nodes) : view.decl.nodes;
    const tex = [...new Set(nodes.map((n) => n.tex))].map((t) => `$${t}$`).join('、');
    view.caption.pos = this.world([this.cacheStart(view) - SLOT_GAP, y + TILE_THICKNESS / 2, 0]);
    view.caption.setContent(tex, () => {
      const span = document.createElement('span');
      renderText(tex, span);
      return span;
    });
    view.caption.visible = true;
    const note = entry.kind === 'state' ? `固定 ${entry.values} 个数` : entry.kind === 'window' ? `读最近 ${entry.read.length} 行` : '';
    view.note.pos = this.world([this.cacheStart(view), y + CACHE_BAR_CAP + LABEL_CLEAR, -view.depth / 2]);
    view.note.setText(note);
    view.note.visible = note !== '';
    if (view.shared && slots.length > 0) {
      const end = slots[slots.length - 1];
      this.shared.pos = this.world([end.x + end.width / 2 + BLOCK_PAD + SLOT_GAP, y + TILE_THICKNESS / 2, 0]);
      this.shared.visible = true;
    }
  }

  dispose(): void {
    this.c.compactScene.remove(this.group);
    this.tileGeometry.dispose();
    this.slotGeometry.dispose();
    for (const mesh of this.tileMeshes) (mesh.material as MeshLambertMaterial).dispose();
    for (const view of this.caches) {
      view.bars?.dispose();
      for (const plate of view.plates) (plate.material as MeshLambertMaterial).dispose();
    }
    this.outlines.dispose();
  }
}
