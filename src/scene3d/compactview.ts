import { Group, Vector3 } from 'three';
import type { Node } from '../autograd';
import { positionsOf } from '../explain';
import { renderText } from '../formula';
import { type CompactViewSpec, type ModuleTile, PAGE_REQUEST, type TileState, engineStepAt, pageTokens, tileLeaves } from '../steps';
import { PlateMesh } from './blocks';
import { COMPACT_MIN_ZOOM } from './camera';
import { mix } from './colors';
import { type CompactPlan, type CompactPool, compactBlocks } from './compactlayout';
import type { SceneController } from './controller';
import type { Label } from './labels';
import type { ResolvedLayout, Vec3 } from './layout';
import { LineBatch } from './lines';
import { BRACE_LABEL_GAP, type Brace, bracePoints, braceScale, braceTip } from './modules';

export const SAMPLE_TITLE = '采样';
export const SHARED_TABLE_TEXT = '每一层按同一张块表存放';

const TILE_PAD_PX = 14;
const TILE_PAD_Y_PX = 8;
const TILE_MIN_WIDTH = 5;
const TILE_MIN_DEPTH = 1.2;
const CHIP_GAP = 0.4;
const BRACE_GAP = 0.5;
const LIT_MIX = 0.45;
const IDLE_MIX = 0.55;
const NOTE_FRONT = 1.2;

const UP = new Vector3(0, 1, 0);

function captionOf(tile: ModuleTile): string {
  return tile.names.length === 1 && tile.names[0] === tile.title ? '' : tile.names.join(' · ');
}

function twoLines(label: Label, key: string, title: string, caption: string, titleClass: string, captionClass: string): void {
  label.setContent(key, () => {
    const fragment = document.createDocumentFragment();
    for (const [text, className] of [[title, titleClass], [caption, captionClass]]) {
      if (text === '') continue;
      const span = document.createElement('span');
      span.className = className;
      renderText(text, span);
      fragment.appendChild(span);
    }
    return fragment;
  });
}

export class CompactView {
  private readonly group = new Group();
  private readonly leaves: ModuleTile[];
  private readonly titles: Label[];
  private readonly loopLabels: Label[];
  private readonly chip: Label;
  private readonly notes: Label[];
  private readonly braces: LineBatch;
  private readonly horizon: (node: Node, offset: number) => number | undefined;
  private plates: PlateMesh[] = [];
  private plateKey = '';
  private usedScale = 0;
  plan: CompactPlan | undefined;

  constructor(
    private readonly c: SceneController,
    private readonly spec: CompactViewSpec,
  ) {
    const steps = c.steps;
    if (steps === undefined) throw new Error('compact_view: the scene has no steps');
    this.leaves = tileLeaves(steps.tiles);
    const placed = this.leaves.flatMap((tile) => tile.caches);
    const missing = spec.infer.caches.find((decl) => !placed.includes(decl));
    if (missing !== undefined) throw new Error(`compact_view: no module tile computes the cached tensor '${missing.nodes[0].name}'`);
    const positions = positionsOf(c.g, spec.infer.input.shape[0]);
    this.horizon = (node, offset) => positions.horizon(node, offset);
    this.titles = [...this.leaves, undefined].map((tile) => {
      const label = c.labels.add('kk', 'vf-tile');
      label.anchor = 'left-middle';
      const title = tile?.title ?? SAMPLE_TITLE;
      const caption = tile === undefined ? '' : captionOf(tile);
      twoLines(label, `${title}\n${caption}`, title, caption, 'vf-tile-title', 'vf-tile-caption');
      return label;
    });
    this.loopLabels = steps.tiles.filter((tile) => tile.children.length > 0).map((tile) => {
      const label = c.labels.add('kk', 'vf-tile-loop');
      label.anchor = 'right-middle';
      twoLines(label, `${tile.title}\n${tile.subtitle ?? ''}`, tile.title, tile.subtitle ?? '', 'vf-mod-title', 'vf-mod-sub');
      return label;
    });
    this.chip = c.labels.add('kk', 'vf-tile-chip');
    this.chip.anchor = 'right-middle';
    this.notes = spec.infer.caches.map(() => {
      const label = c.labels.add('kk', 'vf-tile-note');
      label.anchor = 'bottom-center';
      return label;
    });
    this.braces = new LineBatch(c.theme.ink);
    this.group.add(this.braces.object);
    c.scene.add(this.group);
  }

  get active(): boolean {
    return this.c.state.generation?.compact !== undefined;
  }

  private pixelsPerUnit(): number {
    const camera = this.c.camera;
    return (this.c.height / (camera.top - camera.bottom)) * COMPACT_MIN_ZOOM;
  }

  private pool(t: number): CompactPool | undefined {
    const view = this.c.kvPool;
    const spec = this.c.options.kvPool;
    if (view === undefined || spec === undefined) return undefined;
    const steps = view.steps();
    return { blockSize: spec.blockSize, table: steps[engineStepAt(steps, t)].requests.find((r) => r.id === PAGE_REQUEST)?.table ?? [] };
  }

  layout(resolved: ResolvedLayout): ResolvedLayout {
    const generation = this.c.state.generation;
    if (generation?.compact === undefined) {
      this.plan = undefined;
      return resolved;
    }
    const tokens = pageTokens(this.spec.infer);
    const t = generation.position;
    this.titles[this.leaves.length].setFlag('vf-tile-sampled', generation.compact.sample);
    const sampled = generation.compact.sample ? `抽到 ${tokens[t + 1]}` : '';
    twoLines(this.titles[this.leaves.length], `${SAMPLE_TITLE}\n${sampled}`, SAMPLE_TITLE, sampled, 'vf-tile-title', 'vf-tile-caption');
    this.chip.setText(`位置 ${t}：${tokens[t]}`);
    this.usedScale = this.pixelsPerUnit();
    const widest = Math.max(...this.titles.map((label) => label.measure()[0]));
    const tallest = Math.max(...this.titles.map((label) => label.measure()[1]));
    const direction = this.c.rig.direction;
    const up = UP.clone().addScaledVector(direction, -direction.y).normalize();
    const side = new Vector3().crossVectors(UP, direction).normalize();
    this.plan = compactBlocks(resolved, this.c.steps!.tiles, generation, this.pool(t), {
      tileWidth: Math.max(TILE_MIN_WIDTH, (widest + 2 * TILE_PAD_PX) / this.usedScale),
      tileDepth: Math.max(TILE_MIN_DEPTH, (tallest + 2 * TILE_PAD_Y_PX) / (this.usedScale * Math.max(0.1, Math.abs(up.z)))),
      up: [up.x, up.y, up.z],
      side: [side.x, side.y, side.z],
      tokens,
      horizon: this.horizon,
    });
    return this.plan.layout;
  }

  private chipRoom(): number {
    return (this.chip.measure()[0] + TILE_PAD_PX) / this.usedScale + CHIP_GAP;
  }

  points(): Vec3[] {
    const plan = this.plan;
    if (plan === undefined) return [];
    const left = -this.chipRoom() - BRACE_GAP - BRACE_LABEL_GAP * braceScale(COMPACT_MIN_ZOOM) - Math.max(0, ...this.loopLabels.map((label) => label.measure()[0])) / this.usedScale;
    return plan.tiles.flatMap((tile) => [
      [left, tile.y, tile.z[0]],
      [tile.x[1], tile.y, tile.z[1]],
    ]);
  }

  litPoints(): Vec3[] {
    const plan = this.plan;
    if (plan === undefined) return [];
    const lit = plan.tiles.filter((tile) => tile.state === 'lit');
    const ys = new Set(lit.map((tile) => tile.y));
    return [
      ...lit.flatMap((tile): Vec3[] => [
        [tile.x[0], tile.y, tile.z[0]],
        [tile.x[1], tile.y, tile.z[1]],
      ]),
      ...plan.boards.filter((board) => ys.has(board.y)).flatMap((board): Vec3[] => [
        [board.x[0], board.y, board.z[0]],
        [board.x[1], board.y, board.z[1]],
      ]),
    ];
  }

  update(): void {
    const c = this.c;
    const plan = this.plan;
    const generation = c.state.generation;
    const shown = plan !== undefined && generation !== undefined && !c.lod;
    this.group.visible = shown;
    for (const label of [...this.titles, ...this.loopLabels, this.chip, ...this.notes]) label.visible = false;
    if (!shown) return;
    const theme = c.theme;
    this.syncPlates(plan);
    const fill = (state: TileState) => (state === 'lit' ? mix(theme.line, theme.pos, LIT_MIX) : state === 'done' ? theme.line : mix(theme.line, theme.surface, IDLE_MIX));
    plan.tiles.forEach((tile, i) => {
      const plate = this.plates[i];
      plate.setTheme(theme);
      plate.setFill(fill(tile.state));
      plate.update({ revealed: tile.state !== 'idle', far: false, seeThrough: false }, [], [], 0);
      const label = this.titles[i];
      label.pos = [tile.x[0] + TILE_PAD_PX / this.usedScale, tile.y, (tile.z[0] + tile.z[1]) / 2];
      label.setFlag('vf-tile-lit', tile.state === 'lit');
      label.setFlag('vf-tile-idle', tile.state === 'idle');
      label.visible = true;
    });
    const running = plan.tiles.map((tile) => tile.state).lastIndexOf('lit');
    if (running >= 0) {
      const tile = plan.tiles[running];
      this.chip.pos = [tile.x[0] - CHIP_GAP, tile.y, (tile.z[0] + tile.z[1]) / 2];
      this.chip.visible = true;
    }
    const x = -this.chipRoom() - BRACE_GAP;
    const z = (plan.tiles[0].z[0] + plan.tiles[0].z[1]) / 2;
    const scale = braceScale(c.rig.zoomNow());
    const braces = plan.loops.map(({ y }, k) => {
      const brace: Brace = { module: { title: '', nodes: [] }, side: 'left', thick: false, nodes: [], column: 0, x, y0: y[0] - 0.4, y1: y[1] + 0.4, z, cx: x, cz: z };
      const label = this.loopLabels[k];
      label.pos = braceTip(brace, 0, scale);
      label.visible = true;
      return { points: bracePoints(brace, 0, scale), opacity: 0.55 };
    });
    this.braces.setColor(theme.ink);
    this.braces.setPolylines(braces);
    plan.boards.forEach((board) => {
      const k = this.spec.infer.caches.indexOf(board.decl);
      const entry = generation.caches.find((e) => e.decl === board.decl)!;
      const label = this.notes[k];
      const text = entry.kind === 'state' ? `固定 ${entry.values} 个数` : plan.shared === board ? SHARED_TABLE_TEXT : '';
      if (text === '' || label.visible) return;
      label.setText(text);
      label.pos = [(board.x[0] + board.x[1]) / 2, board.y, board.z[1] + NOTE_FRONT];
      label.visible = true;
    });
  }

  private syncPlates(plan: CompactPlan): void {
    const key = plan.tiles.map((tile) => `${tile.y}:${tile.x[1]}`).join('|');
    if (key === this.plateKey) return;
    this.plateKey = key;
    for (const plate of this.plates) {
      this.group.remove(plate.group);
      plate.dispose();
    }
    this.plates = plan.tiles.map((tile) => {
      const plate = new PlateMesh({ plate: { x: (tile.x[0] + tile.x[1]) / 2, y: tile.y, z: (tile.z[0] + tile.z[1]) / 2, w: tile.x[1] - tile.x[0], d: tile.z[1] - tile.z[0] }, frameRows: [], elements: [] }, this.c.theme);
      this.group.add(plate.group);
      return plate;
    });
  }

  dispose(): void {
    this.c.scene.remove(this.group);
    for (const plate of this.plates) plate.dispose();
    this.braces.dispose();
  }
}
