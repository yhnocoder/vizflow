import { BufferGeometry, Color, DoubleSide, Group, LineBasicMaterial, LineLoop, Mesh, MeshBasicMaterial, PlaneGeometry, Vector3 } from 'three';
import type { Node } from '../autograd';
import { positionsOf } from '../explain';
import { renderText } from '../formula';
import { type CompactViewSpec, type ModuleTile, PAGE_REQUEST, type TileState, engineStepAt, pageTokens, tileLeaves } from '../steps';
import { mix } from './colors';
import { COMPACT_MAX_ZOOM, type CompactPlan, type CompactPool, type CompactText, compactBlocks } from './compactlayout';
import type { SceneController } from './controller';
import type { Box } from './avoid';
import { LABEL_SAFE_MARGIN, type Label } from './labels';
import type { ResolvedLayout, Vec3 } from './layout';
import { LineBatch } from './lines';
import { BRACE_LABEL_GAP, BRACE_WIDTH, type Brace, bracePoints, braceScale, braceTip } from './modules';

export const SAMPLE_TITLE = '采样';

const TILE_PAD_PX = 14;
const TILE_PAD_Y_PX = 10;
const CHIP_GAP_PX = 12;
const BRACE_GAP_PX = 10;
const LIT_MIX = 0.45;
const IDLE_MIX = 0.55;
const OUTLINE_OPACITY = 0.5;

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

function along(point: Vec3, axis: Vector3, distance: number): Vec3 {
  return [point[0] + axis.x * distance, point[1] + axis.y * distance, point[2] + axis.z * distance];
}

interface Card {
  group: Group;
  fill: MeshBasicMaterial;
  outline: LineBasicMaterial;
  geometries: BufferGeometry[];
}

export class CompactView {
  private readonly group = new Group();
  private readonly leaves: ModuleTile[];
  private readonly titles: Label[];
  private readonly loopLabels: Label[];
  private readonly chip: Label;
  private readonly notes: Label[];
  private readonly probes: Record<CompactText, Label>;
  private readonly braces: LineBatch;
  private readonly horizon: (node: Node, offset: number) => number | undefined;
  private up = new Vector3(0, 1, 0);
  private cards: Card[] = [];
  private cardKey = '';
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
    this.chip.anchor = 'left-middle';
    this.notes = spec.infer.caches.map(() => {
      const label = c.labels.add('kk', 'vf-tile-note');
      label.anchor = 'top-center';
      return label;
    });
    this.probes = { row: c.labels.add('kk'), note: c.labels.add('kk', 'vf-tile-note') };
    for (const probe of Object.values(this.probes)) probe.visible = false;
    this.braces = new LineBatch(c.theme.ink);
    this.group.add(this.braces.object);
    c.scene.add(this.group);
    c.controls.addEventListener('change', this.face);
  }

  get active(): boolean {
    return this.c.state.generation?.compact !== undefined;
  }

  get zoom(): number {
    return this.plan?.zoom ?? 1;
  }

  private perUnit(): number {
    const camera = this.c.camera;
    return this.c.height / (camera.top - camera.bottom);
  }

  private pool(t: number): CompactPool | undefined {
    const view = this.c.kvPool;
    const spec = this.c.options.kvPool;
    if (view === undefined || spec === undefined) return undefined;
    const steps = view.steps();
    return { blockSize: spec.blockSize, table: steps[engineStepAt(steps, t)].requests.find((r) => r.id === PAGE_REQUEST)?.table ?? [] };
  }

  private measure(text: string, kind: CompactText): [number, number] {
    const probe = this.probes[kind];
    probe.setText(text);
    return probe.measure();
  }

  private leftPx(): number {
    const loop = Math.max(0, ...this.loopLabels.map((label) => label.measure()[0]));
    const brace = (BRACE_LABEL_GAP + BRACE_WIDTH) * this.perUnit() * Math.max(1, COMPACT_MAX_ZOOM);
    return 2 * LABEL_SAFE_MARGIN + loop + brace + BRACE_GAP_PX;
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
    const chipRoom = this.chip.measure()[0] + CHIP_GAP_PX;
    const widest = Math.max(...this.titles.map((label) => Math.max(label.measure()[0], this.titleWidth(label) + chipRoom)));
    const tallest = Math.max(...this.titles.map((label) => label.measure()[1]));
    const direction = this.c.rig.direction;
    this.up = UP.clone().addScaledVector(direction, -direction.y).normalize();
    const side = new Vector3().crossVectors(UP, direction).normalize();
    this.plan = compactBlocks(resolved, this.c.steps!.tiles, generation, this.pool(t), {
      up: [this.up.x, this.up.y, this.up.z],
      side: [side.x, side.y, side.z],
      tokens,
      horizon: this.horizon,
      perUnit: this.perUnit(),
      room: this.c.width - this.leftPx(),
      tile: [widest + 2 * TILE_PAD_PX, tallest + 2 * TILE_PAD_Y_PX],
      measure: (text, kind) => this.measure(text, kind),
    });
    return this.plan.layout;
  }

  private corners(center: Vec3, size: [number, number], left = 0): Vec3[] {
    const side = this.facing();
    const w = size[0];
    const h = size[1] / this.upY();
    return [
      along(along(center, side, -w / 2 - left), UP, -h / 2),
      along(along(center, side, -w / 2 - left), UP, h / 2),
      along(along(center, side, w / 2), UP, -h / 2),
      along(along(center, side, w / 2), UP, h / 2),
    ];
  }

  private leftEdge(): number {
    return this.plan === undefined ? 0 : (this.leftPx() - 2 * LABEL_SAFE_MARGIN) / this.plan.scale;
  }

  points(): Vec3[] {
    const plan = this.plan;
    if (plan === undefined) return [];
    const left = this.leftEdge();
    return [...plan.tiles.flatMap((tile) => this.corners(tile.center, tile.size, left)), ...plan.extent];
  }

  litPoints(): Vec3[] {
    const plan = this.plan;
    if (plan === undefined) return [];
    const lit = plan.tiles.filter((tile) => tile.state === 'lit');
    const ys = new Set(lit.map((tile) => tile.y));
    return [
      ...lit.flatMap((tile) => this.corners(tile.center, tile.size)),
      ...plan.boards.filter((board) => ys.has(board.y)).flatMap((board): Vec3[] => [
        [board.x[0], board.y, board.z[0]],
        [board.x[1], board.y, board.z[1]],
      ]),
    ];
  }

  screenBoxes(): Box[] {
    const plan = this.plan;
    if (plan === undefined || !this.group.visible) return [];
    const { camera, width, height } = this.c;
    return plan.tiles.map((tile) => {
      const xs: number[] = [];
      const ys: number[] = [];
      for (const corner of this.corners(tile.center, tile.size)) {
        const p = new Vector3(...corner).project(camera);
        xs.push(((p.x + 1) / 2) * width);
        ys.push(((1 - p.y) / 2) * height);
      }
      return { x: Math.min(...xs), y: Math.min(...ys), w: Math.max(...xs) - Math.min(...xs), h: Math.max(...ys) - Math.min(...ys) };
    });
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
    this.syncCards(plan);
    const fill = (state: TileState) => (state === 'lit' ? mix(theme.line, theme.pos, LIT_MIX) : state === 'done' ? theme.line : mix(theme.line, theme.surface, IDLE_MIX));
    plan.tiles.forEach((tile, i) => {
      const card = this.cards[i];
      card.fill.color.setRGB(...fill(tile.state));
      card.outline.color.setRGB(...theme.ink);
      card.outline.opacity = tile.state === 'idle' ? OUTLINE_OPACITY / 2 : OUTLINE_OPACITY;
      const label = this.titles[i];
      label.setFlag('vf-tile-lit', tile.state === 'lit');
      label.setFlag('vf-tile-idle', tile.state === 'idle');
      label.visible = true;
    });
    this.chip.visible = plan.tiles.some((tile) => tile.state === 'lit');
    for (const label of this.loopLabels) label.visible = true;
    this.braces.setColor(theme.ink);
    this.place(plan);
    for (const board of plan.boards) {
      if (board.note === undefined) continue;
      const label = this.notes[this.spec.infer.caches.indexOf(board.decl)];
      if (label.visible) continue;
      label.setText(board.note.text);
      label.pos = board.note.pos;
      label.visible = true;
    }
  }

  private facing(): Vector3 {
    const view = this.c.camera.position.clone().sub(this.c.controls.target);
    return new Vector3(view.z, 0, -view.x).normalize();
  }

  private place(plan: CompactPlan): void {
    const side = this.facing();
    const pad = TILE_PAD_PX / plan.scale;
    plan.tiles.forEach((tile, i) => {
      this.titles[i].pos = along(tile.center, side, -tile.size[0] / 2 + pad);
    });
    const running = plan.tiles.map((tile) => tile.state).lastIndexOf('lit');
    if (running >= 0) {
      const tile = plan.tiles[running];
      const title = this.titles[running];
      const lift = (title.measure()[1] - this.titleHeight(title)) / 2 / plan.scale / this.upY();
      this.chip.pos = along(along(tile.center, side, -tile.size[0] / 2 + pad + (this.titleWidth(title) + CHIP_GAP_PX) / plan.scale), UP, lift);
    }
    const first = plan.tiles[0];
    const base = along(first.center, side, -first.size[0] / 2 - BRACE_GAP_PX / plan.scale);
    const halfY = first.size[1] / 2 / this.upY();
    const scale = braceScale(this.c.rig.zoomNow());
    this.braces.setPolylines(
      plan.loops.map(({ y }, k) => {
        const brace: Brace = { module: { title: '', nodes: [] }, side: 'left', thick: false, nodes: [], column: 0, x: base[0], y0: y[0] - halfY, y1: y[1] + halfY, z: base[2], cx: base[0], cz: base[2] };
        this.loopLabels[k].pos = braceTip(brace, 0, scale);
        return { points: bracePoints(brace, 0, scale), opacity: 0.55 };
      }),
    );
  }

  private upY(): number {
    return Math.max(1e-6, this.up.y);
  }

  private titleWidth(label: Label): number {
    label.measure();
    return (label.el.firstElementChild as HTMLElement | null)?.offsetWidth ?? 0;
  }

  private titleHeight(label: Label): number {
    label.measure();
    return (label.el.firstElementChild as HTMLElement | null)?.offsetHeight ?? 0;
  }

  private readonly face = (): void => {
    const side = this.facing();
    for (const card of this.cards) card.group.rotation.set(0, Math.atan2(-side.z, side.x), 0);
    if (this.plan !== undefined && this.group.visible) this.place(this.plan);
  };

  private syncCards(plan: CompactPlan): void {
    const key = plan.tiles.map((tile) => `${tile.center.join(',')}:${tile.size.join(',')}`).join('|');
    if (key === this.cardKey) return;
    this.cardKey = key;
    this.disposeCards();
    this.cards = plan.tiles.map((tile) => {
      const w = tile.size[0];
      const h = tile.size[1] / this.upY();
      const group = new Group();
      group.position.set(...tile.center);
      const plane = new PlaneGeometry(w, h);
      const fill = new MeshBasicMaterial({ color: new Color(...this.c.theme.line), side: DoubleSide, depthWrite: false });
      const mesh = new Mesh(plane, fill);
      mesh.renderOrder = -1;
      group.add(mesh);
      const edge = new BufferGeometry().setFromPoints([new Vector3(-w / 2, -h / 2, 0), new Vector3(w / 2, -h / 2, 0), new Vector3(w / 2, h / 2, 0), new Vector3(-w / 2, h / 2, 0)]);
      const outline = new LineBasicMaterial({ color: new Color(...this.c.theme.ink), transparent: true, opacity: OUTLINE_OPACITY, depthWrite: false });
      const loop = new LineLoop(edge, outline);
      loop.renderOrder = -1;
      group.add(loop);
      this.group.add(group);
      return { group, fill, outline, geometries: [plane, edge] };
    });
    this.face();
  }

  private disposeCards(): void {
    for (const card of this.cards) {
      this.group.remove(card.group);
      card.fill.dispose();
      card.outline.dispose();
      for (const geometry of card.geometries) geometry.dispose();
    }
    this.cards = [];
  }

  dispose(): void {
    this.c.controls.removeEventListener('change', this.face);
    this.c.scene.remove(this.group);
    this.disposeCards();
    this.braces.dispose();
  }
}
