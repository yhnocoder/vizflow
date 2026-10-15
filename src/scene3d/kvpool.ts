import { BoxGeometry, Group, Mesh, MeshLambertMaterial, type Raycaster } from 'three';
import type { Node } from '../autograd';
import { offsetsByPosition, positionsOf } from '../explain';
import { renderText } from '../formula';
import { type PagedRequestState, type PagedStep, simulateVllmPages } from '../sim';
import { type CacheDecl, type KvBlocksSpec, type KvSlotSource, PAGE_REQUEST, engineStepAt, kvScenario, kvTokenText, pageTokens, slotOf, slotSources, touchedPositions, writtenPositions } from '../steps';
import { indexOf } from '../tensor';
import type { LinearRgb } from '../theme';
import { BarMesh } from './blocks';
import { VALUE_MIN_ZOOM } from './blockview';
import { INACTIVE_MIX, barColor, mix, outlineOpacity } from './colors';
import type { SceneController } from './controller';
import { graphPoints } from './framing';
import type { Label } from './labels';
import { PLATE_THICKNESS, type Vec3 } from './layout';
import { LineBatch, type Polyline, rectPoints } from './lines';
import { anchorOut, elementOf, renderableOf } from './renderable';
import { barHeight } from './scales';

export const POOL_TITLE = 'vLLM 的物理块池';
export const NULL_BLOCK_TEXT = '空块';

const POOL_GAP = 4;
const SLOT_PITCH_X = 0.8;
const SLOT_PITCH_Z = 1.5;
const BAR_FOOTPRINT: [number, number] = [0.6, 0.6];
const BLOCK_PAD = 0.25;
const BLOCK_GAP = 0.5;
const TABLE_GAP = 3;
const TABLE_WIDTH = 1.6;
const TABLE_SPACING = 2.6;
const TABLE_HEAD = 0.5;
const TABLE_LABEL_ROOM = 1;
const TITLE_BACK = 1.4;
const TOKEN_INSET = 0.2;
const ARC_RISE = 0.15;
const ARC_POINTS = 16;
const SLOT_INSET = 0.05;
const PLACEHOLDER_HEIGHT = 0.3;
const OUTLINE_LIFT = 0.005;
const WRITE_TARGET_LIFT = 0.15;
const BLOCK_OUTLINE = { current: 0.9, used: 0.5, free: 0.25, reserved: 0.12 };
const SLOT_OUTLINE = { current: 0.8, filled: 0.45, empty: 0.25, reserved: 0.1 };
const LINK_CURRENT = 0.85;
const LINK_IDLE = 0.3;
const WRITE_OPACITY = 0.9;

type PoolPick = { block: number } | { request: string; index: number };

interface PoolFrame {
  x0: number;
  y: number;
  zStart: number;
  dir: number;
}

interface TableLabels {
  head: Label;
  blocks: Label[];
  tokens: Label[];
}

export class KvPoolView {
  private readonly group = new Group();
  private readonly node: Node;
  private readonly decl: CacheDecl;
  private readonly paged: CacheDecl[];
  private readonly rows: number[][];
  private readonly cols: number;
  private readonly width: number;
  private readonly depth: number;
  private readonly bars: BarMesh;
  private readonly physicalGeometry: BoxGeometry;
  private readonly logicalGeometry: BoxGeometry;
  private readonly physicalMaterial = new MeshLambertMaterial();
  private readonly logicalMaterial = new MeshLambertMaterial();
  private readonly physical: Mesh[];
  private readonly logical = new Map<string, Mesh[]>();
  private readonly outlines: LineBatch;
  private readonly links: LineBatch;
  private readonly writes: LineBatch;
  private readonly title: Label;
  private readonly headers: Label[];
  private readonly tableLabels = new Map<string, TableLabels>();
  private simulated: { key: string; steps: PagedStep[] } | undefined;
  private step: PagedStep | undefined;

  constructor(
    private readonly c: SceneController,
    private readonly spec: KvBlocksSpec,
  ) {
    const positions = positionsOf(c.g, spec.infer.input.shape[0]);
    const rowsOf = (node: Node) => offsetsByPosition(positions, node);
    this.paged = spec.infer.caches.filter((decl) => rowsOf(decl.nodes[0]).length > 1);
    const decl = this.paged[0];
    if (decl === undefined) throw new Error('kv_blocks scene: no cache: declaration holds one row per position');
    this.decl = decl;
    this.node = decl.nodes[0];
    if (renderableOf(c, this.node) === undefined) throw new Error(`kv_blocks scene: the cached tensor '${this.node.name}' is not drawn in the scene`);
    this.rows = rowsOf(this.node);
    this.cols = Math.max(...this.rows.map((r) => r.length));
    this.width = this.cols * SLOT_PITCH_X + 2 * BLOCK_PAD;
    this.depth = spec.blockSize * SLOT_PITCH_Z + 2 * BLOCK_PAD;
    this.physicalGeometry = new BoxGeometry(this.width, PLATE_THICKNESS, this.depth);
    this.logicalGeometry = new BoxGeometry(TABLE_WIDTH, PLATE_THICKNESS, this.depth);
    this.bars = new BarMesh(spec.numBlocks * spec.blockSize * this.cols, BAR_FOOTPRINT, c.theme);
    this.physical = Array.from({ length: spec.numBlocks }, (_, block) => this.plate(this.physicalGeometry, this.physicalMaterial, { block }));
    this.outlines = new LineBatch(c.theme.ink, c.theme.surface);
    this.links = new LineBatch(c.theme.ink);
    this.writes = new LineBatch(c.theme.ink);
    this.group.add(this.bars.group, this.outlines.object, this.links.object, this.writes.object);
    c.scene.add(this.group);
    this.title = c.labels.add('boxlb', 'vf-pool-title');
    this.title.anchor = 'above-right';
    this.headers = Array.from({ length: spec.numBlocks }, () => {
      const label = c.labels.add('kk', 'vf-pool-lb');
      label.anchor = 'right-middle';
      return label;
    });
  }

  private plate(geometry: BoxGeometry, material: MeshLambertMaterial, pick: PoolPick): Mesh {
    const mesh = new Mesh(geometry, material);
    mesh.userData.kvPool = pick;
    this.group.add(mesh);
    return mesh;
  }

  steps(): PagedStep[] {
    const key = pageTokens(this.spec.infer).join('\u0000');
    if (this.simulated?.key !== key) this.simulated = { key, steps: simulateVllmPages(kvScenario(this.spec)) };
    return this.simulated.steps;
  }

  private shown(): boolean {
    return this.c.state.generation !== undefined && this.c.state.generation.compact === undefined && !this.c.lod;
  }

  private frame(): PoolFrame {
    const block = renderableOf(this.c, this.node)!.block;
    const zOf = (position: number) => elementOf(block, this.node, indexOf(this.node.shape, this.rows[position][0])).z;
    const dir = zOf(1) < zOf(0) ? -1 : 1;
    const zs = block.elements.map((e) => e.z);
    const total = this.spec.numBlocks * this.depth + (this.spec.numBlocks - 1) * BLOCK_GAP;
    const x0 = Math.max(...graphPoints(this.c).map((p) => p[0])) + POOL_GAP;
    return { x0, y: block.elements[0].y, zStart: (Math.min(...zs) + Math.max(...zs)) / 2 - (dir * total) / 2, dir };
  }

  private blockZ(f: PoolFrame, i: number): number {
    return f.zStart + f.dir * (this.depth / 2 + i * (this.depth + BLOCK_GAP));
  }

  private slotZ(f: PoolFrame, i: number, j: number): number {
    return this.blockZ(f, i) + f.dir * (BLOCK_PAD + (j + 0.5) * SLOT_PITCH_Z - this.depth / 2);
  }

  private poolX(f: PoolFrame): number {
    return f.x0 + this.width / 2;
  }

  private colX(f: PoolFrame, q: number): number {
    return f.x0 + BLOCK_PAD + (q + 0.5) * SLOT_PITCH_X;
  }

  private tableX(f: PoolFrame, r: number): number {
    return f.x0 + this.width + TABLE_GAP + r * (TABLE_WIDTH + TABLE_SPACING) + TABLE_WIDTH / 2;
  }

  private requestIds(): string[] {
    return [PAGE_REQUEST, ...this.spec.others.map((o) => o.id)];
  }

  points(): Vec3[] {
    if (!this.shown()) return [];
    const f = this.frame();
    const zEnd = this.blockZ(f, this.spec.numBlocks - 1) + (f.dir * this.depth) / 2;
    const xEnd = this.tableX(f, this.requestIds().length - 1) + TABLE_WIDTH / 2 + TABLE_LABEL_ROOM;
    const top = f.y + renderableOf(this.c, this.node)!.cap;
    return [
      [f.x0, f.y, f.zStart - f.dir * TITLE_BACK],
      [f.x0, top, f.zStart],
      [f.x0, f.y, zEnd],
      [xEnd, f.y, f.zStart],
      [xEnd, f.y, zEnd],
    ];
  }

  followPoints(): Vec3[] {
    return this.c.state.active(this.node) ? this.points() : [];
  }

  update(): void {
    const c = this.c;
    const generation = c.state.generation;
    if (generation === undefined || !this.shown()) {
      this.group.visible = false;
      this.step = undefined;
      for (const label of this.labels()) label.visible = false;
      return;
    }
    this.group.visible = true;
    const steps = this.steps();
    const at = engineStepAt(steps, generation.position);
    const step = steps[at];
    const previous = at > 0 ? steps[at - 1] : undefined;
    this.step = step;
    const f = this.frame();
    const theme = c.theme;
    this.physicalMaterial.color.setRGB(...theme.line);
    this.logicalMaterial.color.setRGB(...theme.bg);
    this.outlines.setColor(theme.ink, theme.surface);
    this.links.setColor(theme.ink);
    this.writes.setColor(theme.ink);
    this.bars.setTheme(theme);
    const bs = this.spec.blockSize;
    const revealed = c.state.revealed(this.node);
    const detailed = !c.follow || c.rig.zoomNow() >= VALUE_MIN_ZOOM;
    const sources = slotSources(step, bs);
    const written = new Map(step.requests.map((r) => [r.id, writtenPositions(r)]));
    const current = (s: KvSlotSource) => {
      const [lo, hi] = written.get(s.request)!;
      return s.position >= lo && s.position < hi;
    };
    const drawn = (s: KvSlotSource | null): s is KvSlotSource => s !== null && (revealed || !current(s));
    const touched = (r: PagedRequestState, k: number) => {
      const [lo, hi] = touchedPositions(previous, r);
      return hi > lo && k * bs < hi && (k + 1) * bs > lo;
    };
    const touchedBlocks = new Set(step.requests.flatMap((r) => r.table.filter((_, k) => touched(r, k))));
    this.updateBars(f, sources, current, drawn);
    const outlines: Polyline[] = [];
    const px = this.poolX(f);
    (step.blocks ?? []).forEach((block, i) => {
      const z = this.blockZ(f, i);
      this.physical[i].position.set(px, f.y - PLATE_THICKNESS / 2 - OUTLINE_LIFT, z);
      const reserved = i === 0;
      const opacity = reserved ? BLOCK_OUTLINE.reserved : touchedBlocks.has(i) ? BLOCK_OUTLINE.current : block.refCnt > 0 || block.hashed ? BLOCK_OUTLINE.used : BLOCK_OUTLINE.free;
      outlines.push({ points: rect(px, f.y, z, this.width, this.depth), opacity });
      for (let j = 0; j < bs; j++) {
        const source = sources[i][j];
        const slot = reserved ? SLOT_OUTLINE.reserved : !drawn(source) ? SLOT_OUTLINE.empty : current(source) ? SLOT_OUTLINE.current : SLOT_OUTLINE.filled;
        outlines.push({ points: rect(px, f.y, this.slotZ(f, i, j), this.width - BLOCK_PAD, SLOT_PITCH_Z - 2 * SLOT_INSET), opacity: slot });
      }
      const header = this.headers[i];
      header.pos = [f.x0, f.y, z];
      header.visible = detailed;
      header.setText(reserved ? NULL_BLOCK_TEXT : block.refCnt > 0 ? `块 ${i} · 引用 ${block.refCnt}` : block.hashed ? `块 ${i} · 可复用` : `块 ${i}`);
    });
    const links: Polyline[] = [];
    for (const [r, id] of this.requestIds().entries()) {
      const request = step.requests.find((q) => q.id === id);
      const count = request?.table.length ?? 0;
      const meshes = this.logicalMeshes(id, count);
      const labels = this.labelsOf(id, count);
      const x = this.tableX(f, r);
      labels.head.pos = [x, f.y, f.zStart - f.dir * TABLE_HEAD];
      labels.head.visible = detailed && request !== undefined;
      meshes.forEach((mesh, k) => {
        const on = request !== undefined && k < request.table.length;
        mesh.visible = on;
        labels.blocks[k].visible = detailed && on;
        for (let j = 0; j < bs; j++) labels.tokens[k * bs + j].visible = detailed && on && k * bs + j < request.computed;
        if (!on) return;
        const z = this.blockZ(f, k);
        mesh.position.set(x, f.y - PLATE_THICKNESS / 2 - OUTLINE_LIFT, z);
        labels.blocks[k].pos = [x + TABLE_WIDTH / 2, f.y, z];
        const fresh = touched(request, k);
        outlines.push({ points: rect(x, f.y, z, TABLE_WIDTH, this.depth), opacity: fresh ? BLOCK_OUTLINE.current : BLOCK_OUTLINE.used });
        for (let j = 0; j < bs; j++) {
          const t = k * bs + j;
          const slotZ = this.slotZ(f, k, j);
          outlines.push({ points: rect(x, f.y, slotZ, TABLE_WIDTH - BLOCK_PAD, SLOT_PITCH_Z - 2 * SLOT_INSET), opacity: t < request.computed ? SLOT_OUTLINE.filled : SLOT_OUTLINE.empty });
          const token = labels.tokens[t];
          token.pos = [x - TABLE_WIDTH / 2 + TOKEN_INSET, f.y, slotZ];
          token.setText(t < request.computed ? kvTokenText(request.tokens[t]) : '');
        }
        const target = request.table[k];
        links.push({ points: arc([x - TABLE_WIDTH / 2, f.y, z], [f.x0 + this.width, f.y, this.blockZ(f, target)]), opacity: fresh ? LINK_CURRENT : LINK_IDLE });
      });
    }
    this.outlines.setPolylines(outlines);
    this.links.setPolylines(links);
    this.writes.setPolylines(revealed ? this.writeLines(f, step) : []);
    this.title.pos = [f.x0, f.y, f.zStart - f.dir * TITLE_BACK];
    this.title.visible = detailed;
    const lines = this.titleLines();
    this.title.setContent(lines.join('\n'), () => {
      const fragment = document.createDocumentFragment();
      for (const line of lines) {
        const el = document.createElement('div');
        renderText(line, el);
        fragment.appendChild(el);
      }
      return fragment;
    });
  }

  private titleLines(): string[] {
    const shared = [...this.decl.nodes.slice(1).map((n) => `$${n.tex}$`), ...(this.paged.length > 1 ? ['其他层的缓存'] : [])];
    return [POOL_TITLE, `每格存一个位置的 $${this.node.tex}$`, ...(shared.length === 0 ? [] : [`${shared.join('和')}按同一张块表存放`])];
  }

  private updateBars(f: PoolFrame, sources: (KvSlotSource | null)[][], current: (s: KvSlotSource) => boolean, drawn: (s: KvSlotSource | null) => s is KvSlotSource): void {
    const theme = this.c.theme;
    const renderable = renderableOf(this.c, this.node)!;
    const values = this.node.value.data;
    const elements: { x: number; y: number; z: number }[] = [];
    const heights: number[] = [];
    const clipped: boolean[] = [];
    const colors: LinearRgb[] = [];
    const outline: number[] = [];
    for (let i = 0; i < this.spec.numBlocks; i++) {
      for (let j = 0; j < this.spec.blockSize; j++) {
        const source = sources[i][j];
        for (let q = 0; q < this.cols; q++) {
          elements.push({ x: this.colX(f, q), y: f.y, z: this.slotZ(f, i, j) });
          if (!drawn(source)) {
            heights.push(0);
            clipped.push(false);
            colors.push(theme.surface);
            outline.push(0);
            continue;
          }
          const on = current(source);
          outline.push(outlineOpacity({ active: on, far: false, tile: false }));
          if (source.request === PAGE_REQUEST) {
            const value = values[this.rows[source.position][q]];
            const bar = barHeight(value, renderable.scale.forward, true, renderable.cap);
            heights.push(bar.height);
            clipped.push(bar.clipped);
            colors.push(barColor(value, theme, { active: on, far: false, focused: true }));
          } else {
            heights.push(PLACEHOLDER_HEIGHT);
            clipped.push(false);
            colors.push(on ? theme.lineStrong : mix(theme.lineStrong, theme.surface, INACTIVE_MIX));
          }
        }
      }
    }
    this.bars.update(elements, { heights, clipped, colors, outlineOpacity: outline, seeThrough: false });
  }

  private writeLines(f: PoolFrame, step: PagedStep): Polyline[] {
    const c = this.c;
    const request = step.requests.find((r) => r.id === PAGE_REQUEST);
    if (request === undefined) return [];
    const [lo, hi] = writtenPositions(request);
    const lines: Polyline[] = [];
    for (let t = lo; t < hi; t++) {
      const anchors = this.rows[t].flatMap((offset) => {
        const index = indexOf(this.node.shape, offset);
        if (c.state.row?.(this.node, index) === 'hidden') return [];
        const at = anchorOut(c, this.node, index);
        return at === undefined ? [] : [at];
      });
      if (anchors.length === 0) continue;
      const from = [0, 1, 2].map((axis) => anchors.reduce((sum, p) => sum + p[axis], 0) / anchors.length) as Vec3;
      const { block, offset } = slotOf(request.table, t, this.spec.blockSize);
      lines.push({ points: arc(from, [this.poolX(f), f.y + WRITE_TARGET_LIFT, this.slotZ(f, block, offset)]), opacity: WRITE_OPACITY, dashed: true });
    }
    return lines;
  }

  private logicalMeshes(id: string, count: number): Mesh[] {
    let meshes = this.logical.get(id);
    if (meshes === undefined) this.logical.set(id, (meshes = []));
    while (meshes.length < count) meshes.push(this.plate(this.logicalGeometry, this.logicalMaterial, { request: id, index: meshes.length }));
    return meshes;
  }

  private labelsOf(id: string, count: number): TableLabels {
    let labels = this.tableLabels.get(id);
    if (labels === undefined) {
      const head = this.c.labels.add('kk', 'vf-pool-lb');
      head.anchor = 'top-center';
      head.setText(`请求 ${id}`);
      this.tableLabels.set(id, (labels = { head, blocks: [], tokens: [] }));
    }
    while (labels.blocks.length < count) {
      const label = this.c.labels.add('kk', 'vf-pool-lb');
      label.anchor = 'left-middle';
      label.setText(String(labels.blocks.length));
      labels.blocks.push(label);
    }
    while (labels.tokens.length < count * this.spec.blockSize) {
      const label = this.c.labels.add('kk', 'vf-pool-tok');
      label.anchor = 'left-middle';
      labels.tokens.push(label);
    }
    return labels;
  }

  private labels(): Label[] {
    return [this.title, ...this.headers, ...[...this.tableLabels.values()].flatMap((t) => [t.head, ...t.blocks, ...t.tokens])];
  }

  cardAt(raycaster: Raycaster): DocumentFragment | undefined {
    const step = this.step;
    if (step === undefined || !this.group.visible) return undefined;
    const meshes = [...this.physical, ...[...this.logical.values()].flat()].filter((m) => m.visible);
    const hit = raycaster.intersectObjects(meshes, false)[0];
    if (hit === undefined) return undefined;
    const pick = hit.object.userData.kvPool as PoolPick;
    return 'block' in pick ? this.blockCard(step, pick.block) : this.logicalCard(step, pick.request, pick.index);
  }

  private blockCard(step: PagedStep, id: number): DocumentFragment {
    if (id === 0) return card(`${NULL_BLOCK_TEXT}（0 号块）`, [['用途', 'vLLM 保留的空块，不存数据，也不分配给请求']]);
    const block = step.blocks![id];
    return card(`物理块 ${id}`, [
      ['引用计数', String(block.refCnt)],
      ['块哈希', block.hashed ? '已登记，新请求的前缀与它相同时可以直接命中' : '没有，块写满以后才计算哈希'],
      ['存放的 token', block.tokens.map((t) => (t === null ? '空' : kvTokenText(t))).join(' ')],
      ['使用它的请求', block.owners.length === 0 ? '没有请求在用' : block.owners.join('、')],
    ]);
  }

  private logicalCard(step: PagedStep, id: string, k: number): DocumentFragment {
    const request = step.requests.find((r) => r.id === id)!;
    const bs = this.spec.blockSize;
    const tokens = Array.from({ length: bs }, (_, j) => k * bs + j).map((t) => (t < request.computed ? kvTokenText(request.tokens[t]) : '空'));
    const rows: [string, string][] = [
      ['对应物理块', String(request.table[k])],
      ['块表', `[${request.table.join(', ')}]`],
      ['位置', `${k * bs} 到 ${k * bs + bs - 1}`],
      ['存放的 token', tokens.join(' ')],
      ['已算', `${request.computed} 个位置`],
    ];
    if (request.hit > 0) rows.push(['命中前缀', `${request.hit} 个 token`]);
    return card(`请求 ${id} 的逻辑块 ${k}`, rows);
  }

  dispose(): void {
    this.c.scene.remove(this.group);
    this.bars.dispose();
    this.physicalGeometry.dispose();
    this.logicalGeometry.dispose();
    this.physicalMaterial.dispose();
    this.logicalMaterial.dispose();
    this.outlines.dispose();
    this.links.dispose();
    this.writes.dispose();
  }
}

function rect(x: number, y: number, z: number, w: number, d: number): number[] {
  return rectPoints(x, y + OUTLINE_LIFT, z, w, d);
}

function arc(from: Vec3, to: Vec3): number[] {
  const rise = ARC_RISE * Math.hypot(to[0] - from[0], to[2] - from[2]);
  const points: number[] = [];
  for (let k = 0; k <= ARC_POINTS; k++) {
    const s = k / ARC_POINTS;
    points.push(from[0] + (to[0] - from[0]) * s, from[1] + (to[1] - from[1]) * s + rise * 4 * s * (1 - s), from[2] + (to[2] - from[2]) * s);
  }
  return points;
}

function card(title: string, rows: [string, string][]): DocumentFragment {
  const fragment = document.createDocumentFragment();
  const head = document.createElement('h4');
  head.className = 'vf-pop-title';
  head.textContent = title;
  const dl = document.createElement('dl');
  dl.className = 'vf-rows';
  for (const [name, value] of rows) {
    const dt = document.createElement('dt');
    dt.textContent = name;
    const dd = document.createElement('dd');
    dd.textContent = value;
    dl.append(dt, dd);
  }
  fragment.append(head, dl);
  return fragment;
}
