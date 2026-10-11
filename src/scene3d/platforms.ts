import { CubicBezierCurve3, ConeGeometry, DoubleSide, Group, Mesh, MeshBasicMaterial, PlaneGeometry, TubeGeometry, Vector3 } from 'three';
import { type CommPhase, type CommTransfer, type Device, type MeshTopology, type Node, type Placements, commTransfers, rankDevice } from '../autograd';
import { renderText } from '../formula';
import { forEachIndex } from '../tensor';
import type { LinearRgb, ThemeColors } from '../theme';
import type { SceneController } from './controller';
import type { Label } from './labels';
import type { Block, Vec3 } from './layout';
import { LineBatch } from './lines';
import { dashed } from './blocks';
import { segmentDistance } from './edgelines';
import { MODE_TOKENS } from './modes';
import { bytesText, parallelMode } from './parallel';
import { valuesOf } from './renderable';
import { HEIGHT_CAP } from './scales';

export const PLATFORM_SHOWN_MAX = 4;
export const PLATFORM_GAP = 3;
export const ROW_GAP = 4;
export const PLACEHOLDER_WIDTH = 10;
export const FRAME_PAD = 0.8;
export const NODE_PAD = 1.4;
export const LINK_LIFT = 0.9;
export const LINK_RISE = 0.12;
export const LINK_SPREAD = 0.6;
export const LINK_BULGE = 2.5;
export const LINK_RADIUS = 0.045;
export const LINK_HOVER_DISTANCE = 7;
export const COMM_OPS: ReadonlySet<string> = new Set(['all_reduce', 'dispatch', 'combine']);

const LINK_SEGMENTS = 32;
const LINK_SIDES = 6;
const ARROW_LENGTH = 0.32;
const ARROW_RADIUS = 2.6;
const HOVER_WIDEN = 1.8;
const NODE_DROP = 0.05;
const LABEL_ROOM = 1.5;
const NODE_FILL_OPACITY = 0.55;
const FRAME_OPACITY = 0.45;

export type PlatformSlot = { kind: 'rank'; rank: number; device: Device } | { kind: 'hidden'; first: number; last: number };

export function platformSlots(topology: MeshTopology): PlatformSlot[] {
  const rank = (r: number): PlatformSlot => ({ kind: 'rank', rank: r, device: rankDevice(topology, r) });
  const n = topology.size;
  if (n <= PLATFORM_SHOWN_MAX) return Array.from({ length: n }, (_, r) => rank(r));
  return [rank(0), rank(1), { kind: 'hidden', first: 2, last: n - 2 }, rank(n - 1)];
}

export function slotOf(slots: readonly PlatformSlot[], rank: number): number {
  return slots.findIndex((s) => (s.kind === 'rank' ? s.rank === rank : rank >= s.first && rank <= s.last));
}

export function slotText(slot: PlatformSlot): string {
  return slot.kind === 'rank' ? `rank ${slot.rank}` : `rank ${slot.first} … rank ${slot.last}`;
}

export interface PlatformBox {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  z0: number;
  z1: number;
}

export interface PlatformArrangement {
  boxes: PlatformBox[];
  shifts: ([number, number] | undefined)[];
  rows: { node: number; box: PlatformBox }[];
}

export function slotNode(topology: MeshTopology, slot: PlatformSlot): number {
  return slot.kind === 'rank' ? slot.device.node : rankDevice(topology, slot.first).node;
}

function screenAxes(direction: Vec3): { right: Vec3; up: Vec3 } {
  const [dx, dy, dz] = direction;
  const n = Math.hypot(dx, dy, dz);
  const d: Vec3 = [dx / n, dy / n, dz / n];
  const right: Vec3 = [d[2], 0, -d[0]];
  const rn = Math.hypot(right[0], right[2]);
  const r: Vec3 = [right[0] / rn, 0, right[2] / rn];
  const up: Vec3 = [-d[0] * d[1], 1 - d[1] * d[1], -d[2] * d[1]];
  const un = Math.hypot(...up);
  return { right: r, up: [up[0] / un, up[1] / un, up[2] / un] };
}

function extentAlong(axis: Vec3, span: PlatformBox): number {
  return Math.abs(axis[0]) * (span.x1 - span.x0) + Math.abs(axis[1]) * (span.y1 - span.y0) + Math.abs(axis[2]) * (span.z1 - span.z0);
}

export function arrangePlatforms(topology: MeshTopology, slots: readonly PlatformSlot[], span: PlatformBox, direction: Vec3): PlatformArrangement {
  const { right, up } = screenAxes(direction);
  const nodes = [...new Set(slots.map((slot) => slotNode(topology, slot)))];
  const step = (extentAlong(right, span) + PLATFORM_GAP) / right[0];
  const placeholder = (PLACEHOLDER_WIDTH + PLATFORM_GAP) / right[0];
  const rise = (extentAlong(up, span) + ROW_GAP) / up[1];
  const columns = nodes.map(() => 0);
  const rows = nodes.map((node) => ({ node, box: { ...span, x0: Infinity, x1: -Infinity } }));
  const boxes: PlatformBox[] = [];
  const shifts: ([number, number] | undefined)[] = [];
  for (const slot of slots) {
    const row = nodes.indexOf(slotNode(topology, slot));
    const dy = (nodes.length - 1 - row) * rise;
    const dx = columns[row];
    columns[row] += slot.kind === 'rank' ? step : placeholder;
    const box = slot.kind === 'rank' ? { ...span, x0: span.x0 + dx, x1: span.x1 + dx, y0: span.y0 + dy, y1: span.y1 + dy } : { ...span, x0: span.x0 + dx, x1: span.x0 + dx + PLACEHOLDER_WIDTH, y0: span.y0 + dy, y1: span.y1 + dy };
    boxes.push(box);
    shifts.push(slot.kind === 'rank' ? [dx, dy] : undefined);
    const r = rows[row].box;
    rows[row].box = { ...box, x0: Math.min(r.x0, box.x0), x1: Math.max(r.x1, box.x1) };
  }
  return { boxes, shifts, rows };
}

export interface PlatformHop {
  hop: number;
  messages: number;
  bytes: number;
}

export interface PlatformEdge {
  from: number;
  to: number;
  link: 'nvlink' | 'rdma';
  bytes: number;
  messages: number;
  hops: PlatformHop[];
  phases: CommPhase[];
  pairs: [number, number][];
}

export function platformEdges(transfers: readonly CommTransfer[], slots: readonly PlatformSlot[]): PlatformEdge[] {
  const edges = new Map<string, PlatformEdge>();
  for (const t of transfers) {
    if (t.link === 'local') continue;
    const from = slotOf(slots, t.src);
    const to = slotOf(slots, t.dst);
    if (from === to) continue;
    const key = `${from}>${to}>${t.link}`;
    let edge = edges.get(key);
    if (edge === undefined) {
      edge = { from, to, link: t.link, bytes: 0, messages: 0, hops: [], phases: [], pairs: [] };
      edges.set(key, edge);
    }
    edge.bytes += t.bytes;
    edge.messages += 1;
    let hop = edge.hops.find((h) => h.hop === t.hop);
    if (hop === undefined) {
      hop = { hop: t.hop, messages: 0, bytes: 0 };
      edge.hops.push(hop);
      edge.hops.sort((a, b) => a.hop - b.hop);
    }
    hop.messages += 1;
    hop.bytes += t.bytes;
    if (t.phase !== undefined && !edge.phases.includes(t.phase)) edge.phases.push(t.phase);
    if (!edge.pairs.some(([a, b]) => a === t.src && b === t.dst)) edge.pairs.push([t.src, t.dst]);
  }
  return [...edges.values()];
}

export interface CommCall {
  op: string;
  node: Node;
  carried: Node[];
}

export function commCalls(nodes: readonly Node[]): CommCall[] {
  const calls: CommCall[] = [];
  for (const node of nodes) {
    const op = node.op;
    if (op === undefined || !COMM_OPS.has(op)) continue;
    const same = op === 'all_reduce' ? undefined : calls.find((call) => call.op === op && call.node.inputs[1] === node.inputs[1]);
    if (same === undefined) calls.push({ op, node, carried: [] });
    else same.carried.push(node);
  }
  return calls;
}

export interface LocalPiece {
  values: ArrayLike<number>;
  held: boolean[] | undefined;
  partial: boolean;
}

export function localPiece(placements: Placements, node: Node, rank: number, dir: 'forward' | 'backward', reduceShown: boolean): LocalPiece {
  if (dir === 'forward') {
    const p = placements.of(node);
    if (p.kind === 'partial') return { values: placements.ranks(node)![rank].data, held: undefined, partial: true };
    if (p.kind === 'replicate') return { values: node.value.data, held: undefined, partial: false };
    return { values: node.value.data, held: heldBy(placements, node, rank, 'forward'), partial: false };
  }
  const grad = node.grad;
  if (grad === null) return { values: new Float64Array(node.value.size), held: undefined, partial: false };
  const g = placements.grad(node);
  if (g.reduce && reduceShown) return { values: placements.gradRanks(node)![rank].data, held: undefined, partial: true };
  if (g.placement.kind === 'replicate' || g.placement.kind === 'partial') return { values: grad.data, held: undefined, partial: false };
  return { values: grad.data, held: heldBy(placements, node, rank, 'backward'), partial: false };
}

function heldBy(placements: Placements, node: Node, rank: number, dir: 'forward' | 'backward'): boolean[] {
  const held: boolean[] = [];
  forEachIndex(node.shape, (index, offset) => {
    held[offset] = placements.owner(node, index, dir) === rank;
  });
  return held;
}

export function blockValues(c: SceneController, block: Block): LocalPiece {
  const placements = c.options.parallel?.placements;
  if (block.platform === undefined || placements === undefined) return { values: valuesOf(c, block.node), held: undefined, partial: false };
  return localPiece(placements, block.node, block.platform, c.state.dir, c.stepNodes.has(block.node));
}

export function shiftBlock(block: Block, [dx, dy]: [number, number], rank: number, key: string | undefined): Block {
  const move = <T extends { x: number; y: number }>(p: T): T => ({ ...p, x: p.x + dx, y: p.y + dy });
  return {
    ...block,
    elements: block.elements.map(move),
    plate: block.plate === undefined ? undefined : move(block.plate),
    axis: block.axis === undefined ? undefined : { ...block.axis, x: block.axis.x + dx },
    title: move(block.title),
    classLabels: block.classLabels.map(move),
    rowLabels: block.rowLabels.map(move),
    frameRows: block.frameRows.map((row) => ({ ...row, y: row.y + dy })),
    extent: block.extent.map(([px, py, pz]): Vec3 => [px + dx, py + dy, pz]),
    platform: rank,
    ...(key === undefined ? {} : { key }),
  };
}

export function blockSpan(blocks: readonly Block[]): PlatformBox {
  const box = { x0: Infinity, x1: -Infinity, y0: Infinity, y1: -Infinity, z0: Infinity, z1: -Infinity };
  const grow = (x: number, y: number, z: number) => {
    box.x0 = Math.min(box.x0, x);
    box.x1 = Math.max(box.x1, x);
    box.y0 = Math.min(box.y0, y);
    box.y1 = Math.max(box.y1, y);
    box.z0 = Math.min(box.z0, z);
    box.z1 = Math.max(box.z1, z);
  };
  for (const block of blocks) {
    const [w, d] = block.footprint;
    for (const e of block.elements) {
      grow(e.x - w / 2, e.y, e.z - d / 2);
      grow(e.x + w / 2, e.y + HEIGHT_CAP, e.z + d / 2);
    }
    const plate = block.plate;
    if (plate !== undefined) {
      grow(plate.x - plate.w / 2, plate.y, plate.z - plate.d / 2);
      grow(plate.x + plate.w / 2, plate.y, plate.z + plate.d / 2);
    }
  }
  return Number.isFinite(box.x0) ? box : { x0: 0, x1: 0, y0: 0, y1: 0, z0: 0, z1: 0 };
}

interface DrawnLink {
  edge: PlatformEdge;
  call: CommCall;
  curve: CubicBezierCurve3;
  points: Vec3[];
  screen: number[];
  tube: Mesh;
  arrow: Mesh;
}

function hopText(op: string, hop: number): string {
  if (op === 'dispatch') return hop === 0 ? '第 1 跳：经 RDMA 发到目标节点上的同号 GPU' : '第 2 跳：节点内经 NVLink 转发给持有专家的卡';
  if (op === 'combine') return hop === 0 ? '第 1 跳：节点内经 NVLink 写到与 token 所在卡同号的 GPU，在那里按 token 部分求和' : '第 2 跳：部分和写回 token 所在的卡';
  return '环形 all_reduce 的一段';
}

function hiddenLabel(topology: MeshTopology, slot: Extract<PlatformSlot, { kind: 'hidden' }>): DocumentFragment {
  const fragment = document.createDocumentFragment();
  const nodes = [rankDevice(topology, slot.first).node, rankDevice(topology, slot.last).node];
  fragment.append(slotText(slot));
  const small = document.createElement('small');
  small.textContent = `${slot.last - slot.first + 1} 张卡未画出，节点 ${nodes[0] === nodes[1] ? nodes[0] : nodes.join('–')}`;
  fragment.append(document.createElement('br'), small);
  return fragment;
}

const PHASE_TEXT: Record<CommPhase, string> = { reduce_scatter: 'reduce-scatter', all_gather: 'all-gather' };

export class PlatformLayer {
  readonly axis: string;
  readonly slots: PlatformSlot[];
  arrangement: PlatformArrangement | undefined;
  private readonly group = new Group();
  private readonly frames: LineBatch;
  private readonly nodeMaterial: MeshBasicMaterial;
  private readonly nodePanels: Mesh[] = [];
  private readonly linkMaterials: Record<'nvlink' | 'rdma', MeshBasicMaterial>;
  private readonly labels: Label[];
  private readonly nodeLabels = new Map<number, Label>();
  private links: DrawnLink[] = [];
  private screenCached = -1;
  hover: DrawnLink | undefined;

  constructor(private readonly c: SceneController) {
    const parallel = c.options.parallel;
    const axis = c.options.platforms!;
    if (parallel === undefined) throw new Error(`platforms: '${axis}' needs a mesh on the page`);
    const mesh = parallel.placements.mesh;
    if (axis !== mesh.axis) throw new Error(`platforms: '${axis}' is not an axis of the mesh; the mesh has '${mesh.axis}'`);
    this.axis = axis;
    this.slots = platformSlots(mesh);
    this.frames = new LineBatch(c.theme.ink, c.theme.surface);
    this.nodeMaterial = new MeshBasicMaterial({ transparent: true, opacity: NODE_FILL_OPACITY, depthWrite: false, side: DoubleSide });
    this.linkMaterials = { nvlink: new MeshBasicMaterial(), rdma: new MeshBasicMaterial() };
    this.group.add(this.frames.object);
    c.scene.add(this.group);
    this.labels = this.slots.map((slot) => {
      const label = c.labels.add('boxlb', slot.kind === 'rank' ? 'vf-plat' : 'vf-plat vf-plat-hidden');
      label.anchor = 'bottom-center';
      if (slot.kind === 'rank') label.setText(`${slotText(slot)} · 节点 ${slot.device.node} GPU ${slot.device.gpu}`);
      else label.setContent(slotText(slot), () => hiddenLabel(mesh, slot));
      return label;
    });
    this.setTheme(c.theme);
  }

  get topology(): MeshTopology {
    return this.c.options.parallel!.placements.mesh;
  }

  involved(): Set<Node> | undefined {
    const c = this.c;
    const info = c.currentInfo();
    if (info === undefined || info.kind === 'overview' || info.kind === 'inputs') return undefined;
    const set = new Set<Node>(info.nodes);
    for (const node of info.nodes) {
      const inputs = node.op !== undefined && COMM_OPS.has(node.op) ? node.inputs.slice(0, 1) : node.inputs;
      for (const input of inputs) {
        set.add(input);
        if (input.op === 'transpose') set.add(input.inputs[0]);
      }
    }
    const module = c.steps?.modules.find((m) => info.id === `F:${m.id}`);
    if (module !== undefined) for (const node of [...module.inputs, ...module.params]) set.add(node);
    return set;
  }

  foldStep(): string | undefined {
    const info = this.c.currentInfo();
    return info?.kind === 'module' ? info.id.replace(/^F:/, '') : undefined;
  }

  expand(blocks: readonly Block[]): Block[] {
    const involved = this.involved();
    const drawn = (block: Block) => block.hidden !== true && (involved === undefined || involved.has(block.node));
    const { x, y, z } = this.c.rig.direction;
    const arrangement = arrangePlatforms(this.topology, this.slots, blockSpan(blocks.filter(drawn)), [x, y, z]);
    this.arrangement = arrangement;
    const out: Block[] = [];
    this.slots.forEach((slot, i) => {
      const shift = arrangement.shifts[i];
      if (slot.kind !== 'rank' || shift === undefined) return;
      for (const block of blocks) {
        const key = i === 0 ? block.key : `p${slot.rank}/${block.key ?? `n${block.node.id}`}`;
        const copy = shiftBlock(block, shift, slot.rank, key);
        if (!drawn(block)) copy.hidden = true;
        out.push(copy);
      }
    });
    return out;
  }

  shifts(): [number, number][] {
    return (this.arrangement?.shifts ?? [[0, 0]]).filter((x): x is [number, number] => x !== undefined);
  }

  framePoints(): Vec3[] {
    const arrangement = this.arrangement;
    if (arrangement === undefined) return [];
    return arrangement.rows.flatMap(({ box: b }): Vec3[] => [
      [b.x0 - NODE_PAD, b.y0 - FRAME_PAD, b.z0 - NODE_PAD],
      [b.x1 + NODE_PAD, b.y1, b.z1 + NODE_PAD + LABEL_ROOM],
      [b.x1 + NODE_PAD, b.y0 - FRAME_PAD, b.z1 + NODE_PAD + LABEL_ROOM],
    ]);
  }

  private firstRank(): number {
    const first = this.slots[0];
    return first.kind === 'rank' ? first.rank : 0;
  }

  update(): void {
    const arrangement = this.arrangement;
    if (arrangement === undefined) return;
    const lines: { points: number[]; opacity: number }[] = [];
    const floor = (b: PlatformBox) => b.y0 - FRAME_PAD;
    const tray = (b: PlatformBox, pad: number, y: number) => [b.x0 - pad, y, b.z0 - pad, b.x1 + pad, y, b.z0 - pad, b.x1 + pad, y, b.z1 + pad, b.x0 - pad, y, b.z1 + pad, b.x0 - pad, y, b.z0 - pad];
    this.slots.forEach((slot, i) => {
      const box = arrangement.boxes[i];
      const outline = tray(box, FRAME_PAD, floor(box));
      if (slot.kind === 'rank') lines.push({ points: outline, opacity: FRAME_OPACITY });
      else for (const dash of dashed(outline)) lines.push({ points: dash, opacity: FRAME_OPACITY });
    });
    this.frames.setPolylines(lines);
    while (this.nodePanels.length < arrangement.rows.length) {
      const geometry = new PlaneGeometry(1, 1);
      geometry.rotateX(-Math.PI / 2);
      const panel = new Mesh(geometry, this.nodeMaterial);
      this.nodePanels.push(panel);
      this.group.add(panel);
    }
    const named = !this.c.lod;
    arrangement.rows.forEach((row, i) => {
      const { x0, x1, z0, z1 } = row.box;
      const y = floor(row.box) - NODE_DROP;
      const panel = this.nodePanels[i];
      panel.scale.set(x1 - x0 + 2 * NODE_PAD, 1, z1 - z0 + 2 * NODE_PAD);
      panel.position.set((x0 + x1) / 2, y, (z0 + z1) / 2);
      let label = this.nodeLabels.get(row.node);
      if (label === undefined) {
        label = this.c.labels.add('boxlb', 'vf-plat vf-plat-node');
        label.anchor = 'right-middle';
        label.setText(`节点 ${row.node}`);
        this.nodeLabels.set(row.node, label);
      }
      label.pos = [x0 - NODE_PAD, y, z1 + NODE_PAD];
      label.visible = named;
    });
    this.labels.forEach((label, i) => {
      const b = arrangement.boxes[i];
      const rank = this.slots[i].kind === 'rank';
      label.anchor = 'top-center';
      label.pos = rank ? [b.x1, floor(b), b.z1 + FRAME_PAD] : [(b.x0 + b.x1) / 2, floor(b), (b.z0 + b.z1) / 2];
      label.visible = named;
    });
    this.updateLinks();
  }

  private updateLinks(): void {
    for (const link of this.links) {
      this.group.remove(link.tube, link.arrow);
      link.tube.geometry.dispose();
      link.arrow.geometry.dispose();
    }
    this.links = [];
    const c = this.c;
    const placements = c.options.parallel!.placements;
    if (c.state.dir !== 'forward' || this.arrangement === undefined) return;
    const info = c.currentInfo();
    if (info === undefined || info.kind === 'overview') return;
    const tokens = MODE_TOKENS[parallelMode(c)];
    for (const call of commCalls(info.nodes)) {
      const transfers = commTransfers(placements, call.node, { tokens });
      for (const edge of platformEdges(transfers, this.slots)) {
        const a = this.anchor(call.node, edge.from, edge.to, edge.link);
        const b = this.anchor(call.node, edge.to, edge.from, edge.link);
        if (a === undefined || b === undefined) continue;
        this.links.push(this.drawLink(edge, call, a, b));
      }
    }
    this.screenCached = -1;
    if (this.hover !== undefined) this.hover = this.links.find((l) => l.edge.from === this.hover!.edge.from && l.edge.to === this.hover!.edge.to && l.call.node === this.hover!.call.node);
    this.restyle();
  }

  private anchor(node: Node, slot: number, other: number, link: 'nvlink' | 'rdma'): Vec3 | undefined {
    const { boxes, shifts } = this.arrangement!;
    const base = this.c.renderables.find((r) => r.block.node === node && r.block.platform === this.firstRank());
    if (base === undefined || base.block.hidden === true) return undefined;
    const elements = base.block.elements;
    const box = boxes[slot];
    const sameRow = box.y0 === boxes[other].y0;
    const side = slot < other ? 1 : -1;
    const cx = elements.reduce((s, e) => s + e.x, 0) / elements.length - (shifts[0]?.[0] ?? 0) + (sameRow ? 0 : side * LINK_SPREAD);
    const cz = elements.reduce((s, e) => s + e.z, 0) / elements.length + (sameRow ? side * LINK_SPREAD : 0) + (link === 'rdma' ? LINK_SPREAD : -LINK_SPREAD) / 2;
    const y = elements[0].y - boxes[0].y0 + box.y0 + HEIGHT_CAP;
    const shift = shifts[slot];
    return [shift === undefined ? (box.x0 + box.x1) / 2 + (sameRow ? 0 : side * LINK_SPREAD) : cx + shift[0], y, cz];
  }

  private drawLink(edge: PlatformEdge, call: CommCall, a: Vec3, b: Vec3): DrawnLink {
    const sameRow = Math.abs(a[1] - b[1]) < ROW_GAP;
    const lift = sameRow ? (b[0] > a[0] ? 2 : 1) * LINK_LIFT + LINK_RISE * Math.abs(b[0] - a[0]) : LINK_LIFT;
    const bulge = sameRow ? 0 : LINK_BULGE;
    const p0 = new Vector3(...a);
    const p3 = new Vector3(...b);
    const curve = new CubicBezierCurve3(p0, new Vector3(a[0], a[1] + lift, a[2] + bulge), new Vector3(b[0], b[1] + lift, b[2] + bulge), p3);
    const radius = this.radius();
    const material = this.linkMaterials[edge.link];
    const tube = new Mesh(new TubeGeometry(curve, LINK_SEGMENTS, radius, LINK_SIDES, false), material);
    const arrow = new Mesh(new ConeGeometry(radius * ARROW_RADIUS, ARROW_LENGTH / Math.sqrt(this.c.camera.zoom), LINK_SIDES * 2), material);
    const tangent = curve.getTangent(1);
    arrow.position.copy(p3);
    arrow.quaternion.setFromUnitVectors(new Vector3(0, 1, 0), tangent);
    this.group.add(tube, arrow);
    const points = curve.getPoints(LINK_SEGMENTS).map((p): Vec3 => [p.x, p.y, p.z]);
    return { edge, call, curve, points, screen: [], tube, arrow };
  }

  private radius(): number {
    return LINK_RADIUS / Math.sqrt(this.c.camera.zoom);
  }

  private restyle(): void {
    for (const link of this.links) {
      const scale = link === this.hover ? HOVER_WIDEN : 1;
      link.tube.geometry.dispose();
      link.tube.geometry = new TubeGeometry(link.curve, LINK_SEGMENTS, this.radius() * scale, LINK_SIDES, false);
      link.arrow.scale.setScalar(scale);
    }
  }

  linkAt(x: number, y: number): DrawnLink | undefined {
    const c = this.c;
    if (this.screenCached !== c.screenVersion) {
      const v = new Vector3();
      for (const link of this.links) {
        link.screen = [];
        for (const p of link.points) {
          v.set(...p).project(c.camera);
          link.screen.push(((v.x + 1) / 2) * c.width, ((1 - v.y) / 2) * c.height);
        }
      }
      this.screenCached = c.screenVersion;
    }
    let best: DrawnLink | undefined;
    let bestDistance = LINK_HOVER_DISTANCE;
    for (const link of this.links) {
      for (let i = 0; i + 3 < link.screen.length; i += 2) {
        const d = segmentDistance(x, y, link.screen[i], link.screen[i + 1], link.screen[i + 2], link.screen[i + 3]);
        if (d < bestDistance) {
          bestDistance = d;
          best = link;
        }
      }
    }
    return best;
  }

  setHover(link: DrawnLink | undefined): void {
    if (this.hover === link) return;
    this.hover = link;
    this.restyle();
    this.c.requestFrame();
  }

  card(link: DrawnLink): DocumentFragment {
    const { edge, call } = link;
    const c = this.c;
    const fragment = document.createDocumentFragment();
    const title = document.createElement('h4');
    title.className = 'vf-pop-title';
    renderText(`${call.op}：${slotText(this.slots[edge.from])} → ${slotText(this.slots[edge.to])}`, title);
    fragment.appendChild(title);
    const dl = document.createElement('dl');
    dl.className = 'vf-rows';
    const row = (name: string, text: string) => {
      const dt = document.createElement('dt');
      dt.textContent = name;
      const dd = document.createElement('dd');
      renderText(text, dd);
      dl.append(dt, dd);
    };
    row('链路', edge.link === 'nvlink' ? 'NVLink，同一节点内' : 'RDMA，节点之间');
    if (call.op === 'all_reduce') row('阶段', edge.phases.map((p) => PHASE_TEXT[p]).join('、'));
    else for (const hop of edge.hops) row(`第 ${hop.hop + 1} 跳`, `${hopText(call.op, hop.hop)}；${hop.messages} 条消息，$${bytesText(hop.bytes)}$`);
    row('合计', `${edge.messages} 条消息，$${bytesText(edge.bytes)}$`);
    const hidden = [this.slots[edge.from], this.slots[edge.to]].some((s) => s.kind === 'hidden');
    if (hidden) row('卡', edge.pairs.map(([s, d]) => `${s}→${d}`).join('、'));
    if (call.op === 'all_reduce') row('规模', `$T=${MODE_TOKENS[parallelMode(c)]}$ 个 token，隐藏维取真实模型的大小`);
    else row('规模', '这一页的 token 和路由；每条消息按真实模型的隐藏维计算，含 top-$k$ 专家编号与权重');
    if (call.carried.length > 0) row('同一条消息', call.carried.map((n) => `$${n.tex}$`).join('、'));
    fragment.appendChild(dl);
    return fragment;
  }

  setTheme(theme: ThemeColors): void {
    this.frames.setColor(theme.ink, theme.surface);
    this.nodeMaterial.color.setRGB(...theme.line);
    const set = (material: MeshBasicMaterial, rgb: LinearRgb) => material.color.setRGB(...rgb);
    set(this.linkMaterials.nvlink, theme.nvlink);
    set(this.linkMaterials.rdma, theme.rdma);
  }

  dispose(): void {
    for (const link of this.links) {
      link.tube.geometry.dispose();
      link.arrow.geometry.dispose();
    }
    for (const panel of this.nodePanels) panel.geometry.dispose();
    this.frames.dispose();
    this.nodeMaterial.dispose();
    this.linkMaterials.nvlink.dispose();
    this.linkMaterials.rdma.dispose();
    this.c.scene.remove(this.group);
  }
}
