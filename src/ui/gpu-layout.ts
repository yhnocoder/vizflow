export type GpuLevel = 'node' | 'gpu' | 'sm';
export type GpuKind = 'compute' | 'nvlink' | 'rdma' | 'reduce' | 'copy';
export type GpuColor = GpuKind | number;
export type GpuBadgeLine = string | { text: string; chips: number[] };

export interface GpuTopology {
  nodes: number;
  gpusPerNode: number;
  smsPerGpu: number;
  smLabel?: string;
  gpuLabels?: (rank: number) => string;
  nic: boolean;
}

export interface GpuSmOptions {
  arch?: 'sm90' | 'sm100';
  numStages?: number;
  mathWarpgroups?: number;
  note?: string;
}

export interface GpuFocus {
  gpu?: number;
  sm?: number;
}

export interface GpuFlow {
  from: string;
  to: string;
  kind: GpuKind;
  label?: string;
  via?: string[];
}

export interface GpuPacket {
  flow: number | GpuFlow;
  color?: GpuColor;
  delay?: number;
}

export interface GpuTileOwner {
  sm: number;
  wave: number;
}

export interface GpuTileGroup {
  rows: [number, number];
  cols: [number, number];
  label?: string;
}

export interface GpuTiles {
  rows: number;
  cols: number;
  owner: (row: number, col: number) => GpuTileOwner | null;
  wave?: number;
  title?: string;
  group?: GpuTileGroup;
}

export interface GpuDiagramProps {
  topology: GpuTopology;
  level: GpuLevel;
  focus?: GpuFocus;
  sm?: GpuSmOptions;
  highlight?: Record<string, GpuKind>;
  flows?: GpuFlow[];
  packets?: GpuPacket[];
  badges?: Record<string, GpuBadgeLine[]>;
  tiles?: GpuTiles;
}

export interface Point {
  x: number;
  y: number;
}

export interface GpuBox {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  title: string;
  note: string;
  lines: GpuBadgeLine[];
  text: 'full' | 'vertical' | 'none';
}

export interface GpuText {
  x: number;
  y: number;
  text: string;
  anchor: 'start' | 'middle' | 'end';
}

export interface GpuTileCell {
  x: number;
  y: number;
  w: number;
  h: number;
  state: 'empty' | 'done' | 'current';
  lines: string[];
}

export interface GpuTileLayout {
  cells: GpuTileCell[];
  group?: { x: number; y: number; w: number; h: number; label?: string; labelY: number };
}

export interface GpuLayout {
  width: number;
  height: number;
  frames: GpuBox[];
  boxes: GpuBox[];
  wires: Point[][];
  texts: GpuText[];
  tiles?: GpuTileLayout;
  route(flow: GpuFlow): Point[];
  units(flow: GpuFlow): string[];
}

export const GPU_WIDE_WIDTH = 760;
export const GPU_NARROW_WIDTH = 340;
export const GPU_DEFAULT_STAGES = 4;
export const GPU_DEFAULT_WARPGROUPS = 2;

const marginOf = (width: number) => (width >= GPU_WIDE_WIDTH ? 24 : 16);
export const GPU_LINE_HEIGHT = 14;
export const GPU_TITLE_HEIGHT = 24;
const GAP = 10;
const NIC_W = 14;
const NIC_H = 32;
const SWITCH_W = 22;
const NODE_HEAD = 30;
const NODE_FOOT = 22;
const IB_W = 120;
const IB_H = 26;
const FRAME_HEAD = 48;
const SM_MIN_W = 64;
const SM_DENSE_CELL = 22;
const TILE_GAP = 4;

const range = (n: number) => Array.from({ length: n }, (_, i) => i);
const cx = (b: GpuBox) => b.x + b.w / 2;
const cy = (b: GpuBox) => b.y + b.h / 2;

function box(id: string, x: number, y: number, w: number, h: number, title: string, lines: GpuBadgeLine[] = [], note = '', text: GpuBox['text'] = 'full'): GpuBox {
  return { id, x, y, w, h, title, note, lines, text };
}

function contentHeight(lines: number, min: number): number {
  return Math.max(min, GPU_TITLE_HEIGHT + lines * GPU_LINE_HEIGHT + 6);
}

export function connector(a: GpuBox, b: GpuBox): Point[] {
  const separatedX = a.x + a.w <= b.x || b.x + b.w <= a.x;
  if (separatedX) {
    const forward = cx(a) < cx(b);
    const start = { x: forward ? a.x + a.w : a.x, y: cy(a) };
    const end = { x: forward ? b.x : b.x + b.w, y: cy(b) };
    if (start.y === end.y) return [start, end];
    const mx = (start.x + end.x) / 2;
    return [start, { x: mx, y: start.y }, { x: mx, y: end.y }, end];
  }
  const down = cy(a) < cy(b);
  const start = { x: cx(a), y: down ? a.y + a.h : a.y };
  const end = { x: cx(b), y: down ? b.y : b.y + b.h };
  if (start.x === end.x) return [start, end];
  const my = (start.y + end.y) / 2;
  return [start, { x: start.x, y: my }, { x: end.x, y: my }, end];
}

function joinPaths(parts: Point[][]): Point[] {
  const out: Point[] = [];
  for (const part of parts) {
    const last = out[out.length - 1];
    const first = part[0];
    if (last !== undefined && first !== undefined && last.x !== first.x && last.y !== first.y) out.push({ x: first.x, y: last.y });
    for (const p of part) {
      const last = out[out.length - 1];
      if (last === undefined || last.x !== p.x || last.y !== p.y) out.push(p);
    }
  }
  return out;
}

export function pathLength(points: Point[]): number {
  let total = 0;
  for (let i = 1; i < points.length; i++) total += Math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y);
  return total;
}

export function pointAt(points: Point[], fraction: number): Point {
  const target = Math.max(0, Math.min(1, fraction)) * pathLength(points);
  let walked = 0;
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1];
    const b = points[i];
    const len = Math.hypot(b.x - a.x, b.y - a.y);
    if (walked + len >= target && len > 0) {
      const t = (target - walked) / len;
      return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
    }
    walked += len;
  }
  return points[points.length - 1];
}

class Wiring {
  private readonly edges = new Map<string, Point[]>();
  private readonly neighbors = new Map<string, string[]>();

  constructor(private readonly boxes: Map<string, GpuBox>, private readonly level: GpuLevel) {}

  add(a: string, b: string, points: Point[]): void {
    this.edges.set(`${a}|${b}`, points);
    this.edges.set(`${b}|${a}`, [...points].reverse());
    for (const [from, to] of [[a, b], [b, a]]) {
      const list = this.neighbors.get(from) ?? [];
      list.push(to);
      this.neighbors.set(from, list);
    }
  }

  get wires(): Point[][] {
    return [...this.edges.entries()].filter(([key]) => key.split('|')[0] < key.split('|')[1]).map(([, points]) => points);
  }

  unit(id: string): GpuBox {
    const found = this.boxes.get(id);
    if (found === undefined) throw new Error(`gpu diagram: unknown unit '${id}' at level ${this.level}`);
    return found;
  }

  hops(from: string, to: string): string[] {
    this.unit(from);
    this.unit(to);
    if (from === to) return [from];
    const previous = new Map<string, string>([[from, from]]);
    const queue = [from];
    while (queue.length > 0) {
      const current = queue.shift()!;
      for (const next of this.neighbors.get(current) ?? []) {
        if (previous.has(next)) continue;
        previous.set(next, current);
        if (next === to) {
          const path = [to];
          while (path[0] !== from) path.unshift(previous.get(path[0])!);
          return path;
        }
        queue.push(next);
      }
    }
    return [from, to];
  }

  units(flow: GpuFlow): string[] {
    const stops = [flow.from, ...(flow.via ?? []), flow.to];
    const units: string[] = [stops[0]];
    for (let i = 1; i < stops.length; i++) units.push(...this.hops(stops[i - 1], stops[i]).slice(1));
    return units;
  }

  route(flow: GpuFlow): Point[] {
    const units = this.units(flow);
    const parts = range(units.length - 1).map((i) => this.edges.get(`${units[i]}|${units[i + 1]}`) ?? connector(this.unit(units[i]), this.unit(units[i + 1])));
    return joinPaths(parts);
  }
}

function finish(width: number, height: number, frames: GpuBox[], boxes: GpuBox[], texts: GpuText[], wiring: Wiring, tiles?: GpuTileLayout): GpuLayout {
  return { width, height, frames, boxes, texts, tiles, wires: wiring.wires, route: (flow) => wiring.route(flow), units: (flow) => wiring.units(flow) };
}

function linesOf(badges: Record<string, GpuBadgeLine[]>, id: string): GpuBadgeLine[] {
  return badges[id] ?? [];
}

function layoutNodes(props: GpuDiagramProps, width: number): GpuLayout {
  const MARGIN = marginOf(width);
  const t = props.topology;
  const badges = props.badges ?? {};
  const columns = width >= GPU_WIDE_WIDTH ? Math.min(t.nodes, 2) : 1;
  const nodeGap = 40;
  const nodeW = (width - 2 * MARGIN - (columns - 1) * nodeGap) / columns;
  const gpuRows = Math.ceil(t.gpusPerNode / 2);
  const ranks = range(t.nodes * t.gpusPerNode);
  const maxLines = Math.max(0, ...ranks.map((r) => linesOf(badges, `gpu:${r}`).length));
  const gpuH = contentHeight(maxLines, 44);
  const nodeH = NODE_HEAD + gpuRows * (gpuH + GAP) - GAP + NODE_FOOT;
  const hasIb = t.nodes > 1;
  const nicW = t.nic ? NIC_W : 0;
  const frames: GpuBox[] = [];
  const boxes: GpuBox[] = [];
  const texts: GpuText[] = [];
  const pending: [string, string, Point[]][] = [];
  const lanes: { x: number; nic: string; from: Point }[] = [];
  for (const n of range(t.nodes)) {
    const x = MARGIN + (n % columns) * (nodeW + nodeGap);
    const y = MARGIN + Math.floor(n / columns) * (nodeH + 24);
    frames.push(box(`node:${n}`, x, y, nodeW, nodeH, `节点 ${n}`));
    const swx = x + nodeW / 2;
    const gpuTop = y + NODE_HEAD;
    const switchBox = box(`nvswitch:${n}`, swx - SWITCH_W / 2, gpuTop, SWITCH_W, gpuRows * (gpuH + GAP) - GAP, 'NVSwitch', [], '', 'none');
    boxes.push(switchBox);
    texts.push({ x: swx, y: y + nodeH - 8, text: 'NVSwitch', anchor: 'middle' });
    const inner = 8;
    const gpuW = nodeW / 2 - SWITCH_W / 2 - 14 - inner - (t.nic ? nicW + 6 : 0);
    for (const local of range(t.gpusPerNode)) {
      const rank = n * t.gpusPerNode + local;
      const right = local % 2 === 1;
      const gy = gpuTop + Math.floor(local / 2) * (gpuH + GAP);
      const gx = right ? swx + SWITCH_W / 2 + 14 : swx - SWITCH_W / 2 - 14 - gpuW;
      const gpu = box(`gpu:${rank}`, gx, gy, gpuW, gpuH, `GPU ${rank}`, linesOf(badges, `gpu:${rank}`), t.gpuLabels?.(rank) ?? '');
      boxes.push(gpu);
      const edgeX = right ? gx : gx + gpuW;
      pending.push([gpu.id, switchBox.id, [{ x: edgeX, y: cy(gpu) }, { x: swx, y: cy(gpu) }]]);
      if (t.nic) {
        const nx = right ? x + nodeW - inner - NIC_W : x + inner;
        const nic = box(`nic:${rank}`, nx, cy(gpu) - NIC_H / 2, NIC_W, NIC_H, 'NIC', [], '', 'vertical');
        boxes.push(nic);
        pending.push([gpu.id, nic.id, [{ x: right ? gx + gpuW : gx, y: cy(gpu) }, { x: right ? nx : nx + NIC_W, y: cy(gpu) }]]);
        if (hasIb) lanes.push({ x: right ? x + nodeW + 12 : x - 12, nic: nic.id, from: { x: right ? nx + NIC_W : nx, y: cy(gpu) } });
      } else if (hasIb) {
        lanes.push({ x: right ? x + nodeW + 12 : x - 12, nic: gpu.id, from: { x: right ? gx + gpuW : gx, y: cy(gpu) } });
      }
    }
  }
  const nodeRows = Math.ceil(t.nodes / columns);
  let height = MARGIN + nodeRows * (nodeH + 24) - 24 + MARGIN;
  if (hasIb) {
    const ibY = height - MARGIN + 28;
    const ib = box('ib', width / 2 - IB_W / 2, ibY, IB_W, IB_H, 'IB 交换机');
    boxes.push(ib);
    const busY = ib.y + ib.h + 12;
    for (const lane of lanes) {
      const portX = lane.x < cx(ib) ? ib.x + 16 : ib.x + ib.w - 16;
      pending.push([lane.nic, 'ib', [lane.from, { x: lane.x, y: lane.from.y }, { x: lane.x, y: busY }, { x: portX, y: busY }, { x: portX, y: ib.y + ib.h }]]);
    }
    height = busY + MARGIN;
  }
  const wiring = new Wiring(new Map([...frames, ...boxes].map((b) => [b.id, b])), 'node');
  for (const [a, b, points] of pending) wiring.add(a, b, points);
  return finish(width, height, frames, boxes, texts, wiring);
}

function layoutTiles(tiles: GpuTiles, x: number, y: number, w: number): { layout: GpuTileLayout; height: number } {
  const tileW = Math.min(64, (w - (tiles.cols - 1) * TILE_GAP) / tiles.cols);
  const roomy = tileW >= 52;
  const tileH = roomy ? 38 : 26;
  const wave = tiles.wave;
  const cells: GpuTileCell[] = [];
  for (const r of range(tiles.rows)) {
    for (const c of range(tiles.cols)) {
      const owner = tiles.owner(r, c);
      const shown = owner !== null && wave !== undefined && owner.wave <= wave;
      const state = !shown ? 'empty' : owner.wave === wave ? 'current' : 'done';
      const lines = !shown ? [] : roomy ? [`SM ${owner.sm}`, `第 ${owner.wave + 1} 轮`] : [`${owner.sm}`];
      cells.push({ x: x + c * (tileW + TILE_GAP), y: y + r * (tileH + TILE_GAP), w: tileW, h: tileH, state, lines });
    }
  }
  const gridH = tiles.rows * (tileH + TILE_GAP) - TILE_GAP;
  let height = gridH;
  let group: GpuTileLayout['group'];
  if (tiles.group !== undefined) {
    const [r0, r1] = tiles.group.rows;
    const [c0, c1] = tiles.group.cols;
    const gx = x + c0 * (tileW + TILE_GAP) - 3;
    const gy = y + r0 * (tileH + TILE_GAP) - 3;
    const gw = (c1 - c0 + 1) * (tileW + TILE_GAP) - TILE_GAP + 6;
    const gh = (r1 - r0 + 1) * (tileH + TILE_GAP) - TILE_GAP + 6;
    group = { x: gx, y: gy, w: gw, h: gh, label: tiles.group.label, labelY: y + gridH + 18 };
    if (tiles.group.label !== undefined) height += 24;
  }
  return { layout: { cells, group }, height };
}

function layoutGpu(props: GpuDiagramProps, width: number): GpuLayout {
  const MARGIN = marginOf(width);
  const t = props.topology;
  const badges = props.badges ?? {};
  const rank = props.focus?.gpu ?? 0;
  const frameX = MARGIN - 4;
  const inner = frameX + 16;
  const innerW = width - 2 * inner;
  const boxes: GpuBox[] = [];
  const texts: GpuText[] = [];
  let y = MARGIN + FRAME_HEAD;
  const dense = t.smsPerGpu > 16;
  const cell = width >= GPU_WIDE_WIDTH ? SM_DENSE_CELL : 18;
  const gap = dense ? 4 : 8;
  const cols = dense ? Math.floor((innerW + gap) / (cell + gap)) : Math.min(t.smsPerGpu, Math.floor((innerW + gap) / (SM_MIN_W + gap)));
  const smW = dense ? cell : (innerW - (cols - 1) * gap) / cols;
  const maxLines = dense ? 0 : Math.max(0, ...range(t.smsPerGpu).map((i) => linesOf(badges, `sm:${i}`).length));
  const smH = dense ? cell : contentHeight(maxLines, 34);
  const smRows = Math.ceil(t.smsPerGpu / cols);
  for (const i of range(t.smsPerGpu)) {
    const sx = inner + (i % cols) * (smW + gap);
    const sy = y + Math.floor(i / cols) * (smH + gap);
    boxes.push(box(`sm:${i}`, sx, sy, smW, smH, `SM ${i}`, dense ? [] : linesOf(badges, `sm:${i}`), '', dense ? 'none' : 'full'));
  }
  y += smRows * (smH + gap) - gap + 22;
  const l2Lines = linesOf(badges, 'l2');
  const l2 = box('l2', inner, y, innerW, contentHeight(l2Lines.length, 28), 'L2', l2Lines);
  boxes.push(l2);
  y += l2.h + 16;
  const hbmLines = linesOf(badges, 'hbm');
  const hbm = box('hbm', inner, y, innerW, contentHeight(hbmLines.length, 28), 'HBM', hbmLines);
  boxes.push(hbm);
  y += hbm.h;
  let tiles: GpuTileLayout | undefined;
  if (props.tiles !== undefined) {
    y += 30;
    texts.push({ x: inner, y: y - 10, text: props.tiles.title ?? `输出：${props.tiles.rows} x ${props.tiles.cols} 个 tile`, anchor: 'start' });
    const placed = layoutTiles(props.tiles, inner, y, innerW);
    tiles = placed.layout;
    y += placed.height;
  }
  y += 16;
  const frame = box(`gpu:${rank}`, frameX, MARGIN, width - 2 * frameX, y - MARGIN, `GPU ${rank}`, [], t.smLabel ?? '');
  const map = new Map([frame, ...boxes].map((b) => [b.id, b]));
  const wiring = new Wiring(map, 'gpu');
  for (const i of range(t.smsPerGpu)) {
    const sm = map.get(`sm:${i}`)!;
    const lastRow = Math.floor(i / cols) === smRows - 1;
    const gutter = sm.x - gap / 2;
    const points = lastRow
      ? [{ x: cx(sm), y: sm.y + sm.h }, { x: cx(sm), y: l2.y }]
      : [{ x: cx(sm), y: sm.y + sm.h }, { x: cx(sm), y: sm.y + sm.h + gap / 2 }, { x: gutter, y: sm.y + sm.h + gap / 2 }, { x: gutter, y: l2.y }];
    wiring.add(sm.id, 'l2', points);
  }
  wiring.add('l2', 'hbm', [{ x: inner + innerW / 2, y: l2.y + l2.h }, { x: inner + innerW / 2, y: hbm.y }]);
  return { ...finish(width, y + MARGIN, [frame], boxes, texts, wiring, tiles), wires: [] };
}

function smUnits(props: GpuDiagramProps): { columns: string[][]; titles: Map<string, [string, string[]]> } {
  const options = props.sm ?? {};
  const stages = options.numStages ?? GPU_DEFAULT_STAGES;
  const groups = options.mathWarpgroups ?? GPU_DEFAULT_WARPGROUPS;
  const titles = new Map<string, [string, string[]]>();
  titles.set('tma', ['TMA', []]);
  titles.set('hbm', ['HBM', []]);
  for (const i of range(stages)) titles.set(`smem:stage${i}`, [`stage ${i}`, []]);
  for (const i of range(groups)) titles.set(`wg:${i}`, [`math WG ${i}`, ['Tensor Core']]);
  if (options.arch === 'sm100') titles.set('tmem', ['Tensor Memory', []]);
  titles.set('cuda-core', ['CUDA core', []]);
  titles.set('regs', ['寄存器', []]);
  titles.set('epilogue', ['epilogue', []]);
  const right = [...(options.arch === 'sm100' ? ['tmem'] : []), 'cuda-core', 'regs'];
  return {
    columns: [['tma', 'hbm'], range(stages).map((i) => `smem:stage${i}`), [...range(groups).map((i) => `wg:${i}`), 'epilogue'], right],
    titles,
  };
}

function layoutSm(props: GpuDiagramProps, width: number): GpuLayout {
  const MARGIN = marginOf(width);
  const badges = props.badges ?? {};
  const { columns, titles } = smUnits(props);
  const frameX = MARGIN - 4;
  const inner = frameX + 16;
  const innerW = width - 2 * inner;
  const top = MARGIN + FRAME_HEAD;
  const boxes: GpuBox[] = [];
  const texts: GpuText[] = [];
  const make = (id: string, x: number, y: number, w: number, h?: number) => {
    const [title, fixed] = titles.get(id)!;
    const lines = [...fixed, ...linesOf(badges, id)];
    return box(id, x, y, w, h ?? contentHeight(lines.length, 34), title, lines);
  };
  const smemHead = 20;
  let bottom = top;
  if (width >= GPU_WIDE_WIDTH) {
    const widths = [130, 170, 170, 150];
    const colGap = (innerW - widths.reduce((a, b) => a + b, 0)) / (widths.length - 1);
    let x = inner;
    columns.forEach((ids, c) => {
      let y = top + (c === 1 ? smemHead : 0);
      for (const id of ids) {
        const b = make(id, x, y, widths[c]);
        boxes.push(b);
        y += b.h + 12;
      }
      bottom = Math.max(bottom, y - 12);
      x += widths[c] + colGap;
    });
    texts.push({ x: inner + widths[0] + colGap, y: top + 12, text: `共享内存 ${columns[1].length} 级环形缓冲`, anchor: 'start' });
    const hbm = boxes.find((b) => b.id === 'hbm')!;
    hbm.y = bottom - hbm.h;
  } else {
    const pairs = (ids: string[]) => range(Math.ceil(ids.length / 2)).map((i) => ids.slice(2 * i, 2 * i + 2));
    const rows = [columns[0], ...pairs(columns[1]), ...pairs(columns[2].slice(0, -1)), ...pairs(columns[3]), ['epilogue']];
    let y = top;
    rows.forEach((ids, r) => {
      if (r === 1) {
        texts.push({ x: inner, y: y + 12, text: `共享内存 ${columns[1].length} 级环形缓冲`, anchor: 'start' });
        y += smemHead;
      }
      const w = (innerW - (ids.length - 1) * 8) / ids.length;
      const heights = ids.map((id) => make(id, 0, 0, w).h);
      const h = Math.max(...heights);
      ids.forEach((id, i) => boxes.push(make(id, inner + i * (w + 8), y, w, h)));
      y += h + 16;
    });
    bottom = y - 16;
  }
  const height = bottom + 16;
  const focus = props.focus ?? {};
  const frame = box('sm-frame', frameX, MARGIN, width - 2 * frameX, height - MARGIN, `GPU ${focus.gpu ?? 0} 的 SM ${focus.sm ?? 0}`, [], props.sm?.note ?? '');
  const map = new Map(boxes.map((b) => [b.id, b]));
  const wiring = new Wiring(map, 'sm');
  const link = (a: string, b: string) => wiring.add(a, b, connector(map.get(a)!, map.get(b)!));
  link('hbm', 'tma');
  for (const stage of columns[1]) {
    link('tma', stage);
    for (const wg of columns[2].slice(0, -1)) link(stage, wg);
  }
  for (const wg of columns[2].slice(0, -1)) link(wg, props.sm?.arch === 'sm100' ? 'tmem' : 'cuda-core');
  return finish(width, height + MARGIN, [frame], boxes, texts, wiring);
}

export function layoutGpuDiagram(props: GpuDiagramProps, width: number): GpuLayout {
  if (props.level === 'node') return layoutNodes(props, width);
  if (props.level === 'gpu') return layoutGpu(props, width);
  return layoutSm(props, width);
}

export function gpuUnitIds(props: GpuDiagramProps): string[] {
  const layout = layoutGpuDiagram(props, GPU_WIDE_WIDTH);
  return [...(props.level === 'node' ? layout.frames : []), ...layout.boxes].map((b) => b.id);
}
