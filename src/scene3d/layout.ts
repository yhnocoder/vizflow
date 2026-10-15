import type { Node } from '../autograd';
import { explainOf } from '../explain';
import { formatShape, indexOf, normDim, offsetOf } from '../tensor';
import type { LevelLayout, LevelPlan } from './levels';

export type Vec3 = [number, number, number];

export interface Scales {
  forward?: number;
  backward?: number;
}

export interface TemplateOptions {
  at?: Vec3;
  gap?: number;
  scale?: number;
  scales?: Record<string, Scales>;
}

export interface StackOptions extends TemplateOptions {
  level?: number;
  back?: number;
  offsets?: Record<string, Vec3>;
  scalars?: 'rise' | 'row';
}

export type LayoutItem =
  | { template: 'board'; a: Node; b: Node; out: Node; options: TemplateOptions }
  | { template: 'stack'; nodes: Node[]; options: StackOptions }
  | { template: 'node'; node: Node; options: { at?: Vec3 } };

const BOARD_OPTION_KEYS: ReadonlySet<string> = new Set(['at', 'gap', 'scale', 'scales']);

export function board(a: Node, b: Node, out: Node, options: TemplateOptions = {}): LayoutItem {
  const unknown = Object.keys(options).find((key) => !BOARD_OPTION_KEYS.has(key));
  if (unknown !== undefined) throw new Error(`board: unknown option '${unknown}'`);
  return { template: 'board', a, b, out, options };
}

export function stack(...items: (Node | StackOptions)[]): LayoutItem {
  const isNode = (item: Node | StackOptions): item is Node => typeof (item as Node).id === 'number';
  const nodes = items.filter(isNode);
  const options = Object.assign({}, ...items.filter((item) => !isNode(item))) as StackOptions;
  return { template: 'stack', nodes, options };
}

export function node(n: Node, options: { at?: Vec3 } = {}): LayoutItem {
  const [, cols] = rowsCols(n);
  if (cols > 1) throw new Error(`node: '${n.name}' has shape (${n.shape.join(', ')}); use stack for tensors with more than one element per row`);
  return { template: 'node', node: n, options };
}

export const BOARD_COLUMN_SPACING = 1.45;
export const BOARD_ROW_GAP = 0.7;
export const BOARD_FRONT = 2.9;
export const BOARD_BACK_OFFSET = 1.3;
export const BOARD_A_OFFSET = 4.3;
export const BOARD_A_SPACING = 0.95;
export const BATCH_ROW_SPACING = 1.4;
export const STACK_LEVEL = 5.2;
export const STACK_BACK_OFFSET = 0.7;
export const SIDE_NODE_OFFSET = 2.6;
export const SELECT_NODE_RISE = 2.6;
export const SCALAR_NODE_RISE = 0.7;
export const SCALAR_ROW_GAP = 3.0;
export const LINE_HEIGHT = 0.03;
export const PLATE_THICKNESS = 0.04;
export const AXIS_OFFSET = 2.0;
export const PLATE_SIDE_MARGIN = 0.75;
export const ROW_LABEL_MARGIN = 0.15;

export const FOOTPRINTS = {
  bar: [0.62, 0.62],
  a: [0.6, 0.6],
  tile: [0.58, 0.4],
  disc: [0.54, 0.54],
  box: [1.1, 0.8],
} as const;
export const DISC_RADIUS = 0.27;
export const DISC_HEIGHT = 0.17;
export const BOX_HEIGHT = 0.5;

export type BlockShape = 'bar' | 'tile' | 'disc' | 'box';
export type BlockRole = 'board-a' | 'board-b' | 'board-out' | 'planar' | 'scalar';

export interface Element {
  index: number[];
  row: number;
  col: number;
  x: number;
  y: number;
  z: number;
}

export interface Plate {
  x: number;
  y: number;
  z: number;
  w: number;
  d: number;
}

export interface Slab {
  plate: Plate;
  used: boolean;
}

export interface Anchor {
  x: number;
  y: number;
  z: number;
  anchor: 'left' | 'below' | 'above' | 'right';
}

export interface TextLabel extends Anchor {
  text: string;
  col: number;
}

export interface Block {
  node: Node;
  role: BlockRole;
  shape: BlockShape;
  rows: number;
  cols: number;
  batched: boolean;
  elements: Element[];
  footprint: readonly [number, number];
  plate?: Plate;
  slabs?: Slab[];
  chunks?: Plate[];
  rowGroups?: number[];
  axis?: { x: number; z: number };
  title: Anchor;
  classLabels: TextLabel[];
  rowLabels: TextLabel[];
  frameHalf?: [number, number];
  frameRows: { z: number; y: number }[];
  scales: Scales;
  extent: Vec3[];
  key?: string;
  copyOf?: Node;
  slot?: readonly Node[];
  hidden?: boolean;
  quiet?: boolean;
  untitled?: boolean;
  table?: { label: Node; half: [number, number] };
}

export interface BoardGeometry {
  op: Node;
  a: Block;
  sharedA: boolean;
  b: Block;
  out: Block;
  y: number;
  previous?: Block;
}

export interface ResolvedLayout {
  blocks: Block[];
  boards: BoardGeometry[];
  levels?: LevelLayout;
}

export interface LayoutContext {
  visible: readonly Node[];
  aligned: Map<Node, Node>;
  targetOf(label: Node, row: number): number;
  focus: number;
  folded?: (id: string) => boolean;
  plan?: LevelPlan;
}

const LAYOUT_OPS: ReadonlySet<string> = new Set(['view', 'reshape', 'transpose', 'permute', 'unsqueeze', 'squeeze', 'contiguous', 'identity', 'grad_of']);

export function isLayoutOp(n: Node): boolean {
  return n.kind === 'op' && LAYOUT_OPS.has(n.op as string);
}

export function layoutSourceIndex(n: Node, index: readonly number[]): number[] {
  const input = n.inputs[0];
  if (n.op === 'transpose') {
    const { dim0, dim1 } = n.attrs as { dim0: number; dim1: number };
    const d0 = normDim(dim0, index.length);
    const d1 = normDim(dim1, index.length);
    const out = [...index];
    out[d0] = index[d1];
    out[d1] = index[d0];
    return out;
  }
  if (n.op === 'permute') {
    const dims = (n.attrs as { dims: number[] }).dims.map((d) => normDim(d, index.length));
    const out = [...index];
    dims.forEach((d, k) => {
      out[d] = index[k];
    });
    return out;
  }
  return indexOf(input.shape, offsetOf(n.shape, index));
}

export function sourceIndex(n: Node, source: Node, index: readonly number[]): number[] | undefined {
  let node = n;
  let at = [...index];
  while (node !== source) {
    if (!isLayoutOp(node)) return undefined;
    at = layoutSourceIndex(node, at);
    node = node.inputs[0];
  }
  return at;
}

function splitsLastDim(shape: readonly number[], out: readonly number[]): boolean {
  const lead = shape.findIndex((size) => size !== 1);
  const input = lead < 0 ? shape : shape.slice(lead);
  return out.length === input.length + 1 && input.slice(0, -1).every((size, d) => size === out[d]) && out[out.length - 2] * out[out.length - 1] === input[input.length - 1];
}

const chunkAxes = new WeakMap<Node, number>();

function derivedChunkAxis(n: Node): number {
  if (n.kind !== 'op') return 0;
  const input = n.inputs[0];
  const inherited = n.inputs.find((i) => i.shape.length === 3);
  switch (n.op) {
    case 'view':
    case 'reshape':
      if (splitsLastDim(input.shape, n.shape)) return n.shape.length - 2;
      return input.shape.join() === n.shape.join() ? chunkAxis(input) : 0;
    case 'transpose': {
      if (input.shape.length !== 3) return 0;
      const { dim0, dim1 } = n.attrs as { dim0: number; dim1: number };
      const axis = chunkAxis(input);
      const d0 = normDim(dim0, 3);
      const d1 = normDim(dim1, 3);
      return axis === d0 ? d1 : axis === d1 ? d0 : axis;
    }
    case 'permute':
      return input.shape.length === 3 ? (n.attrs as { dims: number[] }).dims.map((d) => normDim(d, 3)).indexOf(chunkAxis(input)) : 0;
    case 'unsqueeze':
      return input.shape.length === 2 ? normDim((n.attrs as { dim: number }).dim, 3) : 0;
  }
  if (explainOf(n.op as string)?.pattern === 'contract') return 0;
  return inherited === undefined ? 0 : chunkAxis(inherited);
}

export function chunkAxis(n: Node): number {
  const known = chunkAxes.get(n);
  if (known !== undefined) return known;
  const axis = n.shape.length === 3 ? derivedChunkAxis(n) : 0;
  chunkAxes.set(n, axis);
  return axis;
}

export function plateAxes(n: Node): { chunk: number; row: number; col: number } {
  const chunk = chunkAxis(n);
  const [row, col] = [0, 1, 2].filter((d) => d !== chunk);
  return { chunk, row, col };
}

export function rowsCols(n: Node, shape: readonly number[] = n.shape): [number, number] {
  if (shape.length === 0) return [1, 1];
  if (shape.length === 1) return [1, shape[0]];
  if (shape.length === 3) {
    const { chunk, row, col } = plateAxes(n);
    return [shape[row], shape[chunk] * shape[col]];
  }
  if (shape.length === 4) return [shape[2], shape[0] * shape[1] * shape[3]];
  return [shape[0], shape[1]];
}

export function rowCol(n: Node, index: readonly number[], shape: readonly number[] = n.shape): [number, number] {
  if (shape.length === 0) return [0, 0];
  if (shape.length === 1) return [0, index[0]];
  if (shape.length === 3) {
    const { chunk, row, col } = plateAxes(n);
    return [index[row], index[chunk] * shape[col] + index[col]];
  }
  if (shape.length === 4) return [index[2], (index[0] * shape[1] + index[1]) * shape[3] + index[3]];
  return [index[0], index[1]];
}

export function plates(n: Node, shape: readonly number[] = n.shape): { count: number; width: number } {
  if (shape.length === 4) return { count: shape[0] * shape[1], width: shape[3] };
  if (shape.length !== 3) return { count: 1, width: rowsCols(n, shape)[1] };
  const { chunk, col } = plateAxes(n);
  return { count: shape[chunk], width: shape[col] };
}

function checkDims(n: Node): void {
  if (n.shape.length > 4) throw new Error(`scene3d: node '${n.name}' with shape ${formatShape(n.shape)} has more than four dimensions`);
}

function elementsOf(n: Node, position: (row: number, col: number) => Vec3): Element[] {
  checkDims(n);
  const elements: Element[] = [];
  for (let offset = 0; offset < n.value.size; offset++) {
    const index = indexOf(n.shape, offset);
    const [row, col] = rowCol(n, index);
    const [x, y, z] = position(row, col);
    elements.push({ index, row, col, x, y, z });
  }
  return elements;
}

export function targetColumn(label: Node, row: number): number {
  const data = label.value.data;
  return Math.round(data[Math.min(row, data.length - 1)]);
}

export function labelBehind(n: Node): Node | undefined {
  if (n.kind === 'label') return n;
  return n.op === 'view' && n.inputs[0].kind === 'label' ? n.inputs[0] : undefined;
}

export function classAlignment(nodes: readonly Node[]): Map<Node, Node> {
  const aligned = new Map<Node, Node>();
  const classesOf = (label: Node) => (label.attrs.classes as number | undefined) ?? label.tokens?.length;
  const lastDim = (n: Node) => (n.shape.length === 0 ? undefined : n.shape[n.shape.length - 1]);
  for (const n of nodes) {
    if (n.kind !== 'op' || explainOf(n.op as string)?.pattern !== 'select') continue;
    const index = n.inputs.find((input) => labelBehind(input) !== undefined && classesOf(labelBehind(input) as Node) !== undefined);
    if (index === undefined) continue;
    const label = labelBehind(index) as Node;
    for (const input of n.inputs) {
      if (input !== index && lastDim(input) === classesOf(label)) aligned.set(input, label);
    }
  }
  for (let id = nodes.length - 1; id >= 0; id--) {
    const n = nodes[id];
    const label = aligned.get(n);
    if (label === undefined || n.kind !== 'op') continue;
    for (const input of n.inputs) {
      if (input.kind !== 'label' && lastDim(input) === classesOf(label)) aligned.set(input, label);
    }
  }
  return aligned;
}

function labelsFor(n: Node, ctx: LayoutContext, cols: number, position: (col: number) => Vec3, anchor: Anchor['anchor']): TextLabel[] {
  const label = ctx.aligned.get(n);
  const tokens = label?.tokens;
  const labels: TextLabel[] = [];
  for (let col = 0; col < cols; col++) {
    const [x, y, z] = position(col);
    const text = tokens === undefined ? String(col + 1) : tokenText(tokens[col] ?? String(col + 1));
    labels.push({ text, col, x, y, z, anchor });
  }
  return labels;
}

export function tokenText(token: string): string {
  return token.replace(/^ +/, (spaces) => '\u2423'.repeat(spaces.length));
}

function scalesFor(n: Node, options: TemplateOptions): Scales {
  return options.scales?.[n.name] ?? {};
}

function layoutBoard(item: Extract<LayoutItem, { template: 'board' }>, ctx: LayoutContext, placed: Map<Node, Block>): BoardGeometry {
  const { a, b, out, options } = item;
  const [ax, ay, az] = options.at ?? [0, 0, 0];
  const s = options.scale ?? 1;
  const gap = options.gap ?? BOARD_ROW_GAP;
  checkDims(a);
  checkDims(out);
  if (b.shape.length !== 2) throw new Error(`board: node '${b.name}' with shape ${formatShape(b.shape)} must be two-dimensional`);
  const [d, V] = b.shape;
  const [B] = rowsCols(out);
  const xs = (i: number) => ax + (i - (V - 1) / 2) * BOARD_COLUMN_SPACING * s;
  const zf = az + BOARD_FRONT * s;
  const wr = (k: number) => zf - BOARD_BACK_OFFSET * s - (d - 1 - k) * gap * s;
  const zRow = (r: number) => zf + (B - 1 - r) * BATCH_ROW_SPACING * s;
  const xa = (k: number) => xs(0) - BOARD_A_OFFSET * s + k * BOARD_A_SPACING * s;
  const y = ay;
  const aCenter = (xa(0) + xa(d - 1)) / 2;
  const rowDepth = (B - 1) * BATCH_ROW_SPACING * s;
  const rowCenter = (zRow(0) + zRow(B - 1)) / 2;
  const plateWidth = (V - 1) * BOARD_COLUMN_SPACING * s + 2 * PLATE_SIDE_MARGIN;

  const existingA = placed.get(a);
  const aBlock: Block = existingA ?? {
    node: a,
    role: 'board-a',
    shape: 'bar',
    rows: B,
    cols: d,
    batched: B > 1,
    elements: elementsOf(a, (r, k) => [xa(k), y, zRow(r)]),
    footprint: FOOTPRINTS.a,
    plate: { x: aCenter, y, z: rowCenter, w: (d - 1) * BOARD_A_SPACING * s + 1.0, d: 1.0 + rowDepth },
    title: { x: aCenter, y: y - 0.6, z: zRow(0) + 1.35, anchor: 'below' },
    classLabels: Array.from({ length: d }, (_, k) => ({ text: `k=${k + 1}`, col: k, x: xa(k), y, z: zRow(0) + 0.7, anchor: 'below' as const })),
    rowLabels: [],
    frameRows: [],
    scales: scalesFor(a, options),
    extent: [],
  };
  const bBlock: Block = {
    node: b,
    role: 'board-b',
    shape: 'tile',
    rows: d,
    cols: V,
    batched: false,
    elements: elementsOf(b, (k, i) => [xs(i), y, wr(k)]),
    footprint: FOOTPRINTS.tile,
    plate: { x: ax, y, z: (wr(0) + wr(d - 1)) / 2, w: plateWidth, d: (wr(d - 1) - wr(0)) + 0.85 },
    title: { x: xs(0) - 1.0, y, z: wr(0) - 1.0, anchor: 'left' },
    classLabels: labelsFor(b, ctx, V, (i) => [xs(i), y + 0.05, wr(0) - 0.45], 'above'),
    rowLabels: Array.from({ length: d }, (_, k) => ({ text: `k=${k + 1}`, col: k, x: ax - plateWidth / 2 - ROW_LABEL_MARGIN, y: y + LINE_HEIGHT, z: wr(k), anchor: 'left' as const })),
    frameHalf: ctx.aligned.has(b) ? [0.4, (wr(d - 1) - wr(0)) / 2 + 0.32] : undefined,
    frameRows: [{ z: (wr(0) + wr(d - 1)) / 2, y }],
    scales: scalesFor(b, options),
    extent: [],
  };
  const outBlock: Block = {
    node: out,
    role: 'board-out',
    shape: 'bar',
    rows: B,
    cols: V,
    batched: B > 1,
    elements: elementsOf(out, (r, i) => [xs(i), y, zRow(r)]),
    footprint: FOOTPRINTS.bar,
    plate: { x: ax, y, z: rowCenter, w: plateWidth, d: 1.0 + rowDepth },
    title: { x: ax, y: y - 0.75, z: zRow(0) + 1.45, anchor: 'below' },
    classLabels: [],
    rowLabels: [],
    frameHalf: ctx.aligned.has(out) ? [0.45, 0.45] : undefined,
    frameRows: Array.from({ length: B }, (_, r) => ({ z: zRow(r), y })),
    scales: scalesFor(out, options),
    extent: [],
  };
  placed.set(a, aBlock);
  placed.set(b, bBlock);
  placed.set(out, outBlock);
  return { op: out, a: aBlock, sharedA: existingA !== undefined, b: bBlock, out: outBlock, y };
}

function shiftBlock(block: Block, [dx, dy, dz]: Vec3): void {
  const shift = (p: { x: number; y: number; z: number }) => {
    p.x += dx;
    p.y += dy;
    p.z += dz;
  };
  block.elements.forEach(shift);
  if (block.plate !== undefined) shift(block.plate);
  if (block.axis !== undefined) {
    block.axis.x += dx;
    block.axis.z += dz;
  }
  shift(block.title);
  block.classLabels.forEach(shift);
  block.rowLabels.forEach(shift);
  for (const row of block.frameRows) {
    row.y += dy;
    row.z += dz;
  }
  block.extent = block.extent.map(([x, y, z]): Vec3 => [x + dx, y + dy, z + dz]);
}

interface StackCursor {
  y: number;
  xs: number[];
  zs: number[];
  rowXs?: number[];
  spanXs?: number[];
  row?: boolean;
}

function shiftCursor(cursor: StackCursor, [dx, dy, dz]: Vec3): StackCursor {
  const add = (values: number[] | undefined, d: number) => values?.map((v) => v + d);
  return { y: cursor.y + dy, xs: add(cursor.xs, dx)!, zs: add(cursor.zs, dz)!, rowXs: add(cursor.rowXs, dx), spanXs: add(cursor.spanXs, dx), row: cursor.row };
}

interface Layer {
  y: number;
  xs: number[];
  zs: number[];
  cols: number;
  rows: number;
}

function layerOf(block: Block): Layer {
  const xs = Array.from({ length: block.cols }, (_, c) => block.elements.find((e) => e.col === c)!.x);
  const zs = Array.from({ length: block.rows }, (_, r) => block.elements.find((e) => e.row === r)!.z);
  return { y: block.elements[0].y, xs, zs, cols: block.cols, rows: block.rows };
}

function placedSource(n: Node, placed: Map<Node, Block>): Block | undefined {
  for (const input of n.inputs) {
    const block = placed.get(input);
    if (block !== undefined && input.kind !== 'label') return block;
  }
  return undefined;
}

function planarBlock(n: Node, y: number, xs: number[], zs: number[], ctx: LayoutContext, options: StackOptions): Block {
  const [rows, cols] = rowsCols(n);
  const s = options.scale ?? 1;
  const axisX = xs[0] - AXIS_OFFSET * s;
  const front = Math.max(...zs);
  const back = Math.min(...zs);
  const block: Block = {
    node: n,
    role: 'planar',
    shape: 'bar',
    rows,
    cols,
    batched: rows > 1,
    elements: elementsOf(n, (r, c) => [xs[c], y, zs[r]]),
    footprint: FOOTPRINTS.bar,
    plate: { x: (xs[0] + xs[cols - 1]) / 2, y, z: (zs[0] + zs[rows - 1]) / 2, w: xs[cols - 1] - xs[0] + 2 * PLATE_SIDE_MARGIN, d: 1.0 + (front - back) },
    axis: { x: axisX, z: back - 0.5 },
    title: { x: axisX - 0.95, y: y + 0.35, z: zs[0], anchor: 'left' },
    classLabels: labelsFor(n, ctx, cols, (c) => [xs[c], y + 0.2, front + 0.7], 'below'),
    rowLabels: [],
    frameHalf: ctx.aligned.has(n) ? [0.45, 0.45] : undefined,
    frameRows: zs.map((z) => ({ z, y })),
    scales: scalesFor(n, options),
    extent: [],
  };
  return block;
}

function scalarBlock(n: Node, shape: 'disc' | 'box', position: (row: number) => Vec3, options: TemplateOptions, extent: Vec3[] = [], anchor: Anchor['anchor'] = 'above'): Block {
  const [rows] = rowsCols(n);
  const elements = elementsOf(n, (r) => position(r));
  const top = shape === 'disc' ? DISC_HEIGHT : BOX_HEIGHT;
  const first = elements[0];
  return {
    node: n,
    role: 'scalar',
    shape,
    rows,
    cols: 1,
    batched: rows > 1,
    elements,
    footprint: FOOTPRINTS[shape],
    title: { x: first.x, y: first.y + top / 2 + 0.45, z: first.z, anchor },
    classLabels: [],
    rowLabels: [],
    frameRows: [],
    scales: scalesFor(n, options),
    extent,
  };
}

function patternOf(n: Node): string | undefined {
  return n.kind === 'op' ? explainOf(n.op as string)?.pattern : undefined;
}

function isLoss(n: Node): boolean {
  return n.graph.findLoss() === n;
}

function layoutStack(item: Extract<LayoutItem, { template: 'stack' }>, ctx: LayoutContext, placed: Map<Node, Block>): Block[] {
  const { options } = item;
  const [ax, ay, az] = options.at ?? [0, 0, 0];
  const s = options.scale ?? 1;
  const level = (options.level ?? STACK_LEVEL) * s;
  const back = (options.back ?? STACK_BACK_OFFSET) * s;
  const colGap = (options.gap ?? BOARD_COLUMN_SPACING) * s;
  const blocks: Block[] = [];
  let planar: Layer | undefined;
  let last: StackCursor | undefined;
  for (const n of item.nodes) {
    const existing = placed.get(n);
    if (existing !== undefined) {
      const layer = layerOf(existing);
      if (existing.cols > 1) planar = layer;
      last = layer;
      continue;
    }
    if (!ctx.visible.includes(n)) continue;
    const [rows, cols] = rowsCols(n);
    const source = placedSource(n, placed);
    const pattern = patternOf(n);
    let block: Block;
    if (cols > 1) {
      const base = planar ?? last;
      const y = base === undefined ? ay : (planar === undefined ? base.y : planar.y + level);
      const cx = base === undefined ? ax : (base.xs[0] + base.xs[base.xs.length - 1]) / 2;
      const xs = base !== undefined && base.xs.length === cols ? base.xs : Array.from({ length: cols }, (_, c) => cx + (c - (cols - 1) / 2) * colGap);
      const shift = planar === undefined ? 0 : back;
      const zs = base !== undefined && base.zs.length === rows ? base.zs.map((z) => z - shift) : Array.from({ length: rows }, (_, r) => az - r * BATCH_ROW_SPACING * s);
      block = planarBlock(n, y, xs, zs, ctx, options);
      planar = layerOf(block);
      last = planar;
    } else if (source !== undefined && source.cols > 1 && (pattern === 'reduce' || pattern === 'dense')) {
      const layer = layerOf(source);
      const x = layer.xs[layer.cols - 1] + SIDE_NODE_OFFSET * s;
      const y = layer.y + level / 2;
      const zs = layer.zs.map((z) => z - back / 2);
      block = scalarBlock(n, 'disc', (r) => [x, y, zs[Math.min(r, layer.rows - 1)]], options);
      last = { y, xs: [x], zs, row: true };
    } else if (source !== undefined && source.cols > 1 && pattern === 'select') {
      const layer = layerOf(source);
      const label = ctx.aligned.get(source.node);
      const y = layer.y + SELECT_NODE_RISE * s;
      const zOf = (r: number) => (rows === layer.rows ? layer.zs[r] : (layer.zs[0] + layer.zs[layer.rows - 1]) / 2);
      const xOf = (r: number) => {
        const row = rows === layer.rows ? r : ctx.focus;
        const col = label === undefined ? Math.floor((layer.cols - 1) / 2) : ctx.targetOf(label, row);
        return layer.xs[Math.max(0, Math.min(layer.cols - 1, col))];
      };
      const extent = layer.xs.map((x): Vec3 => [x, y, zOf(0)]);
      block = scalarBlock(n, isLoss(n) ? 'box' : 'disc', (r) => [xOf(r), y, zOf(r)], options, extent, layer.rows > 1 ? 'left' : 'right');
      last = { y, xs: [(layer.xs[0] + layer.xs[layer.cols - 1]) / 2], zs: block.elements.map((e) => e.z), rowXs: block.elements.map((e) => e.x), spanXs: layer.xs };
    } else {
      const base = last ?? { y: ay, xs: [ax], zs: [az] };
      const row = options.scalars === 'row' && base.row === true;
      const y = row ? base.y : base.y + SCALAR_NODE_RISE * s;
      const center = (base.xs[0] + base.xs[base.xs.length - 1]) / 2 + (row ? SCALAR_ROW_GAP * s : 0);
      const rowXs = base.rowXs !== undefined && rows === base.rowXs.length ? base.rowXs : undefined;
      const xOf = (r: number) => rowXs?.[r] ?? center;
      const zOf = (r: number) => (rows === base.zs.length ? base.zs[r] : (base.zs[0] + base.zs[base.zs.length - 1]) / 2);
      const spanXs = rowXs === undefined ? undefined : base.spanXs;
      const extent = (spanXs ?? []).map((x): Vec3 => [x, y, zOf(0)]);
      block = scalarBlock(n, isLoss(n) ? 'box' : 'disc', (r) => [xOf(r), y, zOf(r)], options, extent, base.zs.length > 1 ? 'left' : 'right');
      last = { y, xs: [center], zs: block.elements.map((e) => e.z), rowXs, spanXs, row };
    }
    const offset = options.offsets?.[n.name];
    if (offset !== undefined) {
      shiftBlock(block, offset);
      if (block.role === 'planar') planar = layerOf(block);
      last = block.role === 'planar' ? planar : shiftCursor(last!, offset);
    }
    placed.set(n, block);
    blocks.push(block);
  }
  return blocks;
}

export function resolveLayout(items: LayoutItem[], ctx: LayoutContext): ResolvedLayout {
  const placed = new Map<Node, Block>();
  const blocks: Block[] = [];
  const boards: BoardGeometry[] = [];
  const levels = ctx.plan?.resolve({ folded: ctx.folded ?? (() => false), aligned: ctx.aligned });
  if (levels !== undefined) {
    blocks.push(...levels.blocks);
    boards.push(...levels.boards);
  }
  for (const item of items) {
    if (item.template === 'board') {
      const geometry = layoutBoard(item, ctx, placed);
      boards.push(geometry);
      if (!geometry.sharedA) blocks.push(geometry.a);
      blocks.push(geometry.b, geometry.out);
    } else if (item.template === 'stack') {
      blocks.push(...layoutStack(item, ctx, placed));
    } else {
      if (placed.has(item.node)) continue;
      const at = item.options.at ?? [0, 0, 0];
      const block = scalarBlock(item.node, isLoss(item.node) ? 'box' : 'disc', (r) => [at[0], at[1], at[2] + r * BATCH_ROW_SPACING], {});
      placed.set(item.node, block);
      blocks.push(block);
    }
  }
  return { blocks: blocks.filter((block) => ctx.visible.includes(block.node)), boards, levels };
}

export function sameData(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) return false;
  if (Array.isArray(a)) {
    if (!Array.isArray(b) || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (!sameData(a[i], b[i])) return false;
    return true;
  }
  if (a instanceof Map) {
    if (!(b instanceof Map) || a.size !== b.size) return false;
    for (const [k, v] of a) if (!b.has(k) || !sameData(v, b.get(k))) return false;
    return true;
  }
  if (Object.getPrototypeOf(a) !== Object.prototype || Object.getPrototypeOf(b) !== Object.prototype) return false;
  const x = a as Record<string, unknown>;
  const y = b as Record<string, unknown>;
  let count = 0;
  for (const k in x) {
    if (!(k in y) || !sameData(x[k], y[k])) return false;
    count++;
  }
  for (const _ in y) count--;
  return count === 0;
}

interface ElementIndex {
  byIndex: Map<string, Element>;
  byCell: Map<string, Element>;
  position: Map<Element, number>;
}

const elementIndexes = new WeakMap<readonly Element[], ElementIndex>();

function elementIndex(elements: readonly Element[]): ElementIndex {
  const known = elementIndexes.get(elements);
  if (known !== undefined) return known;
  const index: ElementIndex = { byIndex: new Map(), byCell: new Map(), position: new Map() };
  elements.forEach((e, i) => index.position.set(e, i));
  for (const e of elements) {
    const key = e.index.join(',');
    if (!index.byIndex.has(key)) index.byIndex.set(key, e);
    const cell = `${e.row},${e.col}`;
    if (!index.byCell.has(cell)) index.byCell.set(cell, e);
  }
  elementIndexes.set(elements, index);
  return index;
}

export function elementAt(block: Block, index: readonly number[]): Element {
  return elementIndex(block.elements).byIndex.get(index.join(','))!;
}

export function elementInCell(block: Block, row: number, col: number): Element | undefined {
  return elementIndex(block.elements).byCell.get(`${row},${col}`);
}

export function elementPosition(block: Block, element: Element): number {
  return elementIndex(block.elements).position.get(element)!;
}
