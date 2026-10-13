import { Vector3 } from 'three';
import { type Node, isIntegerDtype } from '../autograd';
import { render } from '../formula';
import { isLeafLike } from '../steps';
import { formatNumber } from '../tensor';
import type { LinearRgb } from '../theme';
import type { BarMesh } from './blocks';
import { barColor, nodeColor, outlineOpacity } from './colors';
import type { SceneController } from './controller';
import type { Label } from './labels';
import type { Vec3 } from './layout';
import { type Renderable, targetOf, valuesOf } from './renderable';
import { barHeight } from './scales';

export const FAR_LEVELS = 2;
export const VALUE_MIN_ZOOM = 0.55;
export const VALUE_ALL_ROWS_ZOOM = 1.45;
export const SEE_THROUGH_MIN_OVERLAP = 6;
export const SEE_THROUGH_DEPTH_GAP = 0.3;

const VALUE_LABEL_GAP = 0.1;
const NODE_TITLE_OFFSET: Vec3 = [0.05, 0.28, -0.2];
const NODE_VALUE_OFFSET = 0.32;
const LEVEL_PRECISION = 1000;

function texOf(label: Label, tex: string): void {
  label.setContent(tex, () => render(tex));
}

function valueOf(label: Label, v: number, integer: boolean): void {
  if (Number.isFinite(v)) label.setText(formatNumber(v, integer ? { digits: 0 } : {}));
  else texOf(label, formatNumber(v, { tex: true }));
}

function levelKey(y: number): number {
  return Math.round(y * LEVEL_PRECISION);
}

export function markFar(c: SceneController): void {
  const levels = [...new Set(c.renderables.map((r) => levelKey(r.block.elements[0].y)))].sort((a, b) => a - b);
  const levelOf = (r: Renderable) => levels.indexOf(levelKey(r.block.elements[0].y));
  const stepLevels = c.renderables.filter((r) => r.block.copyOf === undefined && !isLeafLike(r.block.node) && c.state.active(r.block.node)).map(levelOf);
  const enabled = c.follow && c.farOn && !c.lod && stepLevels.length > 0;
  for (const r of c.renderables) {
    const level = levelOf(r);
    r.far = enabled && !c.state.active(r.block.node) && Math.min(...stepLevels.map((s) => Math.abs(s - level))) > FAR_LEVELS;
  }
}

interface ScreenBox {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  depth: number;
}

const corner = new Vector3();

function screenBox(c: SceneController, renderable: Renderable, direction: Vector3): ScreenBox | undefined {
  const { block } = renderable;
  if (block.hidden === true || block.elements.length === 0) return undefined;
  const [w, d] = block.footprint;
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  const grow = (x: number, y: number, z: number) => {
    lo[0] = Math.min(lo[0], x);
    lo[1] = Math.min(lo[1], y);
    lo[2] = Math.min(lo[2], z);
    hi[0] = Math.max(hi[0], x);
    hi[1] = Math.max(hi[1], y);
    hi[2] = Math.max(hi[2], z);
  };
  const lift = renderable.nodes?.height;
  block.elements.forEach((e, i) => {
    const h = lift ?? renderable.transition.current[i] ?? 0;
    grow(e.x - w / 2, e.y + Math.min(h, 0), e.z - d / 2);
    grow(e.x + w / 2, e.y + Math.max(h, 0), e.z + d / 2);
  });
  const plate = block.plate;
  if (plate !== undefined) {
    grow(plate.x - plate.w / 2, plate.y, plate.z - plate.d / 2);
    grow(plate.x + plate.w / 2, plate.y, plate.z + plate.d / 2);
  }
  const box = { x0: Infinity, x1: -Infinity, y0: Infinity, y1: -Infinity, depth: 0 };
  for (let k = 0; k < 8; k++) {
    corner.set(k & 1 ? hi[0] : lo[0], k & 2 ? hi[1] : lo[1], k & 4 ? hi[2] : lo[2]).project(c.camera);
    const px = ((corner.x + 1) / 2) * c.width;
    const py = ((1 - corner.y) / 2) * c.height;
    box.x0 = Math.min(box.x0, px);
    box.x1 = Math.max(box.x1, px);
    box.y0 = Math.min(box.y0, py);
    box.y1 = Math.max(box.y1, py);
  }
  corner.set((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2).sub(c.camera.position);
  box.depth = corner.dot(direction);
  return box;
}

function stepInvolved(c: SceneController): Set<Node> {
  const involved = new Set(c.stepNodes);
  if (c.state.dir === 'forward') for (const node of c.stepNodes) for (const input of node.inputs) involved.add(input);
  else for (const node of c.visible) if (node.inputs.some((input) => c.stepNodes.has(input))) involved.add(node);
  return involved;
}

function covers(box: ScreenBox, target: ScreenBox): boolean {
  return (
    box.depth < target.depth - SEE_THROUGH_DEPTH_GAP &&
    Math.min(box.x1, target.x1) - Math.max(box.x0, target.x0) > SEE_THROUGH_MIN_OVERLAP &&
    Math.min(box.y1, target.y1) - Math.max(box.y0, target.y0) > SEE_THROUGH_MIN_OVERLAP
  );
}

export function markSeeThrough(c: SceneController): boolean {
  const enabled = c.seeThroughOn && !c.lod && c.options.steps !== undefined && !(c.state.dir === 'forward' && (c.state.index === 0 || c.state.kind === 'inputs'));
  const boxes = new Map<Renderable, ScreenBox>();
  if (enabled) {
    const direction = c.camera.getWorldDirection(new Vector3());
    for (const r of c.renderables) {
      const box = screenBox(c, r, direction);
      if (box !== undefined) boxes.set(r, box);
    }
  }
  const involved = enabled ? stepInvolved(c) : new Set<Node>();
  const inStep = (node: Node) => c.state.active(node) || involved.has(node);
  const targets = [...boxes].filter(([r]) => inStep(r.block.node) && c.state.revealed(r.block.node)).map(([, box]) => box);
  let changed = false;
  for (const r of c.renderables) {
    const box = boxes.get(r);
    const next = box !== undefined && !inStep(r.block.node) && targets.some((t) => covers(box, t));
    if (next !== r.seeThrough) changed = true;
    r.seeThrough = next;
  }
  return changed;
}

export function refreshBlock(c: SceneController, renderable: Renderable, now: number, zoom: number): void {
  const dir = c.state.dir;
  const backward = dir === 'backward';
  const { block } = renderable;
  const node = block.node;
  const hidden = block.hidden === true;
  const revealed = c.state.revealed(node) && !hidden;
  const active = c.state.active(node);
  const far = renderable.far;
  const clear = !renderable.seeThrough;
  const values = valuesOf(c, node);
  const integer = !backward && isIntegerDtype(node.dtype);
  const scale = renderable.scale[dir];
  for (const group of [renderable.bars?.group, renderable.nodes?.group, renderable.plate?.group]) if (group !== undefined) group.visible = !hidden;
  const heights = new Float64Array(block.elements.length);
  for (let i = 0; i < heights.length; i++) {
    const { height, clipped } = barHeight(values[i], scale, revealed, renderable.cap);
    heights[i] = height;
    renderable.clipped[i] = revealed && clipped;
  }
  renderable.transition.retarget(heights, now);
  const focusedRow = (row: number) => !block.batched || row === c.state.focus;
  const front = Math.max(...block.elements.map((e) => e.z));
  const stepped = block.copyOf === undefined || backward ? c.stepNodes.has(node) : active && !c.stepNodes.has(node);
  const showValues = revealed && clear && stepped && block.quiet !== true && !c.lod && (!c.follow || zoom >= VALUE_MIN_ZOOM);
  const named = revealed && clear && !c.lod && !far;
  const title = renderable.title!;
  title.setData('node', node.name);
  title.offset = c.options.labels?.[node.name];
  title.visible = named && block.untitled !== true;
  texOf(title, backward ? `\\partial\\ell/\\partial ${node.tex}` : node.tex);
  if (renderable.bars !== undefined) {
    title.pos = [block.title.x, block.title.y, block.title.z];
    title.anchor = block.title.anchor === 'left' ? 'right-middle' : block.title.anchor === 'right' ? 'left-middle' : block.title.anchor === 'below' ? 'bottom-center' : 'top-center';
    renderable.values.forEach((label, i) => {
      const element = block.elements[i];
      valueOf(label, values[i], integer);
      label.setData('node', node.name);
      label.setData('index', element.index.join(','));
      const rowShown = !c.follow || element.z === front || zoom >= VALUE_ALL_ROWS_ZOOM;
      label.visible = showValues && focusedRow(element.row) && rowShown;
    });
  }
  if (renderable.nodes !== undefined) {
    const mesh = renderable.nodes;
    const first = block.elements[0];
    const middle = first.y + mesh.height / 2;
    title.pos = [first.x + NODE_TITLE_OFFSET[0], middle + NODE_TITLE_OFFSET[1], first.z + NODE_TITLE_OFFSET[2]];
    title.anchor = 'above-right';
    renderable.values.forEach((label, i) => {
      const element = block.elements[i];
      valueOf(label, values[i], integer);
      label.setData('node', node.name);
      label.setData('index', element.index.join(','));
      label.pos = [element.x + NODE_VALUE_OFFSET, element.y + mesh.height / 2, element.z];
      label.visible = showValues && focusedRow(element.row);
    });
    const colors = block.elements.map((element, i) => nodeColor(values[i], c.theme, { dir, revealed, active, far, focused: focusedRow(element.row) }));
    mesh.update(block.elements, colors, block.elements.map(() => revealed), renderable.seeThrough);
  }
  renderable.classes.forEach((label, i) => {
    const text = block.classLabels[i];
    const target = targetOf(c, block, block.rows - 1);
    const isTarget = c.aligned.has(node) && !block.batched && block.rows === 1 && target === text.col;
    label.setText(block.role === 'planar' && isTarget ? `${text.text} · 目标` : text.text);
    label.setFlag('vf-on', isTarget);
    label.pos = [text.x, text.y, text.z];
    label.anchor = text.anchor === 'above' ? 'top-center' : text.anchor === 'left' ? 'right-middle' : 'bottom-center';
    label.visible = named;
  });
  renderable.rows.forEach((label, i) => {
    const text = block.rowLabels[i];
    label.setText(text.text);
    label.pos = [text.x, text.y, text.z];
    label.anchor = text.anchor === 'right' ? 'left-middle' : 'right-middle';
    label.visible = named;
  });
  const ticks = renderable.ticks[dir];
  renderable.tickLabels.forEach((label, i) => {
    const axis = block.axis!;
    if (i < ticks.length) {
      label.setText(String(ticks[i]).replace('-', '−'));
      label.pos = [axis.x - 0.18, block.elements[0].y + ticks[i] * scale, axis.z];
      label.visible = named;
    } else label.visible = false;
  });
  if (renderable.plate !== undefined) {
    const table = block.table;
    const frameXs = block.frameRows.map((_, r) => {
      if (table !== undefined) return block.plate!.x;
      const target = targetOf(c, block, r);
      const column = block.elements.find((e) => e.col === (target ?? 0));
      return column?.x ?? block.elements[0].x;
    });
    const frameZs = table === undefined ? undefined : Array.from(table.label.value.data, (id) => block.elements.find((e) => e.row === Math.round(id))?.z ?? block.elements[0].z);
    renderable.plate.update({ revealed, far, seeThrough: renderable.seeThrough }, frameXs, ticks, scale, frameZs);
  }
}

export function updateBars(c: SceneController, renderable: Renderable, bars: BarMesh): void {
  const { block } = renderable;
  const node = block.node;
  const revealed = c.state.revealed(node);
  const active = c.state.active(node);
  const values = valuesOf(c, node);
  const heights = renderable.transition.current;
  const colors: LinearRgb[] = [];
  const outline: number[] = [];
  const far = renderable.far;
  block.elements.forEach((element, i) => {
    const focused = !block.batched || element.row === c.state.focus;
    const faded = far || block.slabs?.[element.index[0]]?.used === false;
    colors.push(barColor(values[i], c.theme, { active: revealed && active, far: faded, focused }));
    outline.push(outlineOpacity({ active: active && revealed, far: faded, tile: block.shape === 'tile' }));
    const label = renderable.values[i];
    if (label !== undefined) {
      const h = heights[i];
      label.anchor = h < 0 ? 'bottom-center' : 'top-center';
      label.pos = [element.x, element.y + (h >= 0 ? h + VALUE_LABEL_GAP : h - VALUE_LABEL_GAP), element.z];
    }
  });
  bars.update(block.elements, { heights, clipped: renderable.clipped, colors, outlineOpacity: outline, seeThrough: renderable.seeThrough });
}
