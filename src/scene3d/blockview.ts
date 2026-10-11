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

const VALUE_LABEL_GAP = 0.1;
const NODE_TITLE_OFFSET: Vec3 = [0.05, 0.28, -0.2];
const NODE_VALUE_OFFSET = 0.32;
const LEVEL_PRECISION = 1000;

function texOf(label: Label, tex: string): void {
  label.setContent(tex, () => render(tex));
}

function valueOf(label: Label, v: number): void {
  if (Number.isFinite(v)) label.setText(formatNumber(v));
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

export function refreshBlock(c: SceneController, renderable: Renderable, now: number, zoom: number): void {
  const dir = c.state.dir;
  const backward = dir === 'backward';
  const { block } = renderable;
  const node = block.node;
  const hidden = block.hidden === true;
  const revealed = c.state.revealed(node) && !hidden;
  const active = c.state.active(node);
  const far = renderable.far;
  const values = valuesOf(c, node);
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
  const showValues = revealed && stepped && block.quiet !== true && !c.lod && (!c.follow || zoom >= VALUE_MIN_ZOOM);
  const named = revealed && !c.lod && !far;
  const title = renderable.title!;
  title.el.dataset.node = node.name;
  title.offset = c.options.labels?.[node.name];
  title.visible = named && block.untitled !== true;
  const view = block.copyOf === undefined ? [...(c.levelLayout?.sourceOf ?? [])].find(([, source]) => source === node)?.[0] : undefined;
  const continued = c.levelLayout?.boards.some((b) => b.op === node && b.previous !== undefined) === true;
  const tex = view === undefined ? node.tex : continued ? view.tex : `${view.tex}=${node.tex}`;
  texOf(title, backward ? `\\partial\\ell/\\partial ${(view ?? node).tex}` : tex);
  if (renderable.bars !== undefined) {
    title.pos = [block.title.x, block.title.y, block.title.z];
    title.anchor = block.title.anchor === 'left' ? 'right-middle' : block.title.anchor === 'right' ? 'left-middle' : block.title.anchor === 'below' ? 'bottom-center' : 'top-center';
    renderable.values.forEach((label, i) => {
      const element = block.elements[i];
      valueOf(label, values[i]);
      label.el.dataset.node = node.name;
      label.el.dataset.index = element.index.join(',');
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
      valueOf(label, values[i]);
      label.el.dataset.node = node.name;
      label.el.dataset.index = element.index.join(',');
      label.pos = [element.x + NODE_VALUE_OFFSET, element.y + mesh.height / 2, element.z];
      label.visible = showValues && focusedRow(element.row);
    });
    const colors = block.elements.map((element, i) => nodeColor(values[i], c.theme, { dir, revealed, active, far, focused: focusedRow(element.row) }));
    mesh.update(block.elements, colors, block.elements.map(() => revealed));
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
    renderable.plate.update({ revealed, far }, frameXs, ticks, scale, frameZs);
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
    colors.push(barColor(values[i], c.theme, { active: revealed && active, far, focused }));
    outline.push(outlineOpacity({ active: active && revealed, far, tile: block.shape === 'tile' }));
    const label = renderable.values[i];
    if (label !== undefined) {
      const h = heights[i];
      label.anchor = h < 0 ? 'bottom-center' : 'top-center';
      label.pos = [element.x, element.y + (h >= 0 ? h + VALUE_LABEL_GAP : h - VALUE_LABEL_GAP), element.z];
    }
  });
  bars.update(block.elements, { heights, clipped: renderable.clipped, colors, outlineOpacity: outline });
}
