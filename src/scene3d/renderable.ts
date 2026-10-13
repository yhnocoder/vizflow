import type { Object3D } from 'three';
import type { Node } from '../autograd';
import { offsetOf } from '../tensor';
import { BarMesh, NodeMesh, PlateMesh, barBottom, barTop } from './blocks';
import type { SceneController } from './controller';
import type { Label } from './labels';
import { type Block, type Element, type Vec3, elementAt, sourceIndex, targetColumn } from './layout';
import { TICK_COUNT, capFor, niceTicks, scaleFor } from './scales';
import { Transition } from './transition';

export interface Renderable {
  block: Block;
  bars?: BarMesh;
  nodes?: NodeMesh;
  plate?: PlateMesh;
  transition: Transition;
  clipped: boolean[];
  cap: number;
  far: boolean;
  seeThrough: boolean;
  scale: { forward: number; backward: number };
  ticks: { forward: number[]; backward: number[] };
  title: Label | undefined;
  values: Label[];
  classes: Label[];
  rows: Label[];
  tickLabels: Label[];
  pickables: Object3D[];
}

export function createRenderable(c: SceneController, block: Block): Renderable {
  const node = block.node;
  const grad = node.grad;
  const cap = capFor(block.shape, node.shape.length);
  const scale = {
    forward: block.scales.forward ?? scaleFor(node.value.data, node.dtype, cap),
    backward: block.scales.backward ?? scaleFor(grad === null ? [] : grad.data, node.dtype, cap),
  };
  const renderable: Renderable = {
    block,
    transition: new Transition(block.elements.length, () => c.reduceMotion.matches),
    clipped: block.elements.map(() => false),
    cap,
    far: false,
    seeThrough: false,
    scale,
    ticks: {
      forward: block.axis === undefined ? [] : niceTicks(node.value.data, node.dtype, scale.forward, TICK_COUNT, cap),
      backward: block.axis === undefined || grad === null ? [] : niceTicks(grad.data, node.dtype, scale.backward, TICK_COUNT, cap),
    },
    title: undefined,
    values: [],
    classes: [],
    rows: [],
    tickLabels: [],
    pickables: [],
  };
  if (block.shape === 'bar' || block.shape === 'tile') {
    renderable.bars = new BarMesh(block, c.theme);
    renderable.pickables.push(renderable.bars.mesh);
    if (block.shape === 'bar') renderable.values = block.elements.map(() => c.labels.add('val'));
    renderable.title = c.labels.add('ttl');
  } else {
    renderable.nodes = new NodeMesh(block, c.theme);
    renderable.pickables.push(...renderable.nodes.meshes);
    renderable.values = block.elements.map(() => c.labels.add('nv'));
    renderable.title = c.labels.add('ntl');
  }
  c.popups.bindCard(renderable.title, node);
  if (block.plate !== undefined) renderable.plate = new PlateMesh(block, c.theme);
  renderable.classes = block.classLabels.map((text) => {
    const label = c.labels.add(block.role === 'board-b' || text.anchor === 'above' ? 'colh' : 'cls');
    if (block.role === 'planar' && text.anchor === 'below') label.under = renderable.values.filter((_, i) => block.elements[i].col === text.col);
    return label;
  });
  renderable.rows = block.rowLabels.map(() => c.labels.add('kk'));
  if (block.axis !== undefined) renderable.tickLabels = Array.from({ length: Math.max(renderable.ticks.forward.length, renderable.ticks.backward.length) }, () => c.labels.add('tick'));
  return renderable;
}

interface RenderableIndex {
  byKey: Map<Node | string, Renderable>;
  bySource: Map<Node, Renderable>;
}

const indexes = new WeakMap<readonly Renderable[], RenderableIndex>();

function indexOfRenderables(renderables: readonly Renderable[]): RenderableIndex {
  const known = indexes.get(renderables);
  if (known !== undefined) return known;
  const index: RenderableIndex = { byKey: new Map(), bySource: new Map() };
  for (const r of renderables) {
    const key = r.block.key ?? r.block.node;
    if (!index.byKey.has(key)) index.byKey.set(key, r);
    if (r.block.copyOf === undefined && !index.bySource.has(r.block.node)) index.bySource.set(r.block.node, r);
  }
  indexes.set(renderables, index);
  return index;
}

export function renderableOf(c: SceneController, node: Node): Renderable | undefined {
  const source = c.levelLayout?.sourceOf.get(node) ?? node;
  return indexOfRenderables(c.renderables).bySource.get(source);
}

export function renderableFor(c: SceneController, block: Block): Renderable | undefined {
  return indexOfRenderables(c.renderables).byKey.get(block.key ?? block.node);
}

export function elementOf(block: Block, node: Node, index: readonly number[]): Element {
  if (block.node === node) return elementAt(block, index);
  const mapped = sourceIndex(node, block.node, index);
  return mapped === undefined ? block.elements[offsetOf(node.shape, index as number[])] : elementAt(block, mapped);
}

export function shown(block: Block | undefined): boolean {
  return block !== undefined && block.hidden !== true;
}

export function targetOf(c: SceneController, block: Block, row: number): number | undefined {
  const label = c.aligned.get(block.node);
  return label === undefined ? undefined : targetColumn(label, row);
}

export function valuesOf(c: SceneController, node: Node): ArrayLike<number> {
  if (c.state.dir === 'forward') return node.value.data;
  const grad = node.grad;
  return grad === null ? new Float64Array(node.value.size) : grad.data;
}

export function anchorOut(c: SceneController, node: Node, index: number[], renderable = renderableOf(c, node)): Vec3 | undefined {
  if (renderable === undefined) return undefined;
  const element = elementOf(renderable.block, node, index);
  if (renderable.nodes !== undefined) return renderable.nodes.anchorOut(element);
  return barTop(element, renderable.transition.current[offsetOf(renderable.block.node.shape, element.index)]);
}

export function anchorIn(c: SceneController, node: Node, index: number[], renderable = renderableOf(c, node)): Vec3 | undefined {
  if (renderable === undefined) return undefined;
  const element = elementOf(renderable.block, node, index);
  if (renderable.nodes !== undefined) return renderable.nodes.anchorIn(element);
  return barBottom(element, renderable.transition.current[offsetOf(renderable.block.node.shape, element.index)]);
}
