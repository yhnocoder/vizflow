import type { Object3D } from 'three';
import type { Node } from '../autograd';
import { offsetOf } from '../tensor';
import { BarMesh, NodeMesh, PlateMesh, barBottom, barTop } from './blocks';
import type { SceneController } from './controller';
import type { Label } from './labels';
import { type Block, type Element, type Vec3, elementAt, elementPosition, isView, targetColumn, viewSourceIndex } from './layout';
import { TICK_COUNT, capFor, niceTicks, scaleFor } from './scales';
import { Transition } from './transition';

export interface Renderable {
  block: Block;
  node: Node;
  absent: boolean;
  offsets: Int32Array | undefined;
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

function joined(nodes: readonly Node[], data: (n: Node) => ArrayLike<number> | undefined): ArrayLike<number> {
  if (nodes.length === 1) return data(nodes[0]) ?? [];
  return nodes.flatMap((n) => Array.from(data(n) ?? []));
}

export function createRenderable(c: SceneController, block: Block): Renderable {
  const node = block.node;
  const members = block.slot ?? [node];
  const values = joined(members, (n) => n.value.data);
  const grads = joined(members, (n) => n.grad?.data);
  const cap = capFor(block.shape, node.shape.length);
  const scale = {
    forward: block.scales.forward ?? scaleFor(values, node.dtype, cap),
    backward: block.scales.backward ?? (node.phase === 'backward' ? scaleFor(values, node.dtype, cap) : scaleFor(grads, node.dtype, cap)),
  };
  const renderable: Renderable = {
    block,
    node,
    absent: false,
    offsets: memberOffsets(block, node),
    transition: new Transition(block.elements.length, () => c.reduceMotion.matches),
    clipped: block.elements.map(() => false),
    cap,
    far: false,
    seeThrough: false,
    scale,
    ticks: {
      forward: block.axis === undefined ? [] : niceTicks(values, node.dtype, scale.forward, TICK_COUNT, cap),
      backward: block.axis === undefined ? [] : node.phase === 'backward' ? niceTicks(values, node.dtype, scale.backward, TICK_COUNT, cap) : grads.length === 0 ? [] : niceTicks(grads, node.dtype, scale.backward, TICK_COUNT, cap),
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
    renderable.nodes = new NodeMesh(block);
    renderable.pickables.push(renderable.nodes.mesh);
    renderable.values = block.elements.map(() => c.labels.add('nv'));
    renderable.title = c.labels.add('ntl');
  }
  c.popups.bindCard(renderable.title, () => renderable.node);
  if (block.plate !== undefined) renderable.plate = new PlateMesh(block);
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
  const levels = c.levelLayout;
  const source = levels?.sourceOf.get(node) ?? node;
  return indexOfRenderables(c.renderables).bySource.get(levels?.leadOf.get(source) ?? source);
}

export function renderableFor(c: SceneController, block: Block): Renderable | undefined {
  return indexOfRenderables(c.renderables).byKey.get(block.key ?? block.node);
}

const memberSets = new WeakMap<Block, ReadonlySet<Node>>();

function membersOf(block: Block): ReadonlySet<Node> {
  const known = memberSets.get(block);
  if (known !== undefined) return known;
  const members = new Set(block.slot ?? [block.node]);
  memberSets.set(block, members);
  return members;
}

export function elementOf(block: Block, node: Node, index: readonly number[]): Element {
  const members = membersOf(block);
  let at = [...index];
  let n = node;
  while (!members.has(n) && isView(n)) {
    at = viewSourceIndex(n, at);
    n = n.inputs[0];
  }
  return members.has(n) ? elementAt(block, at) : block.elements[offsetOf(node.shape, index as number[])];
}

export function memberOffsets(block: Block, member: Node): Int32Array | undefined {
  const shape = member.shape;
  if (block.elements.length === member.value.size) return undefined;
  return Int32Array.from(block.elements, (e) => (e.index.every((v, d) => v < shape[d]) ? offsetOf(shape, e.index) : -1));
}

export function showMember(renderable: Renderable, member: Node | undefined): void {
  renderable.absent = member === undefined;
  const node = member ?? renderable.block.node;
  if (node === renderable.node) return;
  renderable.node = node;
  renderable.offsets = memberOffsets(renderable.block, node);
}

export function shown(block: Block | undefined): boolean {
  return block !== undefined && block.hidden !== true;
}

export function targetOf(c: SceneController, block: Block, row: number): number | undefined {
  const label = c.aligned.get(block.node);
  return label === undefined ? undefined : targetColumn(label, row);
}

export function showsGrad(c: SceneController, node: Node): boolean {
  return c.state.dir === 'backward' && node.phase === 'forward';
}

export function valuesOf(c: SceneController, node: Node): ArrayLike<number> {
  if (!showsGrad(c, node)) return node.value.data;
  const grad = node.grad;
  return grad === null ? new Float64Array(node.value.size) : grad.data;
}

export function revealedOf(c: SceneController, renderable: Renderable): boolean {
  return !renderable.absent && c.state.revealed(renderable.node);
}

export function activeOf(c: SceneController, renderable: Renderable): boolean {
  return !renderable.absent && c.state.active(renderable.node);
}

export function elementValue(values: ArrayLike<number>, renderable: Renderable, i: number): number | undefined {
  const offsets = renderable.offsets;
  if (offsets === undefined) return values[i];
  const at = offsets[i];
  return at < 0 ? undefined : values[at];
}

export function anchorOut(c: SceneController, node: Node, index: number[], renderable = renderableOf(c, node)): Vec3 | undefined {
  if (renderable === undefined) return undefined;
  const element = elementOf(renderable.block, node, index);
  if (renderable.nodes !== undefined) return renderable.nodes.anchorOut(element);
  return barTop(element, renderable.transition.current[elementPosition(renderable.block, element)]);
}

export function anchorIn(c: SceneController, node: Node, index: number[], renderable = renderableOf(c, node)): Vec3 | undefined {
  if (renderable === undefined) return undefined;
  const element = elementOf(renderable.block, node, index);
  if (renderable.nodes !== undefined) return renderable.nodes.anchorIn(element);
  return barBottom(element, renderable.transition.current[elementPosition(renderable.block, element)]);
}
