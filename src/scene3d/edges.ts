import { type Graph, type Node, moduleJacobian } from '../autograd';
import { type FlowEdge, matmulOperands } from '../explain';
import type { StepModule } from '../steps';
import type { SceneController } from './controller';
import { LONG_EDGE } from './edgelines';
import { tokenAxes } from './hover';
import { type Block, type Element, LINE_HEIGHT, type Vec3, elementAt } from './layout';
import { type Renderable, anchorIn, anchorOut, elementOf, renderableFor, renderableOf, shown } from './renderable';
import { type EdgeStyle, edgeStyle } from './style';

export const BOARD_BOW = -2.6;
export const RESIDUAL_BOW = 1.8;
export const SPLIT_JACOBIAN_BOW = 1.4;

const SAME_LEVEL = 0.05;
const JACOBIAN_EPSILON = 1e-12;
const LINK_REVEALED = 0.32;
const LINK_HIDDEN = 0.06;

export interface DrawnEdge {
  edge: FlowEdge;
  fromNode: Node;
  toNode: Node;
  contract: boolean;
  endpoints: () => [Vec3, Vec3];
  flat: boolean;
  points: number[];
  style: EdgeStyle;
  screen: number[];
  bow?: number;
  jac?: { module: StepModule; value: number };
  link?: boolean;
  backRow?: boolean;
}

type EdgeEntry = Omit<DrawnEdge, 'points' | 'style' | 'screen'>;

const crossRowCache = new WeakMap<object, Set<Node>>();
const jacobianCache = new WeakMap<Graph, { version: number; byPair: Map<string, number[][]> }>();

function cachedJacobian(g: Graph, module: StepModule, input: Node, output: Node): number[][] {
  let cache = jacobianCache.get(g);
  if (cache === undefined || cache.version !== g.version) {
    cache = { version: g.version, byPair: new Map() };
    jacobianCache.set(g, cache);
  }
  const key = `${module.id}:${input.id}:${output.id}`;
  let J = cache.byPair.get(key);
  if (J === undefined) {
    J = moduleJacobian(g, module, input, output);
    cache.byPair.set(key, J);
  }
  return J;
}

function crossRowOps(c: SceneController): Set<Node> {
  const known = crossRowCache.get(c.explainer);
  if (known !== undefined) return known;
  const axes = tokenAxes(c.g.nodes);
  const ops = new Set<Node>();
  for (const edge of c.explainer.edges()) {
    if (!('index' in edge.from) || !('index' in edge.to)) continue;
    const fromAxis = axes.get(edge.from.node);
    const toAxis = axes.get(edge.to.node);
    if (fromAxis !== undefined && toAxis !== undefined && edge.from.index[fromAxis] !== edge.to.index[toAxis]) ops.add(edge.op);
  }
  crossRowCache.set(c.explainer, ops);
  return ops;
}

export function buildEdges(c: SceneController): DrawnEdge[] {
  const drawn: DrawnEdge[] = [];
  if (c.lod) return drawn;
  const levels = c.levelLayout;
  const backward = c.state.dir === 'backward';
  const overview = c.state.dir === 'forward' && c.state.index === 0;
  const rowOf = (renderable: Renderable, index: number[]) => (renderable.block.batched ? elementAt(renderable.block, index).row : undefined);
  const push = (entry: EdgeEntry, from: Renderable, to: Renderable, rows: (number | undefined)[], zeroGrad: boolean) => {
    const style = edgeStyle({
      dir: c.state.dir,
      overview,
      contract: entry.contract,
      zeroGrad,
      focused: rows.every((row) => row === undefined || row === c.state.focus),
      far: from.far && to.far,
      current: c.stepNodes.has(backward ? entry.fromNode : entry.toNode),
      revealedFrom: c.state.revealed(entry.fromNode),
      revealedTo: c.state.revealed(entry.toNode),
    });
    if (style !== null) drawn.push({ ...entry, points: [], style, screen: [] });
  };
  const blocked = (node: Node, index: number[], target: boolean) => {
    const axis = c.positionAxis(node);
    const state = axis === undefined ? undefined : c.state.row?.(node, index[axis]);
    return state === 'hidden' || (target && state === 'inactive');
  };
  const crossRow = c.frontRowOnly ? crossRowOps(c) : new Set<Node>();
  const frontZ = new Map<Block, number>();
  const frontOf = (block: Block) => frontZ.get(block) ?? frontZ.set(block, Math.max(...block.elements.map((e) => e.z))).get(block)!;
  const isBackRow = (op: Node, fromIndex: number[], toIndex: number[], block: Block, element: Element) =>
    c.frontRowOnly && !crossRow.has(op) && fromIndex.length > 1 && toIndex.length > 1 && element.z < frontOf(block) - SAME_LEVEL;
  for (const edge of c.explainer.edges()) {
    if ('cell' in edge.from || 'cell' in edge.to) {
      const board = c.boards.find((b) => b.op === edge.op);
      if (board === undefined || edge.arg === 1) continue;
      const aR = renderableFor(c, board.a);
      const bR = renderableFor(c, board.b);
      const oR = renderableFor(c, board.out);
      if (aR === undefined || bR === undefined || oR === undefined || !shown(board.out) || !shown(board.b) || !shown(board.a)) continue;
      const index = 'cell' in edge.to ? edge.to.cell.index : (edge.from as { cell: { index: [number, number, number] } }).cell.index;
      const [r, k, i] = index;
      const operands = matmulOperands(edge.op, index);
      const y = board.y + LINE_HEIGHT;
      const tile = (block: Block, kk: number, ii: number) => block.elements.find((e) => e.row === kk && e.col === ii)!;
      const [tileW, tileD] = board.b.footprint;
      if (blocked(board.out.node, operands.out, true) || blocked(board.a.node, operands.a, false)) continue;
      const outElement = elementAt(board.out, operands.out);
      const rows = [board.out.batched ? r : undefined];
      let entry: EdgeEntry;
      if (edge.arg === 0) {
        const aElement = elementOf(board.a, edge.op.inputs[0], operands.a);
        const to: Vec3 = [tile(board.b, k, i).x - tileW / 2 - 0.01, y, tile(board.b, k, i).z];
        const previous = board.previous;
        if (i === 0 && previous !== undefined) {
          const last = tile(previous, k, previous.cols - 1);
          const from: Vec3 = [last.x + previous.footprint[0] / 2 + 0.01, y, last.z];
          entry = { edge, fromNode: board.a.node, toNode: edge.op, contract: true, endpoints: () => [from, to], flat: true };
        } else if (i === 0 && board.sharedA) {
          entry = { edge, fromNode: board.a.node, toNode: edge.op, contract: true, endpoints: () => [anchorOut(c, board.a.node, aElement.index, aR)!, to], flat: false };
        } else {
          const from: Vec3 = i === 0 ? [aElement.x, y, aElement.z - board.a.footprint[1] / 2 - 0.02] : [tile(board.b, k, i - 1).x + tileW / 2 + 0.01, y, tile(board.b, k, i - 1).z];
          entry = { edge, fromNode: board.a.node, toNode: edge.op, contract: true, endpoints: () => [from, to], flat: true };
        }
      } else {
        const last = k === board.b.rows - 1;
        const from: Vec3 = [tile(board.b, k, i).x, y, tile(board.b, k, i).z + tileD / 2 + 0.01];
        const to: Vec3 = last ? [outElement.x, y, outElement.z - board.out.footprint[1] / 2 - 0.02] : [tile(board.b, k + 1, i).x, y, tile(board.b, k + 1, i).z - tileD / 2 - 0.01];
        entry = { edge, fromNode: board.b.node, toNode: edge.op, contract: true, endpoints: () => [from, to], flat: true };
      }
      push(entry, aR, oR, rows, c.explainer.zeroGrad(edge));
      continue;
    }
    const from = edge.from;
    const to = edge.to;
    const copy = levels?.copyFor(from.node, to.node);
    const fromR = copy === undefined ? renderableOf(c, from.node) : renderableFor(c, copy);
    const toR = renderableOf(c, to.node);
    if (fromR === undefined || toR === undefined || fromR === toR || !shown(fromR.block) || !shown(toR.block)) continue;
    if (blocked(from.node, from.index, false) || blocked(to.node, to.index, true)) continue;
    const rows = [rowOf(fromR, from.index), rowOf(toR, to.index)];
    const fromElement = elementOf(fromR.block, from.node, from.index);
    const toElement = elementOf(toR.block, to.node, to.index);
    let entry: EdgeEntry;
    if (fromR.block.table !== undefined && Math.abs(fromElement.y - toElement.y) < SAME_LEVEL) {
      const y = toElement.y + LINE_HEIGHT;
      const a: Vec3 = [fromElement.x, y, fromElement.z + fromR.block.footprint[1] / 2];
      const b: Vec3 = [toElement.x, y, toElement.z - toR.block.footprint[1] / 2];
      entry = { edge, fromNode: from.node, toNode: to.node, contract: false, endpoints: () => [a, b], flat: true };
    } else {
      const long = toElement.y - fromElement.y > LONG_EDGE;
      const bow = toR.block.role === 'board-b' && long ? BOARD_BOW : to.node.kind === 'op' && to.node.op === 'add' && long ? RESIDUAL_BOW : undefined;
      entry = {
        edge,
        fromNode: from.node,
        toNode: to.node,
        contract: false,
        endpoints: () => [anchorOut(c, from.node, from.index, fromR)!, anchorIn(c, to.node, to.index, toR)!],
        flat: false,
        bow,
      };
    }
    if (isBackRow(edge.op, from.index, to.index, toR.block, toElement)) entry.backRow = true;
    push(entry, fromR, toR, rows, c.explainer.zeroGrad(edge));
  }
  if (levels === undefined) return drawn;
  for (const pair of levels.io) {
    const inR = renderableOf(c, pair.input);
    const outR = renderableFor(c, pair.outputBlock);
    if (inR === undefined || outR === undefined || !shown(inR.block) || !shown(outR.block)) continue;
    const J = cachedJacobian(c.g, pair.module, pair.input, pair.output);
    const grad = pair.output.grad;
    J.forEach((row, oi) => {
      row.forEach((value, ii) => {
        if (Math.abs(value) <= JACOBIAN_EPSILON) return;
        const fromElement = inR.block.elements[ii];
        const toElement = outR.block.elements[oi];
        if (blocked(pair.input, fromElement.index, false) || blocked(pair.output, toElement.index, true)) return;
        const edge: FlowEdge = { id: `jac:${pair.module.id}:${pair.input.id}:${pair.output.id}:${oi}:${ii}`, op: pair.output, arg: 0, from: { node: pair.input, index: fromElement.index }, to: { node: pair.output, index: toElement.index } };
        const entry = {
          edge,
          fromNode: pair.input,
          toNode: pair.output,
          contract: true,
          endpoints: (): [Vec3, Vec3] => [anchorOut(c, inR.block.node, fromElement.index, inR)!, anchorIn(c, outR.block.node, toElement.index, outR)!],
          flat: false,
          bow: pair.split ? SPLIT_JACOBIAN_BOW : 0,
          jac: { module: pair.module, value },
        };
        push(entry, inR, outR, [rowOf(inR, fromElement.index), rowOf(outR, toElement.index)], (grad === null ? 0 : grad.data[oi]) === 0);
      });
    });
  }
  for (const link of levels.links) {
    const origR = renderableFor(c, link.original);
    const copyR = renderableFor(c, link.copy);
    if (origR === undefined || copyR === undefined || !shown(link.original) || !shown(link.copy) || (origR.far && copyR.far)) continue;
    const node = link.original.node;
    const opacity = c.state.revealed(node) ? LINK_REVEALED : LINK_HIDDEN;
    link.copy.elements.forEach((element) => {
      if (blocked(node, element.index, false)) return;
      const edge: FlowEdge = { id: `link:${link.copy.key}:${element.index.join(',')}`, op: node, arg: 0, from: { node, index: element.index }, to: { node, index: element.index } };
      drawn.push({
        edge,
        fromNode: node,
        toNode: node,
        contract: false,
        endpoints: () => [anchorOut(c, node, element.index, origR)!, anchorIn(c, node, element.index, copyR)!],
        flat: false,
        link: true,
        points: [],
        style: { opacity },
        screen: [],
      });
    });
  }
  return drawn;
}
