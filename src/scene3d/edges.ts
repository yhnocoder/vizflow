import type { Graph, Node } from '../autograd';
import { moduleJacobians } from '../autograd/jacobian';
import { type CellRef, type FlowEdge, matmulOperands } from '../explain';
import type { StepModule } from '../steps';
import type { SceneController } from './controller';
import { LONG_EDGE } from './edgelines';
import { tokenAxes } from './hover';
import { indexOf } from '../tensor';
import type { LevelLayout } from './levels';
import { type Block, type BoardGeometry, type Element, LINE_HEIGHT, type Vec3, elementAt, elementInCell } from './layout';
import { type Renderable, anchorIn, anchorOut, elementOf, renderableFor, renderableIn, renderableOf, shown } from './renderable';
import { type EdgeStyle, edgeStyle } from './style';

export const BOARD_BOW = -2.6;
export const RESIDUAL_BOW = 1.8;
export const SPLIT_JACOBIAN_BOW = 1.4;
export const TABLE_ARC = 0.9;

const SAME_LEVEL = 0.05;
const JACOBIAN_EPSILON = 1e-12;

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
  backRow?: boolean;
  loop?: boolean;
  previous?: { from: Renderable; to: Renderable };
}

type EdgeEntry = Omit<DrawnEdge, 'points' | 'style' | 'screen'>;

const crossRowCache = new WeakMap<object, Set<Node>>();
const jacobianMatrices = new WeakMap<Graph, { version: number; byOutput: Map<string, Map<Node, number[][]>> }>();

function cachedJacobian(g: Graph, module: StepModule, input: Node, output: Node): number[][] {
  let cache = jacobianMatrices.get(g);
  if (cache === undefined || cache.version !== g.version) {
    cache = { version: g.version, byOutput: new Map() };
    jacobianMatrices.set(g, cache);
  }
  const key = `${module.id}:${output.id}`;
  let byInput = cache.byOutput.get(key);
  if (byInput === undefined) {
    byInput = moduleJacobians(g, module, module.inputs, output);
    cache.byOutput.set(key, byInput);
  }
  return byInput.get(input)!;
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

interface Candidate {
  entry: EdgeEntry;
  from: Renderable;
  to: Renderable;
  fromMember: Node;
  toMember: Node;
  rows: (number | undefined)[];
  zeroGrad: boolean;
  constant: boolean;
  jacobian: boolean;
}

interface CandidateSet {
  boards: readonly BoardGeometry[];
  edgesVersion: number;
  frontRow: boolean;
  candidates: Candidate[];
}

interface JacobianSet {
  levels: LevelLayout | undefined;
  version: number;
  candidates: Candidate[];
}

const candidateCache = new WeakMap<SceneController, CandidateSet>();
const jacobianCache = new WeakMap<SceneController, JacobianSet>();

function candidatesOf(c: SceneController): Candidate[] {
  const edgesVersion = c.explainer.edgesVersion();
  let known = candidateCache.get(c);
  if (known === undefined || known.boards !== c.boards || known.edgesVersion !== edgesVersion || known.frontRow !== c.frontRowOnly) {
    known = { boards: c.boards, edgesVersion, frontRow: c.frontRowOnly, candidates: collectCandidates(c) };
    candidateCache.set(c, known);
  }
  let jacobian = jacobianCache.get(c);
  if (jacobian === undefined || jacobian.levels !== c.levelLayout || jacobian.version !== c.g.version) {
    jacobian = { levels: c.levelLayout, version: c.g.version, candidates: jacobianCandidates(c) };
    jacobianCache.set(c, jacobian);
  }
  return jacobian.candidates.length === 0 ? known.candidates : [...known.candidates, ...jacobian.candidates];
}

export function edgeRound(from: { node: Node; slot?: readonly Node[] }, fromMember: Node, to: { node: Node; absent: boolean; block: { slot?: readonly Node[] } }, toMember: Node): 'same' | 'previous' | undefined {
  if (to.absent || (to.block.slot !== undefined && toMember !== to.node)) return undefined;
  if (from.slot === undefined || fromMember === from.node) return 'same';
  return from.slot.indexOf(fromMember) < from.slot.indexOf(from.node) ? 'previous' : undefined;
}

export function buildEdges(c: SceneController): DrawnEdge[] {
  const drawn: DrawnEdge[] = [];
  if (c.lod) return drawn;
  const backward = c.state.dir === 'backward';
  const overview = c.state.dir === 'forward' && c.state.kind === 'overview';
  for (const { entry, from, to, fromMember, toMember, rows, zeroGrad, constant, jacobian } of candidatesOf(c)) {
    if (backward && constant) continue;
    if ((jacobian && from.far && to.far) || from.away || to.away) continue;
    const round = edgeRound({ node: from.node, slot: from.block.slot }, fromMember, to, toMember);
    if (round === undefined || (entry.loop === true && round !== 'previous')) continue;
    const dir = entry.toNode.phase === 'backward' ? 'forward' : c.state.dir;
    const style = edgeStyle({
      dir,
      overview,
      contract: entry.contract,
      zeroGrad,
      focused: rows.every((row) => row === undefined || row === c.state.focus),
      far: from.far && to.far,
      current: c.stepNodes.has(dir === 'backward' ? entry.fromNode : entry.toNode),
      revealedFrom: c.state.revealed(entry.fromNode),
      revealedTo: c.state.revealed(entry.toNode),
    });
    if (style === null) continue;
    const edge: DrawnEdge = { ...entry, points: [], style, screen: [] };
    if (round === 'previous') edge.previous = { from, to };
    drawn.push(edge);
  }
  return drawn;
}

function collectCandidates(c: SceneController): Candidate[] {
  const candidates: Candidate[] = [];
  const rowOf = (renderable: Renderable, index: number[]) => (renderable.block.batched ? elementAt(renderable.block, index).row : undefined);
  const memberOf = (node: Node) => c.levelLayout?.sourceOf.get(node) ?? node;
  const push = (entry: EdgeEntry, from: Renderable, to: Renderable, rows: (number | undefined)[], zeroGrad: boolean, constant = false, jacobian = false) => {
    candidates.push({ entry, from, to, fromMember: memberOf(entry.fromNode), toMember: memberOf(entry.toNode), rows, zeroGrad, constant, jacobian });
  };
  const loops = new Set<string>();
  const crossRow = c.frontRowOnly ? crossRowOps(c) : new Set<Node>();
  const frontZ = new Map<Block, number>();
  const frontOf = (block: Block) => frontZ.get(block) ?? frontZ.set(block, Math.max(...block.elements.map((e) => e.z))).get(block)!;
  const isFrontRow = (block: Block, element: Element) =>
    block.rowGroups === undefined ? element.z >= frontOf(block) - SAME_LEVEL : block.rowGroups[element.row + 1] !== block.rowGroups[element.row];
  const isBackRow = (op: Node, fromIndex: number[], toIndex: number[], block: Block, element: Element) =>
    c.frontRowOnly && !crossRow.has(op) && fromIndex.length > 1 && toIndex.length > 1 && !isFrontRow(block, element);
  const boardOf = new Map<Node, BoardGeometry>();
  for (const b of c.boards) if (!boardOf.has(b.op)) boardOf.set(b.op, b);
  for (const edge of c.explainer.edges()) {
    if ('cell' in edge.from || 'cell' in edge.to) {
      const board = boardOf.get(c.levelLayout?.leadOf.get(edge.op) ?? edge.op);
      if (board === undefined || edge.arg === 1) continue;
      const aR = renderableFor(c, board.a);
      const bR = renderableFor(c, board.b);
      const oR = renderableFor(c, board.out);
      if (aR === undefined || bR === undefined || oR === undefined || !shown(board.out) || !shown(board.b) || !shown(board.a)) continue;
      const index = 'cell' in edge.to ? edge.to.cell.index : (edge.from as CellRef).cell.index;
      const [r, k, i] = index.slice(-3);
      const batch = index.slice(0, -3);
      const operands = matmulOperands(edge.op, index);
      const y = board.y + LINE_HEIGHT;
      const tile = (block: Block, kk: number, ii: number) => (block === board.b ? elementAt(block, matmulOperands(edge.op, [...batch, r, kk, ii]).b) : elementInCell(block, kk, ii)!);
      const [tileW, tileD] = board.b.footprint;
      const outElement = elementAt(board.out, operands.out);
      const aNode = memberOf(edge.op.inputs[0]);
      const rows = [board.out.batched ? r : undefined];
      let entry: EdgeEntry;
      if (edge.arg === 0) {
        const aElement = elementOf(board.a, edge.op.inputs[0], operands.a);
        const to: Vec3 = [tile(board.b, k, i).x - tileW / 2 - 0.01, y, tile(board.b, k, i).z];
        const previous = board.previous;
        if (i === 0 && previous !== undefined) {
          const last = tile(previous, k, previous.cols - 1);
          const from: Vec3 = [last.x + previous.footprint[0] / 2 + 0.01, y, last.z];
          entry = { edge, fromNode: aNode, toNode: edge.op, contract: true, endpoints: () => [from, to], flat: true };
        } else if (i === 0 && board.sharedA) {
          entry = { edge, fromNode: aNode, toNode: edge.op, contract: true, endpoints: () => [anchorOut(c, aNode, aElement.index, aR)!, to], flat: false };
        } else {
          const from: Vec3 = i === 0 ? [aElement.x, y, aElement.z - board.a.footprint[1] / 2 - 0.02] : [tile(board.b, k, i - 1).x + tileW / 2 + 0.01, y, tile(board.b, k, i - 1).z];
          entry = { edge, fromNode: aNode, toNode: edge.op, contract: true, endpoints: () => [from, to], flat: true };
        }
      } else {
        const last = k === board.b.rows - 1;
        const from: Vec3 = [tile(board.b, k, i).x, y, tile(board.b, k, i).z + tileD / 2 + 0.01];
        const to: Vec3 = last ? [outElement.x, y, outElement.z - board.out.footprint[1] / 2 - 0.02] : [tile(board.b, k + 1, i).x, y, tile(board.b, k + 1, i).z - tileD / 2 - 0.01];
        entry = { edge, fromNode: memberOf(edge.op.inputs[1]), toNode: edge.op, contract: true, endpoints: () => [from, to], flat: true };
      }
      push(entry, edge.arg === 0 ? aR : bR, oR, rows, c.explainer.zeroGrad(edge));
      continue;
    }
    const from = edge.from;
    const to = edge.to;
    const fromR = renderableIn(c, from.node, to.node);
    const toR = renderableIn(c, to.node, to.node);
    if (fromR === undefined || toR === undefined || !shown(fromR.block) || !shown(toR.block)) continue;
    const fromElement = elementOf(fromR.block, from.node, from.index);
    const toElement = elementOf(toR.block, to.node, to.index);
    const rows = [rowOf(fromR, fromElement.index), rowOf(toR, toElement.index)];
    let entry: EdgeEntry;
    if (fromR === toR) {
      const key = `${memberOf(from.node).id}:${memberOf(to.node).id}`;
      if (fromR.block.slot === undefined || memberOf(from.node) === memberOf(to.node) || loops.has(key)) continue;
      loops.add(key);
      const block = fromR.block;
      entry = { edge, fromNode: from.node, toNode: to.node, contract: false, endpoints: () => loopEnds(block), flat: false, loop: true };
    } else if (fromR.block.table !== undefined && Math.abs(fromElement.y - toElement.y) < SAME_LEVEL) {
      const y = toElement.y + LINE_HEIGHT;
      const a: Vec3 = [fromElement.x, y, fromElement.z + fromR.block.footprint[1] / 2];
      const b: Vec3 = [toElement.x, y, toElement.z - toR.block.footprint[1] / 2];
      entry = { edge, fromNode: from.node, toNode: to.node, contract: false, endpoints: () => [a, b], flat: true, bow: TABLE_ARC };
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
    push(entry, fromR, toR, rows, c.explainer.zeroGrad(edge), edge.constant === true);
  }
  return candidates;
}

export const LOOP_SPAN = 0.32;
export const LOOP_GAP = 0.08;

export function loopEnds(block: Block): [Vec3, Vec3] {
  const xs = block.elements.map((e) => e.x);
  const zs = block.elements.map((e) => e.z);
  const x = Math.max(...xs) + block.footprint[0] / 2 + LOOP_GAP;
  const y = block.elements[0].y + LINE_HEIGHT;
  const z = (Math.min(...zs) + Math.max(...zs)) / 2;
  return [[x, y, z - LOOP_SPAN], [x, y, z + LOOP_SPAN]];
}

function jacobianCandidates(c: SceneController): Candidate[] {
  const candidates: Candidate[] = [];
  const levels = c.levelLayout;
  if (levels === undefined) return candidates;
  const rowOf = (renderable: Renderable, index: number[]) => (renderable.block.batched ? elementAt(renderable.block, index).row : undefined);
  for (const pair of levels.io) {
    const inR = renderableOf(c, pair.input);
    const outR = renderableFor(c, pair.outputBlock);
    if (inR === undefined || outR === undefined || !shown(inR.block) || !shown(outR.block)) continue;
    const J = cachedJacobian(c.g, pair.module, pair.input, pair.output);
    const grad = pair.output.grad;
    J.forEach((row, oi) => {
      row.forEach((value, ii) => {
        if (Math.abs(value) <= JACOBIAN_EPSILON) return;
        const fromIndex = indexOf(pair.input.shape, ii);
        const toIndex = indexOf(pair.output.shape, oi);
        const fromElement = elementOf(inR.block, pair.input, fromIndex);
        const toElement = elementOf(outR.block, pair.output, toIndex);
        const edge: FlowEdge = { id: `jac:${pair.module.id}:${pair.input.id}:${pair.output.id}:${oi}:${ii}`, op: pair.output, arg: 0, from: { node: pair.input, index: fromIndex }, to: { node: pair.output, index: toIndex } };
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
        candidates.push({ entry, from: inR, to: outR, fromMember: inR.block.node, toMember: outR.block.node, rows: [rowOf(inR, fromElement.index), rowOf(outR, toElement.index)], zeroGrad: (grad === null ? 0 : grad.data[oi]) === 0, constant: false, jacobian: true });
      });
    });
  }
  return candidates;
}
